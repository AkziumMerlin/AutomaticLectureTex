from __future__ import annotations

import math
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

from .config import ASRConfig, RuntimeConfig
from .media import probe_duration
from .schemas import Transcript, TranscriptSegment, TranscriptWord
from .util import run_checked


class ASRBackend(ABC):
    def __init__(self, config: ASRConfig, runtime: RuntimeConfig) -> None:
        self.config = config
        self.runtime = runtime

    @abstractmethod
    def transcribe(self, lecture_id: str, audio_path: Path) -> Transcript:
        raise NotImplementedError


class Qwen3ASRBackend(ASRBackend):
    def __init__(self, config: ASRConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoModelForTokenClassification, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3 ASR backend requires the 'qwen-asr' extra: "
                "pip install 'automatic-lecture-tex[qwen-asr]'"
            ) from exc

        self.torch = torch
        self.AutoProcessor = AutoProcessor
        dtype = getattr(torch, config.dtype)
        self.processor = AutoProcessor.from_pretrained(config.model)
        self.model = self._load_model(AutoModelForMultimodalLM, config.model, dtype)
        self.model.eval()

        self.aligner_processor = None
        self.aligner_model = None
        if config.aligner_model:
            self.aligner_processor = AutoProcessor.from_pretrained(config.aligner_model)
            self.aligner_model = self._load_model(
                AutoModelForTokenClassification, config.aligner_model, dtype
            )
            self.aligner_model.eval()

    def _load_model(self, cls, model_id: str, dtype):
        kwargs = {"device_map": "auto", "dtype": dtype}
        try:
            return cls.from_pretrained(model_id, **kwargs)
        except TypeError:
            kwargs.pop("dtype")
            kwargs["torch_dtype"] = dtype
            return cls.from_pretrained(model_id, **kwargs)

    def _extract_chunk(self, audio_path: Path, start: float, duration: float, output: Path) -> None:
        run_checked(
            [
                self.runtime.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(audio_path),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output),
            ]
        )

    def _transcribe_chunk(self, path: Path) -> tuple[str, str | None]:
        prompt = None
        if self.config.hotwords:
            prompt = "Vocabulary: " + ", ".join(self.config.hotwords) + "."
        inputs = self.processor.apply_transcription_request(
            audio=str(path),
            language=self.config.language,
            prompt=prompt,
        ).to(self.model.device, self.model.dtype)
        with self.torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        parsed = self.processor.decode(generated_ids, return_format="parsed")[0]
        return parsed["transcription"].strip(), parsed.get("language")

    def _align_chunk(
        self,
        path: Path,
        transcript: str,
        language: str | None,
        shift: float,
    ) -> list[TranscriptWord]:
        if not transcript or self.aligner_processor is None or self.aligner_model is None:
            return []
        language = language or self.config.language or "ru"
        inputs, word_lists = self.aligner_processor.prepare_forced_aligner_inputs(
            audio=str(path), transcript=transcript, language=language
        )
        inputs = inputs.to(self.aligner_model.device, self.aligner_model.dtype)
        with self.torch.inference_mode():
            outputs = self.aligner_model(**inputs)
        timestamps = self.aligner_processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self.aligner_model.config.timestamp_token_id,
        )[0]
        return [
            TranscriptWord(
                text=item["text"],
                start=shift + float(item["start_time"]),
                end=shift + float(item["end_time"]),
            )
            for item in timestamps
        ]

    def transcribe(self, lecture_id: str, audio_path: Path) -> Transcript:
        duration = probe_duration(audio_path, self.runtime)
        chunk_seconds = self.config.chunk_seconds
        segments: list[TranscriptSegment] = []
        detected_language: str | None = self.config.language

        with tempfile.TemporaryDirectory(prefix="automatic-lecture-tex-asr-") as tmp_name:
            tmp = Path(tmp_name)
            count = math.ceil(duration / chunk_seconds)
            for index in range(count):
                start = index * chunk_seconds
                length = min(chunk_seconds, duration - start)
                chunk_path = tmp / f"chunk_{index:05d}.wav"
                self._extract_chunk(audio_path, start, length, chunk_path)
                text, language = self._transcribe_chunk(chunk_path)
                detected_language = language or detected_language
                words = self._align_chunk(chunk_path, text, language, start)
                seg_start = words[0].start if words else start
                seg_end = words[-1].end if words else start + length
                segments.append(
                    TranscriptSegment(
                        id=f"seg_{index:05d}",
                        start=seg_start,
                        end=seg_end,
                        text=text,
                        words=words,
                    )
                )

        return Transcript(lecture_id=lecture_id, language=detected_language, segments=segments)


class FasterWhisperBackend(ASRBackend):
    def __init__(self, config: ASRConfig, runtime: RuntimeConfig) -> None:
        super().__init__(config, runtime)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper backend requires: pip install 'automatic-lecture-tex[whisper]'"
            ) from exc
        self.model = WhisperModel(
            config.model,
            device=config.device,
            compute_type=config.whisper_compute_type,
        )

    def transcribe(self, lecture_id: str, audio_path: Path) -> Transcript:
        initial_prompt = None
        if self.config.hotwords:
            initial_prompt = "Vocabulary: " + ", ".join(self.config.hotwords)
        raw_segments, info = self.model.transcribe(
            str(audio_path),
            language=self.config.language,
            vad_filter=True,
            word_timestamps=True,
            initial_prompt=initial_prompt,
        )
        segments: list[TranscriptSegment] = []
        for index, segment in enumerate(raw_segments):
            confidence = None
            if getattr(segment, "avg_logprob", None) is not None:
                confidence = max(0.0, min(1.0, math.exp(float(segment.avg_logprob))))
            words = [
                TranscriptWord(text=w.word, start=float(w.start), end=float(w.end))
                for w in (segment.words or [])
                if w.start is not None and w.end is not None
            ]
            segments.append(
                TranscriptSegment(
                    id=f"seg_{index:05d}",
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                    confidence=confidence,
                    words=words,
                )
            )
        return Transcript(lecture_id=lecture_id, language=info.language, segments=segments)


def make_asr_backend(config: ASRConfig, runtime: RuntimeConfig) -> ASRBackend:
    if config.backend == "qwen3":
        return Qwen3ASRBackend(config, runtime)
    return FasterWhisperBackend(config, runtime)
