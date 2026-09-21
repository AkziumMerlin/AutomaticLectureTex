from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import OmniConfig


class Qwen25OmniEvidenceBackend:
    """Native audio-video sensory pass for one short lecture interval.

    The backend intentionally does not emit canonical lecture events. Its text is supplemental
    evidence consumed by the stronger semantic reconstruction model together with raw ASR and
    directly attached board images.
    """

    def __init__(self, config: OmniConfig, *, output_language: str) -> None:
        try:
            import torch
            from qwen_omni_utils import process_mm_info
            from transformers import (
                Qwen2_5OmniProcessor,
                Qwen2_5OmniThinkerForConditionalGeneration,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Qwen2.5-Omni evidence requires the optional dependencies. "
                "Install with: pip install -e '.[qwen-omni]'"
            ) from exc

        self._torch = torch
        self._process_mm_info = process_mm_info
        self.config = config
        self.output_language = output_language

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        model_kwargs = {
            "torch_dtype": dtype_map[config.dtype],
            "device_map": config.device_map,
        }
        if config.attn_implementation != "auto":
            model_kwargs["attn_implementation"] = config.attn_implementation

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            config.model,
            **model_kwargs,
        )
        self.model.eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            config.model,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )

    def analyze(self, clip_path: Path) -> str:
        clip = clip_path.resolve()
        if not clip.is_file() or clip.stat().st_size == 0:
            raise FileNotFoundError(clip)

        prompt = f"""Act as a literal audiovisual sensor for a short university mathematics lecture
interval. Inspect BOTH the spoken audio and the evolving board/video.

Return compact evidence in language code {self.output_language}. Do not solve the mathematics and do
not repair the lecturer using textbook knowledge. Do not infer theorem names or omitted proof steps
unless they are actually spoken or visibly written.

Report only:
1. SPEECH — the mathematical content you can actually hear; preserve uncertainty when words are
   unclear.
2. BOARD — exact visible mathematical notation/formulas/text when readable; preserve signs,
   subscripts, quantifiers and variable names literally.
3. CROSS_MODAL — explicit temporal links between speech/deictic references and what is being
   written, pointed at, erased, or changed on the board.
4. UNCERTAIN — ambiguities or sensor conflicts.

This output will be treated as fallible auxiliary evidence by another reconstruction model, so prefer
literal observation over explanation."""

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(clip),
                        "fps": self.config.fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = self._process_mm_info(
            conversation,
            use_audio_in_video=True,
        )
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                use_audio_in_video=True,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )

        # Standard HF generate() normally returns prompt+completion. Keep compatibility with custom
        # generation implementations that already return completion-only ids.
        input_ids = inputs.get("input_ids")
        if input_ids is not None and generated.ndim == 2 and generated.shape[1] > input_ids.shape[1]:
            prefix = generated[:, : input_ids.shape[1]]
            if self._torch.equal(prefix, input_ids):
                generated = generated[:, input_ids.shape[1] :]

        decoded = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (decoded[0] if decoded else "").strip()


def make_omni_backend(config: OmniConfig, *, output_language: str):
    if not config.enabled:
        return None
    if config.backend == "qwen2_5_omni":
        return Qwen25OmniEvidenceBackend(config, output_language=output_language)
    raise ValueError(f"unsupported omni backend: {config.backend}")
