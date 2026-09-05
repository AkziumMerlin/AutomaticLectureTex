from __future__ import annotations

import json
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from pydantic import ValidationError

from .asr import make_asr_backend
from .chunking import chunk_transcript
from .config import AppConfig, LectureConfig
from .knowledge_pipeline import run_knowledge_pipeline
from .latex import compile_tex, write_course_tex
from .literature import load_literature, retrieve
from .llm import LectureModelClient
from .media import copy_asset, media_source_from_config
from .schemas import ChunkNotes, LectureIR, ReviewFinding, ReviewReport, Transcript, VisualEvidence
from .util import atomic_json_dump, stable_hash
from .vision import (
    dedupe_visual_requests,
    namespace_visual_requests,
    select_rule_based_visual_requests,
)

logger = logging.getLogger(__name__)

ASR_CACHE_VERSION = 2
NOTES_CACHE_VERSION = 7


class Pipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._asr = None
        self._llm = None

    @property
    def asr(self):
        if self._asr is None:
            self._asr = make_asr_backend(self.config.asr, self.config.runtime)
        return self._asr

    @property
    def llm(self) -> LectureModelClient:
        if self._llm is None:
            self._llm = LectureModelClient(self.config.llm)
        return self._llm

    def _lecture_work_dir(self, lecture: LectureConfig) -> Path:
        path = self.config.runtime.work_dir / lecture.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_notation_registry(self) -> dict[str, str]:
        path = self.config.runtime.work_dir / "course_notation.json"
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in payload.items()}

    def _save_notation_registry(self, registry: dict[str, str]) -> None:
        atomic_json_dump(self.config.runtime.work_dir / "course_notation.json", registry)

    def _load_transcript(self, path: Path) -> Transcript:
        return Transcript.model_validate_json(path.read_text(encoding="utf-8"))

    def _load_ir(self, path: Path) -> LectureIR:
        return LectureIR.model_validate_json(path.read_text(encoding="utf-8"))

    def _ir_fingerprint(self, transcript: Transcript, notation: dict[str, str]) -> str:
        return stable_hash(
            {
                "transcript": transcript.model_dump(mode="json"),
                "notes": self.config.notes.model_dump(mode="json"),
                "vision": self.config.vision.model_dump(mode="json"),
                "llm": self.config.llm.model_dump(mode="json"),
                "known_notation": notation,
                "notes_cache_version": NOTES_CACHE_VERSION,
            }
        )

    def run_lecture(self, lecture: LectureConfig, *, force: bool = False) -> LectureIR:
        run_started = time.perf_counter()
        work = self._lecture_work_dir(lecture)
        transcript_path = work / "transcript.json"
        ir_path = work / "lecture_ir.json"
        audio_path = work / "audio.wav"
        manifest_path = work / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        source = media_source_from_config(lecture.source, self.config.runtime, self.config.vision)
        source_identity = source.identity()
        source_path = work / "source.json"
        previous_source_identity = None
        if source_path.exists():
            try:
                previous_source_identity = json.loads(source_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        atomic_json_dump(source_path, source_identity)

        audio_fingerprint = stable_hash(
            {"source": source_identity, "audio_format": "pcm_s16le-16k-mono-v1"}
        )
        audio_valid = (
            audio_path.is_file()
            and audio_path.stat().st_size > 0
            and not force
            and (
                manifest.get("audio_fingerprint") == audio_fingerprint
                or previous_source_identity == source_identity
            )
        )

        transcript_fingerprint = stable_hash(
            {
                "source": source_identity,
                "asr": self.config.asr.model_dump(mode="json"),
                "asr_cache_version": ASR_CACHE_VERSION,
            }
        )
        transcript_valid = (
            transcript_path.exists()
            and manifest.get("transcript_fingerprint") == transcript_fingerprint
            and not force
        )
        if transcript_valid:
            logger.info("[%s] reusing transcript", lecture.id)
            transcript = self._load_transcript(transcript_path)
            asr_seconds = 0.0
            media_seconds = 0.0
        else:
            if audio_valid:
                logger.info("[%s] reusing normalized audio", lecture.id)
                media_seconds = 0.0
            else:
                logger.info("[%s] preparing normalized audio", lecture.id)
                media_started = time.perf_counter()
                source.prepare_audio(audio_path)
                media_seconds = time.perf_counter() - media_started
            manifest["audio_fingerprint"] = audio_fingerprint
            logger.info("[%s] running %s ASR", lecture.id, self.config.asr.backend)
            asr_started = time.perf_counter()
            transcript = self.asr.transcribe(lecture.id, audio_path)
            asr_seconds = time.perf_counter() - asr_started
            atomic_json_dump(transcript_path, transcript.model_dump(mode="json"))
            manifest["transcript_fingerprint"] = transcript_fingerprint
            manifest.pop("ir_fingerprint", None)
            atomic_json_dump(manifest_path, manifest)

        notation = self._load_notation_registry()
        ir_fingerprint = self._ir_fingerprint(transcript, notation)
        if ir_path.exists() and manifest.get("ir_fingerprint") == ir_fingerprint and not force:
            logger.info("[%s] LectureIR cache hit", lecture.id)
            return self._load_ir(ir_path)

        if self.config.notes.architecture == "knowledge":
            return run_knowledge_pipeline(
                self,
                lecture=lecture,
                transcript=transcript,
                source=source,
                source_identity=source_identity,
                work=work,
                ir_path=ir_path,
                manifest_path=manifest_path,
                manifest=manifest,
                notation=notation,
                media_seconds=media_seconds,
                asr_seconds=asr_seconds,
                run_started=run_started,
                force=force,
            )

        self.llm.reset_usage()
        notes_started = time.perf_counter()
        chunks = chunk_transcript(transcript, self.config.notes.chunk_target_seconds)
        note_chunks = []
        figures_root = self.config.latex.output_dir / "figures" / lecture.id
        chunk_cache_hits = 0
        processed_chunks = 0
        vision_seconds = 0.0
        finalize_seconds = 0.0
        visual_requests_processed = 0
        visual_evidence_successful = 0
        chunk_llm_usages: list[dict] = []

        for chunk in chunks:
            chunk_artifact = work / "chunks" / f"{chunk.id}.json"
            previous_notes = note_chunks[-1] if note_chunks else None
            chunk_fingerprint = stable_hash(
                {
                    "chunk": chunk.model_dump(mode="json"),
                    "previous_notes": (
                        previous_notes.model_dump(mode="json")
                        if previous_notes is not None
                        else None
                    ),
                    "source": source_identity,
                    "notes": self.config.notes.model_dump(mode="json"),
                    "vision": self.config.vision.model_dump(mode="json"),
                    "llm": self.config.llm.model_dump(mode="json"),
                    "known_notation": notation,
                    "notes_cache_version": NOTES_CACHE_VERSION,
                }
            )
            if chunk_artifact.exists() and not force:
                try:
                    payload = json.loads(chunk_artifact.read_text(encoding="utf-8"))
                    if payload.get("fingerprint") == chunk_fingerprint:
                        notes = ChunkNotes.model_validate(payload["notes"])
                        note_chunks.append(notes)
                        for item in notes.notation:
                            notation.setdefault(item.latex, item.meaning)
                        chunk_llm_usages.append(payload.get("llm_usage", {}))
                        chunk_cache_hits += 1
                        logger.info("[%s] %s cache hit", lecture.id, chunk.id)
                        continue
                except (json.JSONDecodeError, KeyError, ValidationError) as exc:
                    logger.warning("[%s] ignoring invalid %s: %s", lecture.id, chunk_artifact, exc)

            logger.info(
                "[%s] processing %s (%d/%d)",
                lecture.id,
                chunk.id,
                processed_chunks + chunk_cache_hits + 1,
                len(chunks),
            )
            usage_before = self.llm.usage_snapshot()
            requests = []
            if self.config.notes.visual_rule_selector:
                requests.extend(
                    select_rule_based_visual_requests(
                        chunk,
                        transcript,
                        self.config.notes.max_low_confidence_visual_requests,
                    )
                )
            if self.config.notes.visual_llm_selector:
                requests.extend(self.llm.analyze_chunk(chunk, notation).visual_requests)
            requests = dedupe_visual_requests(
                requests,
                within_seconds=self.config.notes.visual_dedupe_seconds,
                limit=self.config.vision.max_requests_per_chunk,
            )
            requests = namespace_visual_requests(chunk.id, requests)

            vision_started = time.perf_counter()
            prepared_visuals = []
            for request in requests:
                frame_times = [
                    max(0.0, request.timestamp + offset)
                    for offset in self.config.vision.frame_offsets_seconds
                ]
                frame_dir = work / "frames" / request.id
                frames = source.extract_frames(frame_times, frame_dir)
                prepared_visuals.append((request, frames))

            evidence: list[VisualEvidence] = []
            workers = min(self.config.vision.max_workers, len(prepared_visuals))
            futures: list[Future[VisualEvidence]] = []
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                for request, frames in prepared_visuals:
                    futures.append(
                        executor.submit(
                            self.llm.resolve_visual_request,
                            request,
                            chunk,
                            [frame.path for frame in frames],
                            [frame.timestamp for frame in frames],
                        )
                    )
                for (request, frames), future in zip(prepared_visuals, futures, strict=True):
                    try:
                        visual = future.result()
                    except Exception as exc:
                        logger.warning(
                            "[%s] visual OCR failed for %s: %s",
                            lecture.id,
                            request.id,
                            exc,
                        )
                        visual = VisualEvidence(
                            request_id=request.id,
                            description=(
                                f"Visual OCR failed after retries: {type(exc).__name__}: {exc}"
                            ),
                        )
                    if visual.requires_figure_in_notes and frames:
                        index = (
                            visual.best_frame_index if visual.best_frame_index is not None else 0
                        )
                        index = max(0, min(index, len(frames) - 1))
                        destination = figures_root / f"{request.id}.jpg"
                        copy_asset(frames[index].path, destination)
                        visual.asset_path = str(
                            destination.relative_to(self.config.latex.output_dir)
                        )
                    evidence.append(visual)
            vision_seconds += time.perf_counter() - vision_started
            visual_requests_processed += len(requests)
            visual_evidence_successful += sum(
                item.kind != "none" and item.confidence >= 0.75 for item in evidence
            )

            finalize_started = time.perf_counter()
            notes = self.llm.finalize_chunk(chunk, evidence, notation, previous_notes)
            finalize_seconds += time.perf_counter() - finalize_started
            chunk_usage = LectureModelClient.usage_delta(
                self.llm.usage_snapshot(),
                usage_before,
            )
            chunk_llm_usages.append(chunk_usage)
            note_chunks.append(notes)
            processed_chunks += 1
            for item in notes.notation:
                notation.setdefault(item.latex, item.meaning)

            atomic_json_dump(
                chunk_artifact,
                {
                    "fingerprint": chunk_fingerprint,
                    "chunk": chunk.model_dump(mode="json"),
                    "visual_requests": [r.model_dump(mode="json") for r in requests],
                    "visual_evidence": [e.model_dump(mode="json") for e in evidence],
                    "notes": notes.model_dump(mode="json"),
                    "llm_usage": chunk_usage,
                },
            )

        ir = LectureIR(
            lecture_id=lecture.id,
            title=lecture.title or lecture.id,
            chunks=note_chunks,
        )
        atomic_json_dump(ir_path, ir.model_dump(mode="json"))
        self._save_notation_registry(notation)
        manifest["ir_fingerprint"] = self._ir_fingerprint(transcript, notation)
        atomic_json_dump(manifest_path, manifest)
        atomic_json_dump(
            work / "run_metrics.json",
            {
                "lecture_id": lecture.id,
                "architecture": "legacy",
                "media_seconds": round(media_seconds, 3),
                "asr_seconds": round(asr_seconds, 3),
                "notes_seconds": round(time.perf_counter() - notes_started, 3),
                "vision_seconds": round(vision_seconds, 3),
                "finalize_and_math_audit_seconds": round(finalize_seconds, 3),
                "total_seconds": round(time.perf_counter() - run_started, 3),
                "chunks_total": len(chunks),
                "chunks_processed": processed_chunks,
                "chunk_cache_hits": chunk_cache_hits,
                "visual_requests_processed": visual_requests_processed,
                "visual_evidence_successful": visual_evidence_successful,
                "corrections_total": sum(len(notes.corrections) for notes in note_chunks),
                "unresolved_total": sum(len(notes.unresolved) for notes in note_chunks),
                "llm_usage": LectureModelClient.combine_usage(chunk_llm_usages),
            },
        )
        return ir

    def run(self, lecture_id: str | None = None, *, force: bool = False) -> list[LectureIR]:
        selected = self.config.course.lectures
        if lecture_id is not None:
            selected = [lecture for lecture in selected if lecture.id == lecture_id]
            if not selected:
                raise ValueError(f"unknown lecture id: {lecture_id}")
        irs = [self.run_lecture(lecture, force=force) for lecture in selected]
        self.build()
        return irs

    def review(self, lecture_id: str | None = None) -> list[ReviewReport]:
        literature = load_literature(self.config.literature)
        if not literature:
            raise RuntimeError(
                f"no supported literature files found under {self.config.literature.directory}"
            )
        selected = self.config.course.lectures
        if lecture_id is not None:
            selected = [lecture for lecture in selected if lecture.id == lecture_id]
            if not selected:
                raise ValueError(f"unknown lecture id: {lecture_id}")

        reports: list[ReviewReport] = []
        for lecture in selected:
            ir_path = self._lecture_work_dir(lecture) / "lecture_ir.json"
            if not ir_path.exists():
                continue
            ir = self._load_ir(ir_path)
            findings: list[ReviewFinding] = []
            block_index = 0
            for chunk in ir.chunks:
                for block in chunk.blocks:
                    hits = retrieve(
                        block.latex,
                        literature,
                        self.config.literature.retrieval_top_k,
                    )
                    if hits:
                        excerpts = [
                            {"id": hit.id, "source": hit.source, "text": hit.text} for hit in hits
                        ]
                        decision = self.llm.review_block(block, excerpts)
                        findings.append(
                            ReviewFinding(
                                block_index=block_index,
                                status=decision.status,
                                issue=decision.issue,
                                suggested_patch=decision.suggested_patch,
                                source_excerpt_ids=decision.source_excerpt_ids,
                            )
                        )
                    block_index += 1
            report = ReviewReport(lecture_id=lecture.id, findings=findings)
            atomic_json_dump(
                self._lecture_work_dir(lecture) / "review.json",
                report.model_dump(mode="json"),
            )
            reports.append(report)
        return reports

    def build(self) -> Path:
        irs: list[LectureIR] = []
        for lecture in self.config.course.lectures:
            path = self._lecture_work_dir(lecture) / "lecture_ir.json"
            if path.exists():
                irs.append(self._load_ir(path))
        if not irs:
            raise RuntimeError("no LectureIR artifacts found; run at least one lecture first")
        main = write_course_tex(self.config.course.title, irs, self.config.latex.output_dir)
        if self.config.latex.compile:
            compile_tex(main, self.config.latex.compiler)
        return main
