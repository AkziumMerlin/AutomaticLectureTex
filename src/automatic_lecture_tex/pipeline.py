from __future__ import annotations

import json
from pathlib import Path

from .asr import make_asr_backend
from .chunking import chunk_transcript
from .config import AppConfig, LectureConfig
from .latex import compile_tex, write_course_tex
from .literature import load_literature, retrieve
from .llm import LectureModelClient
from .media import copy_asset, media_source_from_config
from .schemas import LectureIR, ReviewFinding, ReviewReport, Transcript, VisualEvidence
from .util import atomic_json_dump, stable_hash
from .vision import dedupe_visual_requests, select_rule_based_visual_requests


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

    def run_lecture(self, lecture: LectureConfig, *, force: bool = False) -> LectureIR:
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
        atomic_json_dump(work / "source.json", source_identity)

        transcript_fingerprint = stable_hash(
            {
                "source": source_identity,
                "asr": self.config.asr.model_dump(mode="json"),
            }
        )
        transcript_valid = (
            transcript_path.exists()
            and manifest.get("transcript_fingerprint") == transcript_fingerprint
            and not force
        )
        if transcript_valid:
            transcript = self._load_transcript(transcript_path)
        else:
            source.prepare_audio(audio_path)
            transcript = self.asr.transcribe(lecture.id, audio_path)
            atomic_json_dump(transcript_path, transcript.model_dump(mode="json"))
            manifest["transcript_fingerprint"] = transcript_fingerprint
            manifest.pop("ir_fingerprint", None)
            atomic_json_dump(manifest_path, manifest)

        notation = self._load_notation_registry()
        ir_fingerprint = stable_hash(
            {
                "transcript": transcript.model_dump(mode="json"),
                "notes": self.config.notes.model_dump(mode="json"),
                "vision": self.config.vision.model_dump(mode="json"),
                "llm": self.config.llm.model_dump(mode="json"),
                "known_notation": notation,
            }
        )
        if (
            ir_path.exists()
            and manifest.get("ir_fingerprint") == ir_fingerprint
            and not force
        ):
            return self._load_ir(ir_path)

        chunks = chunk_transcript(transcript, self.config.notes.chunk_target_seconds)
        note_chunks = []
        figures_root = self.config.latex.output_dir / "figures" / lecture.id

        for chunk in chunks:
            requests = []
            if self.config.notes.visual_rule_selector:
                requests.extend(select_rule_based_visual_requests(chunk, transcript))
            if self.config.notes.visual_llm_selector:
                requests.extend(self.llm.analyze_chunk(chunk, notation).visual_requests)
            requests = dedupe_visual_requests(
                requests,
                within_seconds=self.config.notes.visual_dedupe_seconds,
                limit=self.config.vision.max_requests_per_chunk,
            )

            evidence: list[VisualEvidence] = []
            for request in requests:
                frame_times = [
                    max(0.0, request.timestamp + offset)
                    for offset in self.config.vision.frame_offsets_seconds
                ]
                frame_dir = work / "frames" / request.id
                frames = source.extract_frames(frame_times, frame_dir)
                visual = self.llm.resolve_visual_request(
                    request, chunk, [frame.path for frame in frames]
                )
                if visual.requires_figure_in_notes and frames:
                    index = visual.best_frame_index if visual.best_frame_index is not None else 0
                    index = max(0, min(index, len(frames) - 1))
                    destination = figures_root / f"{request.id}.jpg"
                    copy_asset(frames[index].path, destination)
                    visual.asset_path = str(destination.relative_to(self.config.latex.output_dir))
                evidence.append(visual)

            notes = self.llm.finalize_chunk(chunk, evidence, notation)
            note_chunks.append(notes)
            for item in notes.notation:
                notation.setdefault(item.latex, item.meaning)

            chunk_artifact = work / "chunks" / f"{chunk.id}.json"
            atomic_json_dump(
                chunk_artifact,
                {
                    "chunk": chunk.model_dump(mode="json"),
                    "visual_requests": [r.model_dump(mode="json") for r in requests],
                    "visual_evidence": [e.model_dump(mode="json") for e in evidence],
                    "notes": notes.model_dump(mode="json"),
                },
            )

        ir = LectureIR(
            lecture_id=lecture.id,
            title=lecture.title or lecture.id,
            chunks=note_chunks,
        )
        atomic_json_dump(ir_path, ir.model_dump(mode="json"))
        self._save_notation_registry(notation)
        manifest["ir_fingerprint"] = ir_fingerprint
        atomic_json_dump(manifest_path, manifest)
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
                            {"id": hit.id, "source": hit.source, "text": hit.text}
                            for hit in hits
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
