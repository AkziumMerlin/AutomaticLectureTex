from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import LiteratureConfig


@dataclass(frozen=True)
class LiteratureChunk:
    id: str
    source: str
    text: str


def _read_pdf(path: Path) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF literature review requires: pip install 'automatic-lecture-tex[review]'"
        ) from exc
    doc = fitz.open(path)
    return "\n".join(page.get_text("text") for page in doc)


def load_literature(config: LiteratureConfig) -> list[LiteratureChunk]:
    root = config.directory
    if not root.exists():
        return []
    chunks: list[LiteratureChunk] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".tex"}:
            text = path.read_text(encoding="utf-8", errors="replace")
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue
        text = re.sub(r"\s+", " ", text).strip()
        for offset in range(0, len(text), config.chunk_chars):
            part = text[offset : offset + config.chunk_chars]
            if part:
                chunks.append(
                    LiteratureChunk(
                        id=f"lit_{len(chunks):06d}",
                        source=str(path.relative_to(root)),
                        text=part,
                    )
                )
    return chunks


def retrieve(query: str, chunks: list[LiteratureChunk], top_k: int) -> list[LiteratureChunk]:
    query_terms = set(re.findall(r"[\wА-Яа-яЁё]{3,}", query.lower()))
    scored: list[tuple[float, LiteratureChunk]] = []
    for chunk in chunks:
        terms = set(re.findall(r"[\wА-Яа-яЁё]{3,}", chunk.text.lower()))
        if not terms:
            continue
        overlap = len(query_terms & terms)
        score = overlap / max(1, len(query_terms))
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]
