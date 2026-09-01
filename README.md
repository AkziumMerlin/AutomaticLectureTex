# AutomaticLectureTex

MVP pipeline for converting semester lecture recordings into reproducible LaTeX notes while
preserving links back to the underlying audio/video evidence.

The central design constraint is that the model does **not** write a monolithic `.tex` file directly
from a video. The pipeline produces timestamped transcription, requests video frames only when audio
is insufficient, constructs a structured `LectureIR`, and renders TeX deterministically.

## MVP data flow

```text
YouTube URL / local video
        |
        +--> audio-only extraction --> Qwen3-ASR --> optional Qwen3 Forced Aligner
        |                                  |
        |                                  v
        |                          timestamped transcript
        |                                  |
        |                          deterministic chunks
        |                                  |
        |                         visual-need selection
        |                                  |
        +---- short video windows <---------+
                    |
                3 frames/request
                    |
              local VLM/LLM
                    |
               visual evidence
                    |
                LectureIR
                    |
            deterministic renderer
                    |
                 main.tex
```

For YouTube sources the full lecture video is never persisted. `yt-dlp` downloads audio-only data for
ASR. Video is downloaded later only as short windows around selected timestamps and immediately
discarded after frame extraction. Selected frames and evidence remain in the work directory.

## Recommended H100 stack

- ASR: `Qwen/Qwen3-ASR-1.7B-hf`
- alignment: `Qwen/Qwen3-ForcedAligner-0.6B-hf`
- LLM/VLM: `Qwen/Qwen3.8-27B-FP8` served through vLLM or SGLang
- TeX: XeLaTeX via `latexmk`

The LLM/VLM layer talks only to an OpenAI-compatible local endpoint, so the main model can be changed
without changing the pipeline.

## Requirements

System binaries:

```bash
ffmpeg
ffprobe
yt-dlp
```

Python 3.11+.

Install the MVP with the Qwen ASR backend:

```bash
pip install -e '.[qwen-asr,dev]'
```

For a lower-dependency ASR fallback:

```bash
pip install -e '.[whisper,dev]'
```

## Serve the main model

On an H100:

```bash
vllm serve 'Qwen/Qwen3.8-27B-FP8' --host 127.0.0.1 --port 8000
```

The default config disables model thinking for bulk extraction calls.

## Configure a course

Copy `configs/example.yaml` and replace the lecture URLs:

```yaml
course:
  id: functional_analysis_2026
  title: Функциональный анализ
  lectures:
    - id: lecture_01
      title: Лекция 1
      source:
        type: youtube
        url: https://www.youtube.com/watch?v=...
```

A local recording is just another source backend:

```yaml
source:
  type: file
  path: ../data/lecture_01.mp4
```

Course-specific mathematical terms should be placed into `asr.hotwords`. Qwen3-ASR receives them as
context, reducing systematic transcription errors in names and terminology.

## Run

Check external dependencies:

```bash
automatic-lecture-tex doctor --config configs/course.yaml
```

Run one lecture:

```bash
automatic-lecture-tex run --config configs/course.yaml --lecture lecture_01
```

Run all configured lectures:

```bash
automatic-lecture-tex run --config configs/course.yaml
```

Re-run a lecture from source:

```bash
automatic-lecture-tex run --config configs/course.yaml --lecture lecture_01 --force
```

Render existing IR without ASR/model calls:

```bash
automatic-lecture-tex build --config configs/course.yaml
```

Optional non-destructive literature review (supports `.txt`, `.md`, `.tex` and PDFs with the
`review` extra):

```bash
automatic-lecture-tex review --config configs/course.yaml --lecture lecture_01
```

The review writes `work/<lecture>/review.json`; it does not silently patch LectureIR or TeX.

## Artifacts

For each lecture:

```text
work/<lecture>/
├── source.json
├── audio.wav
├── transcript.json
├── lecture_ir.json
├── chunks/
│   └── chunk_XXXX.json
└── frames/
    └── <visual-request>/
```

Course-level notation is accumulated in:

```text
work/course_notation.json
```

Rendered output:

```text
tex/
├── main.tex
├── lectures/
└── figures/
```

Every chunk artifact stores the transcript interval, visual requests, extracted visual evidence, and
final note blocks. This is intentionally verbose: a questionable symbol in the final TeX can be
traced back to the exact reconstruction decision instead of requiring a full rerun.

## What is deliberately not in the MVP

The current MVP does not yet implement global board-state indexing, automatic crop detection, TikZ
reconstruction, embedding-based literature retrieval, or automatic literature-based patching. The
review command intentionally emits findings only and currently uses lexical retrieval.

The next useful additions are, in order: content-change indexing for the board, embedding/reranker
retrieval for literature, and automatic crop/TikZ handling for diagrams where retaining a raw frame is
not satisfactory.
