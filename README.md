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
        +--> audio-only extraction --> Whisper large-v3 / Qwen3-ASR
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
              multimodal Qwen
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

## Recommended A100/H100 stack

- default ASR profile: `faster-whisper` with `large-v3`, Silero VAD, word timestamps, and
  confidence-driven visual recovery
- alternative ASR: the official `qwen-asr` toolkit with `Qwen/Qwen3-ASR-1.7B` and
  `Qwen/Qwen3-ForcedAligner-0.6B`
- LLM/VLM: `Qwen/Qwen3.8-27B-FP8` served through vLLM or SGLang
- TeX: XeLaTeX via `latexmk`

The LLM/VLM layer talks only to an OpenAI-compatible local endpoint, so the main model can be changed
without changing the pipeline. Qwen3.8 is used as a true multimodal model here; a separate VLM is not
required when the served checkpoint contains its vision encoder.

## Requirements

System binaries:

```bash
ffmpeg
ffprobe
yt-dlp
```

Python 3.11+.

Install the recommended profile:

```bash
pip install -e '.[whisper,dev]'
```

Install the official Qwen3-ASR backend instead:

```bash
pip install -e '.[qwen-asr,dev]'
```

The old Transformers `-hf` implementation remains available as `backend: qwen3_hf` through the
`qwen-asr-hf` extra for compatibility. New configurations should use `qwen3` with the official
non-`-hf` model IDs.

## Serve the main model

On a shared A100/H100, use the included launch profile:

```bash
conda activate qwen35
./scripts/serve_llm.sh
```

It reserves 65% of GPU memory, uses a 32k context window, and allows four concurrent sequences. This
leaves room for the ASR model; override any setting with the `LECTURE_TEX_LLM_*` environment
variables documented in the script. The pipeline disables model thinking and uses deterministic
decoding for extraction calls. `vision.max_workers` controls concurrent visual requests (three by
default), so `--max-num-seqs` should be at least that value.

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
context; faster-whisper receives them through its dedicated hotword control. Hotwords are never used
as an initial transcript prompt, and prompt-like echoes are filtered before note generation.

The shipped course configs use `faster_whisper`/`large-v3`, because on the included 81.7-minute
Russian functional-analysis recording it was faster and produced substantially better short,
timestamped segments than the previous fixed-minute Qwen `-hf` loop. Mathematical expressions still
require video: low-confidence speech segments automatically request nearby board frames.

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

Normal reruns are incremental. Transcript, final IR, individual note chunks, and extracted frames are
fingerprinted and reused. If a run stops halfway through, the next invocation resumes from completed
chunks. `--force` intentionally bypasses these caches.

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
├── run_metrics.json
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

Visual reconstruction has two layers: `raw_latex` records the literal board OCR, while `latex`
contains the mathematically interpreted version. The model is allowed to repair ASR/OCR errors and
complete a well-supported short derivation. Every content-changing edit is stored in `corrections`
with its original text, replacement, basis, reason, and confidence. Corrections are also emitted as
comments in the rendered lecture `.tex`; unresolved ambiguity remains in `unresolved`.

Formula-dense chunks receive a second, narrow mathematical audit. The reviewer recomputes signs,
scalar factors, domains, and implications; it changes only blocks with a concrete error and records
the full before/after replacement. `llm.math_audit_min_equals` controls the inexpensive trigger, and
`llm.math_audit: false` disables this quality pass when maximum throughput is more important.

Use a separate `runtime.work_dir` and `latex.output_dir` per course. The shipped configs already do
this; otherwise generic lecture IDs such as `lecture_01` from different courses would collide.

## Quality and performance controls

- ASR uses VAD instead of hard one-minute cuts, preserves word timestamps, records segment
  confidence, and passes low-confidence intervals to visual recovery.
- Prompts contain timestamped transcript lines, so visual requests are grounded in the utterance that
  triggered them. Out-of-range model timestamps are discarded and request IDs are path-safe and
  unique per chunk.
- Rule-based visual selection is the default. It captures explicit board/diagram references and ASR
  uncertainty without paying for an extra LLM call on every chunk. Set `visual_llm_selector: true`
  only when recall matters more than throughput.
- Text-only LLM calls use server-side JSON-schema decoding. Multimodal calls use strict JSON prompts
  plus local schema validation because guided decoding in some vLLM versions can collapse valid
  image answers to schema defaults. Both paths use compact prompts, bounded outputs, and retries.
- Independent visual requests inside a chunk are resolved concurrently, while final note chunks stay
  ordered so the course notation registry remains deterministic.
- `run_metrics.json` reports ASR, visual, finalization/audit, and total times together with processed
  chunks, cache hits, successful visual evidence, corrections, and unresolved counts.

On the first ten minutes of the shipped functional-analysis lecture, Whisper large-v3 transcribed
the audio in 60--90 seconds on an A100 80 GB. After fixing multimodal JSON handling, all seven
selected visual intervals produced board evidence; before the fix every visual result was empty.
The note stage took 101 seconds without the mathematical audit and 222 seconds with the final
quality profile on the existing server limited to one concurrent sequence. Exact time varies with
other jobs sharing the GPU; the included four-sequence launch profile allows the independent visual
calls to overlap. A cache-hit rerun takes about 0.3 seconds because normalized audio, transcript,
frames, and completed chunks are reused.

## What is deliberately not in the MVP

The current MVP does not yet implement global board-state indexing, automatic crop detection, TikZ
reconstruction, embedding-based literature retrieval, or automatic literature-based patching. The
review command intentionally emits findings only and currently uses lexical retrieval.

The next useful additions are, in order: content-change indexing for the board, embedding/reranker
retrieval for literature, and automatic crop/TikZ handling for diagrams where retaining a raw frame is
not satisfactory.
