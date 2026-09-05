# Knowledge-backed lecture reconstruction

The default notes pipeline is deliberately not `chunk -> prose -> concatenate`.
Technical transcript windows are only evidence containers.

```text
transcript + selected frames
        |
        v
 overlapping evidence windows
        |
        v
 event extraction
        |
        v
 event-sourced LectureKnowledgeBase
   - observations
   - canonical claims + revision history
   - scoped/type-hinted symbols
   - semantic anchors
        |
        v
 anchor-context outline planning
        |
        v
 section synthesis
        |
        v
 document-level validation
        |
        v
 LectureIR -> deterministic LaTeX
```

## Why overlap and a knowledge base are both needed

`notes.chunk_overlap_seconds` repeats ASR segments near a technical boundary. This prevents a proof,
definition, or lecturer correction from being split so that neither window contains enough context.
Repeated evidence is deduplicated in the knowledge base; similar but non-identical mathematics is not
silently merged by string heuristics.

Overlap alone does not solve long-range state. The knowledge updater maintains canonical claims and
scoped symbol meanings while the video is processed. An explicit later correction creates a new
claim with `supersedes=[old_claim_id]`; the earlier event remains available for provenance but is not
used as current mathematical content.

Source fidelity and mathematical correctness are separate fields. An observed lecturer statement can
remain active but mathematically suspicious. The global validator must not replace it with textbook
knowledge unless the lecture itself supplies the correction; unresolved source-faithful issues remain
visible instead.

## Semantic anchors

Technical windows never become TeX sections directly. The knowledge updater records anchors for real
transitions (definition, theorem, proof, example, notation change, explicit correction). After the
whole recording has been processed, the outline planner re-reads transcript context around those
anchors using `notes.boundary_context_seconds`. It can therefore merge a semantic unit that happened
to straddle two extraction windows.

Each final `NoteBlock` carries `source_claim_ids` and `source_evidence_ids`. These are not rendered in
TeX but allow the final document validator and debugging tools to trace a block back to the knowledge
state and raw evidence.

## Artifacts

A knowledge-mode run writes:

```text
work/<lecture>/
├── transcript.json
├── lecture_kb.json
├── lecture_outline.json
├── global_validation.json
├── lecture_ir.json
├── run_metrics.json
├── knowledge_windows/
│   └── window_XXXX.json
├── knowledge_sections/
│   └── section_XXX.json
└── frames/
```

Window artifacts contain visual evidence, extracted observations, and the knowledge delta. They are
fingerprinted against the compact state before that window, which makes interrupted runs resumable
without accepting a stale downstream knowledge update after an upstream change.

## Main controls

```yaml
notes:
  architecture: knowledge
  chunk_target_seconds: 480
  chunk_overlap_seconds: 120
  boundary_context_seconds: 120
  knowledge_max_active_claims: 160
  knowledge_recent_observations: 80
  max_outline_sections: 40
  global_validation: true
  global_validation_apply_threshold: 0.85
```

`architecture: legacy` keeps the previous sequential `finalize_chunk` path for regression testing and
comparison. It uses non-overlapping chunks and writes the older `chunks/chunk_XXXX.json` artifacts.
