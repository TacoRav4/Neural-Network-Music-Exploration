# Phase 3F.6 — Clementi Sonatina Op. 36 No. 1 Level 5 QA

**Metadata QA only. No tonal-inference evaluation (chroma extraction, pitch-class baseline, chord-id EMA/SRN, uncertainty diagnostics, or disagreement analysis) has been run on this candidate. No existing scripts, notebooks, or Phase 1/1.5/2/3 outputs have been modified.**

## 1. File discovered

`03_MIDI_Data/candidate_intermediate_midi/clementi_opus36_1_1.mid` — found already present in the candidate directory (moved there by the user prior to this task, per the task's context). Filename not changed.

## 2. QA metadata (via `pretty_midi`)

| Field | Value |
|---|---|
| Load status | **OK** (no errors) |
| Duration | 90.47 sec |
| Instruments | 2 — "Piano right" (program 0, 454 notes), "Piano left" (program 0, 212 notes) |
| Total notes | 666 |
| Notes/sec | 7.36 |
| Max polyphony (approx.) | 4 |
| Key signature changes | **1** — C Major at t=0.0s (no further key-signature events embedded) |
| Time signature changes | 1 — 4/4 at t=0.0s |
| Tempo changes | **686 discrete events** (bpm range 131.2–210.8, mean 202.5) — this is an expressive/humanized performance MIDI with continuous micro-rubato, not a flat-tempo quantized score. Noted here as context, not a QA warning category. |
| Warnings triggered | **none** |
| QA recommendation | **pass** |

Notes/sec (7.36) sits close to, but under, the "very dense" warning threshold (8.0) — worth keeping in mind, though it did not trigger the flag.

## 3. Does it pass as a Level 5 candidate?

**Yes — pass, no warnings.** The file loads cleanly, is short (under the 300s long-file threshold), has a manageable instrument count (2, under the many-instruments threshold of 3), and is not flagged dense. It fills the Level 5 slot in the corpus ladder (see section 4), which was previously explicitly marked missing in `PHASE3F_INTERMEDIATE_MIDI_CORPUS_REGISTRY.md`.

## 4. Embedded key signatures

**Only one embedded key-signature event exists: C Major at t=0.0s.** There is no embedded MIDI event marking the expected modulation to G major. This means the G-major target anchor for this piece is **manual/score-based**, not MIDI-metadata-derived — the same convention already used for `Twinkle.mid`'s assumed C-major tonic (which also has zero embedded key-signature events), and distinct from `Twinkle 12.mid`, which is the only piece in this workspace with real embedded modulation metadata.

## 5. Recommended scope

**Exposition only, no repeat** — per the intended Level 5 role. The file **as provided is the full first movement** (90.47s), not pre-cut to the exposition alone. An exact excerpt boundary has not been determined in this task (see section 6) — do not assume `[0.0, 90.47]` (full movement) is the intended evaluation scope; it is not.

**Warning: the exact exposition/no-repeat cut boundary is still TBD.** No excerpt file has been created. `exact_excerpt_start_sec = 0.0` is recorded in the registry as a reasonable assumption (the exposition begins at the start of the movement), but `exact_excerpt_end_sec` remains unset pending the analysis in section 6 and, ideally, manual confirmation.

## 6. Excerpt boundary recommendation (preliminary, metadata-only)

A lightweight, metadata-only gap analysis was run: all note onsets across both hands were pooled and sorted, and the largest inter-onset gaps (candidate rests / phrase or section boundaries) were identified:

| Time of gap | Gap duration |
|---|---|
| ~17.26s | 0.647s |
| ~35.16s | 0.651s |
| ~62.56s | 0.651s |

Three notably large, similarly-sized gaps (~0.65s, well above the typical ~0.3s gap size seen elsewhere) stand out. The first two are roughly in a 2:1 time ratio (17.26s vs. 35.16s ≈ 2×), which is at least **consistent with** — though not proof of — the possibility that the exposition is played once (ending ≈17.3s) and then repeated (ending ≈35.2s), with the gap at ≈62.6s marking a later formal boundary (e.g. development/recapitulation).

**This is a preliminary, data-informed hypothesis only, not a determination.** If it holds, the no-repeat exposition-only excerpt would be approximately **[0.0s, 17.3s]**. Before cutting any excerpt file:

- **Recommend a manual listening pass** against the actual audio (or the source this MIDI was derived from), or
- **Recommend a score-reference check** against a public-domain edition of Clementi's Sonatina Op. 36 No. 1 (a very commonly published teaching piece), to confirm where the exposition repeat sign (if performed) actually falls, and where the C-major → G-major modulation is harmonically complete.

**No excerpt file was cut or created in this task**, per the task's explicit instruction.

## 7. Updated 6-level benchmark ladder

| Level | File | Key/Mode | Role | Scope | Status |
|---|---|---|---|---|---|
| 1 | `Twinkle.mid` | C major | Monophonic sanity check | Full piece | Existing reference, not reprocessed |
| 2 | Bach — Minuet in G Major, BWV Anh. 114 | G major | Non-C tonic + light accompaniment | Full piece | Candidate, QA: warn (no key-signature metadata) |
| 3 | Beethoven — Für Elise (opening theme) | A minor | Relative-major/minor ambiguity test | Excerpt (opening theme only) — exact cut TBD | Candidate, QA: warn (no key-signature metadata) |
| 4 | Chopin — Prelude in E minor, Op. 28 No. 4 | E minor | Minor + slow harmony + chromatic pressure | Full piece (tentative) | Candidate, QA: warn (no key-signature metadata) |
| **5** | **Clementi — Sonatina Op. 36 No. 1, I** | **C major → G major** | **Short, clear single modulation (exposition)** | **Exposition only, no repeat — exact cut TBD (see section 6)** | **Candidate, QA: pass — added this task** |
| 6 | `Twinkle 12.mid` | C major (→ Eb → C) | High-stress ornamented variation / modulation stress test | Full piece | Existing reference, not reprocessed |

**Level 5 is no longer missing.** The ladder is now fully populated (6 of 6 levels have an assigned candidate or reference file), though Levels 3 and 5 both still require a manually/score-confirmed excerpt boundary before Phase 3G evaluation.

## 8. Scope note

**No tonal evaluation has started on this candidate.** No chroma extraction (Phase 2C), pitch-class baseline run (Phase 2D), chord-id EMA/SRN run (Phase 1.5B-style), uncertainty diagnostics (Phase 3B), or disagreement analysis (Phase 3C) has been performed on `clementi_opus36_1_1.mid`. This is QA and registry work only.
