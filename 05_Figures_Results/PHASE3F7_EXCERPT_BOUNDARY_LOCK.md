# Phase 3F.7 — Excerpt Boundary Lock

**Boundary/metadata analysis only. No tonal-inference evaluation (chroma extraction, pitch-class baseline, chord-id EMA/SRN, uncertainty diagnostics, or disagreement analysis) has been run on any candidate. No clipped/excerpt MIDI files were created. No existing scripts, notebooks, or Phase 1/1.5/2/3 outputs have been modified.**

Reviews and locks (where possible) the scope/excerpt decisions for the 6-level intermediate MIDI corpus ladder established in `PHASE3F_INTERMEDIATE_MIDI_CORPUS_REGISTRY.md` and `PHASE3F6_CLEMENTI_LEVEL5_QA.md`, before any Phase 3G tonal-inference evaluation runs.

## 1. Current 6-level ladder and scope decisions

| Level | File | Key/Mode | Scope decision | Status |
|---|---|---|---|---|
| 1 | `Twinkle.mid` | C major | Full piece | Existing reference — confirmed |
| 2 | Bach — Minuet in G Major, BWV Anh. 114 | G major | **Full piece** | **Confirmed** — comparable scale to Twinkle.mid, no excerpting needed |
| 3 | Beethoven — Für Elise (opening theme) | A minor | **Opening excerpt only** | **Manual review required** — see section 2 |
| 4 | Chopin — Prelude in E minor, Op. 28 No. 4 | E minor | **Full piece, tentatively** | **Tentative** — not yet plot-tested for readability |
| 5 | Clementi — Sonatina Op. 36 No. 1, I | C major → G major | **Exposition only, no repeat** | **Tentative** — see section 2 (well-supported, not yet confirmed) |
| 6 | `Twinkle 12.mid` | C major → Eb → C | Full piece | Existing reference — confirmed (real embedded key signatures) |
| *extra* | Chopin — Prelude in A minor, Op. 28 No. 2 | A minor | Backup only, not in main ladder | Not applicable |

Bach Minuet and Chopin No. 4 were already decided as full-piece candidates in Phase 3F (restated here, not re-derived); Chopin No. 2 remains an unassigned backup, not part of the main ladder. The two entries requiring real boundary analysis in this task are **Für Elise (Level 3)** and **Clementi (Level 5)**.

## 2. Für Elise (Level 3) boundary analysis

**Method:** (a) note-onset gap scan across both hands over the full 230.3s file; (b) 2-second-window pitch-class-content scan from t=0 to t=46s, tracking whether the A-natural-minor collection `{A,B,C,D,E,F,G#}` (pitch classes 0,2,3,4,8,9,11 relative to A... expressed here as absolute pitch classes `{0,2,3,4,8,9,11}` = C,D,D#/Eb,E,G#/Ab,A,B, i.e. A natural/harmonic minor) stays intact, and specifically watching for F-natural (pitch class 5, the marker that would indicate a shift toward F major, the piece's B section).

**Findings:**
- Largest early onset gap: **t=15.31s** (0.612s gap).
- Pitch-class content is stable and consistent with A minor from t=0 through **at least t=30s** — no meaningful F major evidence in this range.
- A single transient F-natural appears only in the **t=32-34s** window, then the texture becomes ambiguous (pitch classes 3,4 = Eb/D#,E dominate t=38-42s), consistent with a transitional/sequential passage rather than a clean modulation.

**Interpretation:** the t=15.31s gap is very likely only a **sub-phrase or motif boundary within the opening theme**, not the end of the full A section — cutting there would almost certainly be premature and would truncate the theme mid-statement. The evidence instead suggests the A-minor material continues intact through roughly **t≈28-30s**, giving a defensible *candidate* window of **[0.0s, ~28-30s]** for an A-minor-only excerpt. However:

- Metadata alone does **not** cleanly establish where the modulation to F major (B section) begins — the transient F at 32-34s is weak, single-occurrence evidence, not a confirmed modulation point.
- **This boundary is NOT confirmed.** `excerpt_boundary_status = "manual_review_required"` in the registry. A manual listening pass or score reference is needed to pin down the exact end of the A section before any excerpt file is cut.

## 3. Clementi (Level 5) boundary analysis

**Method:** the same two-signal approach, applied to the full 90.47s file: (a) note-onset gap scan (already partially done in Phase 3F.6); (b) 2-second-window pitch-class-content scan from t=0 to t=40s, tracking F-natural (pitch class 5, in-key for C major) vs. F# (pitch class 6, the marker of G major / dominant preparation, foreign to C major).

**Findings:**
- Onset gaps: t=17.26s (0.647s) and t=35.16s (0.651s) — a ~2x relationship, as already noted in Phase 3F.6.
- Pitch-class content from **t=0-18s** traces a clear harmonic arc: F-natural present (C major) at t=2-6s → F# present (G major / dominant preparation) at t=6-16s → bare tonic triad (C-E-G, no F of either kind) at t=18-20s. This is a complete **tonic → dominant → tonic-ish cadence** arc.
- This **exact same pattern repeats** from **t=20-36s**: F-natural at t=20-24s, F# at t=24-34s.
- After **t=36s**, new pitch-class material appears — Eb/D# (pitch class 3), which is foreign to both C major and G major — not seen anywhere in the first 36 seconds, consistent with a development section beginning.

**Why the ~17.3s boundary should be treated cautiously, but is now better supported:** the Phase 3F.6 hypothesis rested on onset-gap timing alone, which by itself cannot distinguish "end of exposition, about to repeat" from "an ordinary phrase-boundary rest that happens to be a bit longer than usual." Phase 3F.7 adds an **independent, convergent signal**: the *harmonic content itself* (not just the timing of rests) repeats almost identically across the two candidate 18-second spans. A harmonic pattern repeating in lockstep with a timing pattern is meaningfully stronger evidence than either alone — it argues against t=17.3s being a bare sub-phrase boundary (which would not be expected to reset the entire tonic→dominant→tonic harmonic arc) and for it marking a genuine **first-pass-through-the-exposition** boundary, immediately followed by a restatement (written-out or repeated).

**This is still not full confirmation.** The analysis is 2-second-granularity pitch-class-window scanning, not bar-level score alignment or an actual listening pass, and cannot rule out, e.g., a written-out variation rather than a literal repeat, or an off-by-a-few-seconds boundary. `excerpt_boundary_status = "tentative"` in the registry (one level more confident than Für Elise's "manual_review_required", but still short of "confirmed"). **Recommended candidate excerpt: `[0.0s, 17.3s]`** for the no-repeat exposition, pending a quick listening or score check.

## 4. What must be manually confirmed before Phase 3G

| Item | Current state | What's needed |
|---|---|---|
| Für Elise (L3) excerpt end | Candidate range ~28-30s; t=15.31s ruled out as premature | Manual listening or score reference to pinpoint the actual A-section / B-section (F major) boundary |
| Clementi (L5) excerpt end | Tentative ~17.3s, well-supported by convergent timing + harmonic evidence | A quick listening or score check to confirm the exposition-repeat boundary before cutting |
| Chopin No. 4 (L4) full-piece readability | Tentative — not yet plot-tested | Run through Phase 2C/2D plotting once Phase 3G begins to confirm 108s/600-note scale is actually readable in practice |

No excerpt MIDI files have been created for Level 3 or Level 5 — both remain full-piece files on disk pending these confirmations, per this task's explicit instruction not to cut anything without confirmed boundaries.

## 5. Scope note

**No tonal evaluation has started.** No chroma extraction, pitch-class baseline run, chord-id EMA/SRN run, uncertainty diagnostics, or disagreement analysis has been performed on any candidate in this task. This is boundary analysis and registry locking only — Phase 3G (running the existing, unmodified Phase 2C/2D/3B pipeline) remains a future step, gated on the confirmations in section 4.
