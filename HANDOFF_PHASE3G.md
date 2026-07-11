# PROJECT HANDOFF — Recurrent Tonal Inference Workspace, entering Phase 3G

**Read this file first, then `STATUS.md` (full phase-by-phase log), then `README.md` (workspace orientation).**

## Main rule

Work only inside `07_Recurrent_Tonal_Inference/`. Do not touch original root files (the parent `Tonal Inference Modeling - Claude/` folder, outside this workspace) or the old COGS 269/202 notebook, unless explicitly instructed. Preserve all Phase 1/1.5/2/3 outputs — never overwrite or regenerate them without being asked. Every task should be small, scoped, verified (check outputs exist/non-empty/no-NaNs, confirm untouched files via mtime), and logged in `STATUS.md` (§6 Next approved step, §7 Not yet started, plus a dated phase subsection) before ending the turn.

## Project trajectory (why we're here)

- **COGS 269**: original feed-forward MLP (`ChordToKeyMLP`, 24→48→24) on synthetic chord/key associations. Static building block.
- **COGS 202**: real-MIDI pipeline — chroma → hand-coded EMA smoothing → triad-template chord matching → MLP → Circle-of-Fifths plots. Completed, presented, not the current work.
- **Phase 1**: compared hand-coded EMA+MLP vs. a learned Elman SRN on *synthetic* chord sequences. SRN improved accuracy (0.8101 vs. 0.7529 overall); EMA stayed slightly better on modulation lag/switch reliability (4.87 vs. 5.38 mean lag).
- **Phase 1.5**: ran EMA/SRN on *real* MIDI-derived chord-id sequences (Twinkle.mid, Twinkle 12.mid). SRN became smoother/more confident but did **not** recover the correct tonic — both models showed a global F-major bias despite C major being the dominant extracted chord.
- **Phase 2**: built a non-neural pitch-class/scale-template baseline (`pitch_class_baseline.py`) that bypasses triadic chord-matching entirely — 12-dim chroma → direct comparison against 24 full 7-note scale templates. It recovered Twinkle.mid as 100% C major and correctly tracked Twinkle 12.mid's real C→Eb→C modulation (100%/89.6%/94.1% across anchor windows). **Conclusion: representation (triadic forcing), not recurrence, was the main bottleneck** — this reframed everything after Phase 2.
- **Phase 3A–3D**: designed a staged "fast filter → anchors → uncertainty diagnostics → (conditional) neural refinement → evaluation gate" architecture.
  - **Phase 3B**: pitch-class baseline is best understood as a *diatonic-collection* fast filter, not a full tonic/mode resolver. Found `low_margin` (top1-top2 scale-template score margin) is **structurally saturated ~100%** because relative major/minor pairs (e.g. C major / A minor) get numerically **identical** `SCALE_TEMPLATES` rows — sparse/monophonic windows routinely tie across ~5-7 keys (up to 14). **`low_margin` alone is not a usable difficulty signal.** Use `key_switch`, `large_jump`, `anchor_mismatch`, and `tie_count` instead.
  - **Phase 3C**: chord-id EMA/SRN disagreement with the pitch-class baseline is **global** (87-100% of timesteps), not concentrated in the pitch-class baseline's own flagged difficult windows (concentration ratios ≈0.66-1.20, near 1.0 = uniform).
  - **Phase 3D**: decided Stage 4 neural refinement is **deferred** — the two existing benchmark pieces are too extreme (one trivial, one globally-biased) to exercise a "targeted local fix" scenario. Recommended building an intermediate-difficulty corpus first.
- **Phase 3E–3G-prep**: designed and built a 6-level benchmark ladder, QA'd every candidate file, locked/confirmed excerpt boundaries (with real convergent metadata + user-listening evidence), and cut two confirmed excerpt MIDI files.

## Current benchmark ladder (all boundaries now locked)

| Level | File | Key/mode | Role | Scope | Status |
|---|---|---|---|---|---|
| L1 | `Twinkle.mid` | C major | Monophonic sanity check | Full piece | Confirmed, existing reference |
| L2 | Bach — Minuet in G Major, BWV Anh. 114 | G major | Non-C tonic + light accompaniment | Full piece | Confirmed |
| L3 | Beethoven — Für Elise (opening) | A minor | Relative-major/minor ambiguity | **Excerpt `[0.0, 54.0]`s** | Confirmed (user listening + metadata) |
| L4 | Chopin — Prelude in E minor, Op. 28 No. 4 | E minor | Slow harmony + chromatic pressure | Full piece | **Tentative** — readability at 108s/600-note scale not yet verified by actually plotting it |
| L5 | Clementi — Sonatina Op. 36 No. 1, I (exposition) | C major → G major | Short, clear, single modulation | **Excerpt `[0.0, 17.3]`s** | Confirmed (user listening + metadata) |
| L6 | `Twinkle 12.mid` | C major → Eb → C | High-stress ornamented variation / modulation stress test | Full piece | Confirmed, existing reference, real embedded key signatures |

Extra/unassigned backup: Chopin Prelude in A minor, Op. 28 No. 2 (`f-f-chopin-prelude-op-28-no-2.mid`) — not in the main ladder, kept in case L3/L4 need a harder replacement later.

### Confirmed excerpt files (already cut and verified — do not re-cut)

- `03_MIDI_Data/candidate_intermediate_midi/excerpts/Fur_Elise_opening_0_54s.mid` — 53.99995s, 278 notes, 1 instrument, tempo/time-sig preserved. Boundary reason: F-natural (F major's marker) completely absent t=42-54s, present in every window from 54s onward; Bb confirms F major by ~60s. **Source of truth is the excerpt file, not `for_elise_by_beethoven.mid` (the full 169.7s piece) for any L3 evaluation.**
- `03_MIDI_Data/candidate_intermediate_midi/excerpts/Clementi_Op36_No1_I_exposition_norepeat_0_17p3s.mid` — 17.29999s, 135 notes, both hands preserved, key sig C Major preserved. Boundary reason: user-confirmed repeat of the intro at ~17s, matching a convergent onset-gap + repeating-harmonic-content metadata signature exactly (t=0-18s's C→G-major-prep→tonic arc repeats almost identically t=20-36s; new material only after t=36s). **Source of truth is this excerpt, not `clementi_opus36_1_1.mid` (the full 90.5s movement) for any L5 evaluation.**

Both excerpts were built with `pretty_midi.crop()` + a manual patch step to *clip* (not drop) the one sustained note in each file that crossed the boundary — `crop()`'s default behavior silently drops any note whose end exceeds the boundary, which is wrong for "clip cleanly."

**Known non-issue, do not misdiagnose:** Chopin No. 4 (L4) has a genuine, compositionally-intentional 4.29s silence at t≈95.36-99.64s (Chopin's famous pause near the end of the piece). This is real music, not file corruption or a pipeline bug — it will show up as a long "inactive"/undefined-key stretch under this workspace's active-window convention. Already documented in the registry's `structural_ambiguity_note` for L4.

## ⚠️ The one thing the ChatGPT draft got wrong: Phase 3G cannot literally "run the existing pipeline" unmodified

Every Phase 2C/2D/3B/3C script hardcodes exactly two files via module-level constants:

```python
TWINKLE_MIDI = os.path.join(_MIDI_DIR, "Twinkle.mid")
TWINKLE12_MIDI = os.path.join(_MIDI_DIR, "Twinkle 12.mid")
```

This appears in `evaluate_pitch_class_phase2d.py`, `pitch_class_uncertainty_diagnostics.py`, and `compare_phase3c_disagreement.py`. None of them accept a file list or loop over a corpus. **Phase 3G therefore requires writing new, corpus-aware scripts** (e.g. `midi_chroma_extraction_corpus.py`, `evaluate_pitch_class_corpus.py`, or a single generalized script covering the whole ladder) that reuse the same *logic, formulas, and conventions* as the Twinkle-only scripts — but do not modify the existing hardcoded ones. This is consistent with the workspace's established pattern (freeze old scripts, add new ones for new scope) and should not be treated as a deviation requiring special permission — just don't edit `evaluate_pitch_class_phase2d.py` etc. in place.

Concretely, for the new corpus (L2 Bach, L3 Für Elise excerpt, L4 Chopin, L5 Clementi excerpt — L1/L6 already fully processed), **nothing has been extracted yet**: no chroma arrays, no pitch-class baseline run, no diagnostics. Phase 3G starts from the Phase 2C chroma-extraction step for all four new files.

## Reusable modules and conventions to carry forward exactly

| Module | Role | Key constants |
|---|---|---|
| `shared_music_defs.py` | Frozen chord/key vocabulary, `decode_key`, `fifth_distance`, `key_tonic_pc`, `FIFTH_POS`, `key_index` | — |
| `midi_chord_extraction.py` (Phase 1.5A) | Chroma → chord-id (triadic) extraction | `window_sec=0.5`, `memory_decay=0.6` |
| `midi_chroma_extraction.py` (Phase 2C) | Raw/smoothed/thresholded chroma extraction | `window_sec=0.5`, `memory_decay=0.8`, `threshold_ratio=0.10` |
| `pitch_class_baseline.py` (Phase 2B) | Frozen `SCALE_TEMPLATES` (24×12 full scales) + `midi_to_key_baseline` | `memory_decay=0.8` |
| `evaluate_pitch_class_phase2d.py` (Phase 2D) | Pitch-class baseline evaluation, trajectory + Circle-of-Fifths plots | Twinkle-only, hardcoded |
| `pitch_class_uncertainty_diagnostics.py` (Phase 3B) | `raw_scores`, `normalized_margin`, `entropy`, `tie_count`, `key_switch`, `large_jump`, anchor-window mismatch | `LOW_MARGIN_THRESHOLD=0.20` (saturated, don't trust alone), `LARGE_JUMP_THRESHOLD=3`, `HIGH_TIE_COUNT_THRESHOLD=8` |
| `mlp_baseline.py` / `srn_model.py` / `sequence_dataset.py` / `run_comparison.py` | Phase 1/1.5 EMA+MLP and SRN training/inference | `SEED=269`, EMA `alpha=0.20`, SRN `epochs=25, lr=1e-3, hidden_size=48` (the "best condition" from Phase 3's diagnostic sweep) |
| `run_midi_phase15_evaluation.py` | `train_models()` — reusable EMA+SRN training entry point, already imported by `compare_phase3c_disagreement.py` | Twinkle-only for the MIDI target, but `train_models()` itself is corpus-agnostic (trains on synthetic data) |
| `compare_phase3c_disagreement.py` (Phase 3C) | Chord-id vs. pitch-class disagreement analysis | Twinkle-only, hardcoded |
| `inspect_intermediate_midi_candidates.py` (Phase 3F) | Generic MIDI QA (duration, notes/sec, polyphony, warnings) | Already directory-generic — reusable as-is on the new corpus |

**Two real bugs already caught and fixed once each — do not reintroduce them:**
1. **Leading-silent-window offset.** `midi_to_key_baseline` / `midi_to_chord_ids` silently drop leading silent chroma windows before their first prediction, so `predictions[0]` does not correspond to real time `t=0`. Any script mapping predictions to wall-clock time must compute `offset_windows = n_chroma_windows - n_predictions` and add it back — verified in Phase 2D/3B/3C via an explicit `alignment_max_time_error_sec` check that should read `0.0`.
2. **`np.argsort` vs `np.argmax` tie-breaking.** Given how often `SCALE_TEMPLATES` rows tie exactly (see Phase 3B's tie-count finding), always use plain `np.argmax` for top-1 key selection — `np.argsort`'s default sort is not guaranteed tie-consistent with `argmax` and silently diverged from the canonical prediction on ~7% of Twinkle 12.mid's windows during Phase 3B development.

## Open scope question for Phase 3G (resolve before diving in, don't assume)

Phase 3A's roadmap defined Phase 3G as "chroma extraction → pitch-class baseline → uncertainty diagnostics" (Stage 1 + Stage 3 of the staged architecture) on the new corpus. It is *not* yet decided whether Phase 3G also re-runs the chord-id EMA/SRN comparison (Phase 1.5B/3C-style) on the four new pieces, or whether that's deferred to a later sub-phase. Ask the user, or propose starting with pitch-class-only (Stage 1) evaluation across the full 6-level ladder first, since that directly extends Phase 2D/3B's already-established methodology and answers the more urgent question (does the fast filter behave differently on non-C-major, minor-key, and single-modulation pieces?) before reintroducing the heavier chord-id/SRN machinery.

## Still explicitly out of scope

- Do not implement a Chroma SRN.
- Do not implement a Transformer.
- Do not implement any neural refinement (Stage 4 remains deferred per Phase 3D).
- Do not modify any existing Phase 1/1.5/2/3 script in place — write new scripts for new corpus scope.
- Do not regenerate old Phase 1/1.5/2/3 plots or outputs.
- Do not treat Chopin No. 4's long silence as corruption.
- `source_license` fields for all 5 new candidate files (Bach, Für Elise, Chopin ×2, Clementi) are still `<TBD>` placeholders in the registry — fine for internal analysis, but flag if anything from this corpus is ever meant to be shared/published.

## Where things are

- `STATUS.md` — the authoritative phase-by-phase log; always read §6/§7 first for current state.
- `README.md` — workspace orientation, folder guide.
- `03_MIDI_Data/candidate_intermediate_midi/candidate_midi_registry.json` — full corpus registry (QA, boundaries, expected challenges per file).
- `05_Figures_Results/PHASE3G_PREP_EXCERPT_FILES_REPORT.md` — most recent report (excerpt creation).
- `05_Figures_Results/PHASE3F*.md`, `PHASE3D_STAGE4_DECISION_REVIEW.md`, `PHASE3A_STAGED_TONAL_INFERENCE_DESIGN.md`, `PHASE3B_pitch_class_uncertainty_report.md`, `PHASE3C_*_disagreement_report.md`, `PHASE2D_pitch_class_baseline_report.md`, `PHASE1_5B_MIDI_EMA_vs_SRN_SUMMARY.md`, `PHASE1_SYNTHETIC_EMA_vs_SRN_SUMMARY.md` — full evidence trail, read in roughly that order if deeper context is needed.
- Also worth knowing: this workspace's `07_Recurrent_Tonal_Inference/` content (through Phase 3F) has been pushed to a separate branch (`recurrent-tonal-inference`) on `github.com/TacoRav4/Neural-Network-Music-Exploration`, kept apart from the original `main` branch (the old Colab notebooks). Not relevant to Phase 3G's technical work, but don't be surprised if asked to push again later — a working deploy key is already set up for that repo.

## Next approved step

**Phase 3G** — extend the Phase 2C→2D→3B pipeline to the full 6-level corpus (new corpus-aware scripts, not edits to existing ones), starting with chroma extraction on the four new files (Bach, Für Elise excerpt, Chopin No. 4, Clementi excerpt), then pitch-class baseline evaluation, then uncertainty diagnostics — following the exact conventions and thresholds documented above.
