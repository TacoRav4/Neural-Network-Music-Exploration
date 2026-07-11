# Phase 3C — Twinkle 12.mid: Pitch-Class vs. Chord-ID Disagreement Report

Compares Phase 3B's pitch-class difficult windows against Phase 1.5B's chord-id EMA/SRN behavior on Twinkle 12.mid. **Comparison/diagnostics only** -- no prediction was changed, no model was tuned (EMA and SRN were regenerated deterministically with Phase 1.5B's exact settings, since Phase 1.5B did not persist per-timestep arrays to disk), and no neural refinement (Chroma SRN, Transformer) was implemented.

## Method

EMA+MLP (alpha=0.2) and SRN (epochs=25, lr=0.001, hidden_size=48) were regenerated via `run_midi_phase15_evaluation.train_models()` (reused, not modified) and run on the Phase 1.5A saved chord-id sequence (1271 predictions). Disagreement at a timestep = the chord-id model's argmax key differs from the pitch-class baseline's (Phase 3B) key at the same real time.

## Alignment handling

Chord-id predictions start `chord_offset_windows=103` windows into the piece (leading silent chroma windows dropped, same convention as the pitch-class path -- see Phase 2D/3B), so `chord_prediction_times_sec = (index + chord_offset_windows) * window_sec`. Each chord-id timestep is matched to the pitch-class path's prediction at the **same real time** (`round(time / window_sec)` into Phase 3B's full, undropped array), not by assuming shared array indices. **Alignment verification: max time error = 0.000000 seconds** (0 confirms the two paths' window grids coincide exactly, as expected since both derive from the same raw chroma extraction at the same `window_sec`).

## Difficulty signals used

Per this task's instruction and Phase 3B's finding: `low_margin` is **not** used as a standalone criterion (Phase 3B found it structurally saturated at ~100% for both pieces, due to `SCALE_TEMPLATES` ties between relative major/minor pairs -- it does not discriminate). Criteria used: `pc_key_switch`, `pc_large_jump`, `pc_anchor_mismatch`, and `pc_high_tie_count` (tie_count >= 8, a fixed, documented, un-tuned threshold above both pieces' mean tie_count).

## Overall disagreement

- EMA vs. pitch-class baseline: 0.9056 of timesteps disagree
- SRN vs. pitch-class baseline: 0.8906 of timesteps disagree

## Chord-id model's own instability (for context)

- EMA: 121 key switches, 52 large jumps, across 1271 predictions
- SRN: 110 key switches, 30 large jumps, across 1271 predictions

## Overlap between disagreement and pitch-class difficulty criteria

| model | criterion | criterion_fraction | disagreement_rate_within | disagreement_rate_outside | concentration_ratio |
|---|---|---|---|---|---|
| EMA | pc_key_switch | 0.0087 | 0.8182 | 0.9063 | 0.90 |
| EMA | pc_large_jump | 0.0039 | 0.6000 | 0.9068 | 0.66 |
| EMA | pc_anchor_mismatch | 0.0315 | 0.7500 | 0.9106 | 0.82 |
| EMA | pc_high_tie_count | 0.5319 | 0.9808 | 0.8202 | 1.20 |
| SRN | pc_key_switch | 0.0087 | 0.8182 | 0.8913 | 0.92 |
| SRN | pc_large_jump | 0.0039 | 1.0000 | 0.8902 | 1.12 |
| SRN | pc_anchor_mismatch | 0.0315 | 1.0000 | 0.8871 | 1.13 |
| SRN | pc_high_tie_count | 0.5319 | 0.9675 | 0.8034 | 1.20 |

`concentration_ratio` = disagreement rate inside the criterion / disagreement rate outside it. A ratio near 1.0 means disagreement is roughly **uniform** regardless of this criterion (a global pattern); a ratio well above 1.0 means disagreement **concentrates** inside these flagged windows (a local, targetable pattern).

## Interpretation for staged architecture

Disagreement is only weakly concentrated inside the flagged difficulty windows (mean concentration ratio ≈ 0.99, close to 1.0) -- overall disagreement rates are 0.9056 (EMA) and 0.8906 (SRN), comparable to the rates seen inside and outside the flagged windows. This supports a **global representation-wide bias** interpretation (consistent with Phase 1.5B's F-major-bias finding) rather than a small number of locally difficult spots -- a Stage 4 neural refinement targeted only at these specific flagged windows would likely leave most of the disagreement untouched, since the disagreement is not concentrated there. This directly informs Phase 3A's Stage 3/4 design question: if disagreement is global, the staged architecture's premise (fix a validated fast filter's rare local failures with a small, targeted neural pass) does not describe the chord-id models' actual failure mode -- the chord-id path's problem is representation-wide, not a handful of hard windows, and no amount of *targeted* refinement (as opposed to representation-level change, e.g. a Chroma SRN operating on chroma directly rather than triadic chord ids) would be expected to close most of the gap. If disagreement is local, the staged premise holds and Stage 4 is worth pursuing on the flagged windows specifically.

## Scope note

This is Phase 3C: comparison/diagnostics only. No neural refinement, Chroma SRN, or Transformer was implemented. No dense per-timestep MIDI accuracy is claimed anywhere in this report -- all comparisons are between two model outputs (pitch-class vs. chord-id), not against ground truth.
