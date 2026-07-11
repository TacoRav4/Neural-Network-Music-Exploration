# Phase 3B — Pitch-Class Baseline Uncertainty Diagnostics Report

Computes uncertainty diagnostics for the pitch-class/scale-template baseline (`pitch_class_baseline.py`, Phase 2B) directly from the saved Phase 2C chroma arrays. **This is diagnostic only** -- no prediction was changed, no model was trained, and no neural refinement (Chroma SRN, Transformer) was implemented. This report also does **not** compare against the chord-id EMA/SRN outputs from Phase 1.5B -- that cross-model comparison is Phase 3C, not here.

## Method

For each timestep, `raw_scores = thresholded_smoothed_chroma @ SCALE_TEMPLATES.T` (24-way scale-template dot product, same formula `pitch_class_baseline.py` uses internally). From `raw_scores` we compute: the top-1/top-2 keys and scores, `raw_margin = top1_score - top2_score`, `normalized_margin = raw_margin / (top1_score + eps)`, and a diagnostic softmax entropy at a fixed, documented temperature (1.0) -- **entropy here is temperature-dependent and not a calibrated confidence measure**, only useful for relative comparison within this report.

## Alignment handling

Unlike `pitch_class_baseline.midi_to_key_baseline` (which silently drops leading silent chroma windows before its first prediction -- the source of a real, caught-and-fixed timing bug in Phase 2D), this module computes scores over the **full** saved chroma array, every window from Phase 2C, none dropped. `prediction_times_sec = prediction_indices * window_sec` always, with no offset correction needed. Silent windows are tracked via an explicit `active` mask rather than being skipped.

## Note: `low_margin` is saturated, and why

`low_margin_proportion` is 1.0 (100%) for both pieces, with `mean_normalized_margin` at essentially 0.0. This is **not a broken metric** -- it reflects a real, structural property of the 24-way full scale-template representation: `SCALE_TEMPLATES` gives *relative* major/minor pairs (e.g. C major and A minor) numerically **identical** rows, since a natural-minor scale shares all 7 pitch classes with its relative major. When a window's active chroma evidence is sparse (a single or few pitch classes, typical of monophonic melodic material like both test pieces), many of the 24 templates tie exactly at the maximum score -- an average of ~5-7 tied keys per active window in this data (see `mean_tie_count` below), and as many as 14 simultaneously. `top1_key` is still resolved deterministically (via `argmax`'s leftmost-tie convention, matching `pitch_class_baseline.py` exactly -- this was verified against `midi_to_key_baseline`'s own output during development), but the *margin* between that pick and the runner-up is close to meaningless as a confidence signal on its own: it is near-zero whether the pick is obviously correct or a coin-flip among many tied candidates. **This means `low_margin`, as currently thresholded, does not discriminate difficult windows from easy ones -- it flags nearly everything.** `tie_count` (also computed below) is a more informative alternative and should be considered in place of, or alongside, `normalized_margin` in Phase 3C.

## Twinkle.mid

- n_predictions: 106 (active: 106, 1.0000)
- n_unique_predicted_keys: 1 (C maj)
- key switches: 0 / 105 eligible transitions (0.0000)
- fifths jump: mean=0.00, max=0.00, large jumps: 0 (0.0000)
- low_margin windows: 106 / 106 (1.0000)
- mean_normalized_margin: 0.0000, mean_entropy: 1.6859
- mean_tie_count (active windows): 5.23 of 24 keys tied at the max scale-template score (max observed: 14)

### Anchor-window diagnostics

**full_piece** (expected: C Major):
- n_predictions=106, proportion_expected_key=1.0000
- low_margin_proportion=1.0000, mean_normalized_margin=0.0000, mean_entropy=1.6859
- mismatch_count=0, mismatch_intervals=0

### Difficult-window summary (Twinkle.mid)

- low_margin: 106 (1.0000); key_switch: 0 (0.0000); large_jump: 0 (0.0000); anchor_mismatch: 0 (0.0000)
- any_difficult: 106 timesteps (1.0000) across 1 contiguous interval(s)
  (note: `any_difficult` is a union that includes the saturated `low_margin` criterion -- see the note above -- so it is dominated by that criterion and is not, by itself, a useful difficulty summary; prefer `key_switch`/`large_jump`/`anchor_mismatch` individually.)

## Twinkle 12.mid

- n_predictions: 1374 (active: 1271, 0.9250)
- n_unique_predicted_keys: 4 (C maj, D# maj, G maj, A# maj)
- key switches: 11 / 1270 eligible transitions (0.0087)
- fifths jump: mean=0.02, max=3.00, large jumps: 5 (0.0039)
- low_margin windows: 1271 / 1374 (0.9250)
- mean_normalized_margin: 0.0000, mean_entropy: 2.4003
- mean_tie_count (active windows): 6.71 of 24 keys tied at the max scale-template score (max observed: 14)

### Anchor-window diagnostics

**pre_384s** (expected: C Major):
- n_predictions=768, proportion_expected_key=1.0000
- low_margin_proportion=1.0000, mean_normalized_margin=0.0000, mean_entropy=2.4907
- mismatch_count=0, mismatch_intervals=0

**384_to_432s** (expected: Eb Major):
- n_predictions=96, proportion_expected_key=0.8958
- low_margin_proportion=1.0000, mean_normalized_margin=0.0000, mean_entropy=1.6018
- mismatch_count=10, mismatch_intervals=3

**post_432s** (expected: C Major):
- n_predictions=510, proportion_expected_key=0.9412
- low_margin_proportion=1.0000, mean_normalized_margin=0.0000, mean_entropy=2.2758
- mismatch_count=30, mismatch_intervals=3

### Difficult-window summary (Twinkle 12.mid)

- low_margin: 1271 (0.9250); key_switch: 11 (0.0080); large_jump: 5 (0.0036); anchor_mismatch: 40 (0.0291)
- any_difficult: 1271 timesteps (0.9250) across 1 contiguous interval(s)
  (note: `any_difficult` is a union that includes the saturated `low_margin` criterion -- see the note above -- so it is dominated by that criterion and is not, by itself, a useful difficulty summary; prefer `key_switch`/`large_jump`/`anchor_mismatch` individually.)

## Does the pitch-class baseline have meaningful failure/uncertainty regions?

Because `low_margin` is saturated for both pieces (see the note above), it is excluded from the characterization below in favor of `key_switch`, `large_jump`, and `anchor_mismatch` -- the criteria that actually discriminate stable regions from unstable ones in this data. **Twinkle.mid** shows essentially no measurable instability by the discriminating criteria: 0 key switches and 0 large Circle-of-Fifths jumps across all 106 predictions (100% C major throughout, matching Phase 2D). There is little for a neural refinement to target on this piece by this analysis. **Twinkle 12.mid** shows concentrated (not uniform) instability: 11 key switches and 5 large Circle-of-Fifths jumps across 1374 predictions (0.0087 / 0.0039 of eligible transitions) -- both rare overall, meaning the piece is mostly stable with a small number of genuinely unstable transition points, not pervasively noisy. In the 384_to_432s anchor window (expected Eb major), proportion_expected_key=0.8958 with 10 mismatched predictions across 3 contiguous interval(s) -- the baseline's correct tracking of this real modulation (per Phase 2D) is not perfect within the window, and these mismatch intervals are natural first candidates for Phase 3C's targeted refinement analysis. Tie-count is a more useful confidence signal here than margin: Twinkle.mid averages 5.23 tied keys per active window and Twinkle 12.mid averages 6.71 -- both pieces' per-window pitch-class evidence is often ambiguous in isolation, and the baseline's real accuracy comes from the temporal chroma-level EMA smoothing accumulating evidence across many windows, not from any single window being individually decisive. This is itself informative for Phase 3C/3D: a neural refinement operating on single windows would face the same structural ambiguity; any useful refinement likely needs its own temporal integration, not just a better per-window classifier.

## Scope note

This is Phase 3B: non-neural uncertainty diagnostics only. **No neural model was implemented** (no Chroma SRN, no Transformer, no refinement of any kind). Comparison against the chord-id EMA/SRN outputs (Phase 1.5B) is Phase 3C, not performed here.
