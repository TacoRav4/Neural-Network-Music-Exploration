# Phase 3G-B — Tie-Aware Diagnostic Interpretation of the Phase 3G-A Corpus

Adds tie-aware diagnostics on top of the **frozen** Phase 3G-A pitch-class baseline results (`PHASE3G_A_pitch_class_corpus_metrics.json`, `03_MIDI_Data/derived_phase3g_corpus/*.npy`). Nothing here recomputes the baseline, changes `np.argmax` tie-breaking, or reruns chord-id EMA/SRN, Chroma SRN, Transformer, or any neural refinement. All Phase 3G-A anchor windows are reused verbatim (imported from `evaluate_phase3g_pitch_class_corpus.PIECES`), so every number below refers to the exact same windows Phase 3G-A already scored.

New, explicitly documented threshold introduced only in this script: `TONIC_NEIGHBORHOOD_FIFTHS = 2` (two predicted tonics are 'tonic-neighborhood' if within this many Circle-of-Fifths steps of each other). Not part of Phase 3G-A's frozen baseline.

## Tie-loss taxonomy definitions

Every anchor-window is classified into exactly one category (priority order below -- a mechanistic explanation for *why* argmax picked something else outranks a purely descriptive property of *what* was picked):

1. `inactive_or_silence_related` — window is inactive (thresholded chroma has no positive max)
2. `expected_tied_for_max_but_lost_to_lower_index` — expected key's raw score exactly ties the window's max score, but a different, lower-`key_index` key was tied too and `np.argmax` selected it instead
3. `collection_equivalent_wrong_mode_or_tonic_label` — predicted key is exactly the expected key's relative major/minor (same 7-pitch-class collection, different tonic/mode label)
4. `large_jump_instability` — this window is flagged `large_jump` (Circle-of-Fifths jump >= 3, Phase 3B's threshold)
5. `tonic_neighborhood_ambiguity` — predicted tonic is within 2 Circle-of-Fifths step(s) of the expected tonic
6. `expected_key_not_tied_for_max_no_support` — none of the above; the expected key had no raw-score support at all in this window

## Per-anchor tie-aware diagnostics

### L1 — `full_piece` (expected C Major)

- n_predictions=106, n_defined=106, n_undefined=0
- **strict_expected_key_proportion** = 1.0000 vs. **collection_equivalent_proportion** (expected key OR its relative A min) = 1.0000
- expected-key-in-top-tie: tied for max score in 106 window(s); of those, actually selected in 106, lost to tie-break in 0
- Tie-loss taxonomy (proportion of mismatched windows only):
  - `inactive_or_silence_related`: 0 (n/a)
  - `expected_tied_for_max_but_lost_to_lower_index`: 0 (n/a)
  - `collection_equivalent_wrong_mode_or_tonic_label`: 0 (n/a)
  - `large_jump_instability`: 0 (n/a)
  - `tonic_neighborhood_ambiguity`: 0 (n/a)
  - `expected_key_not_tied_for_max_no_support`: 0 (n/a)

### L2 — `full_piece` (expected G Major)

- n_predictions=164, n_defined=164, n_undefined=0
- **strict_expected_key_proportion** = 0.4207 vs. **collection_equivalent_proportion** (expected key OR its relative E min) = 0.4207
- expected-key-in-top-tie: tied for max score in 142 window(s); of those, actually selected in 69, lost to tie-break in 73
- Tie-loss taxonomy (proportion of mismatched windows only):
  - `expected_tied_for_max_but_lost_to_lower_index`: 73 (0.7684)
  - `tonic_neighborhood_ambiguity`: 22 (0.2316)
  - `inactive_or_silence_related`: 0 (0.0000)
  - `collection_equivalent_wrong_mode_or_tonic_label`: 0 (0.0000)
  - `large_jump_instability`: 0 (0.0000)
  - `expected_key_not_tied_for_max_no_support`: 0 (0.0000)
- Mismatched-window predicted-key breakdown: C maj (49), D maj (46)

### L3 — `full_excerpt` (expected A Minor)

- n_predictions=107, n_defined=106, n_undefined=1
- **strict_expected_key_proportion** = 0.0000 vs. **collection_equivalent_proportion** (expected key OR its relative C maj) = 0.7170
- expected-key-in-top-tie: tied for max score in 76 window(s); of those, actually selected in 0, lost to tie-break in 76
- Tie-loss taxonomy (proportion of mismatched windows only):
  - `expected_tied_for_max_but_lost_to_lower_index`: 76 (0.7170)
  - `large_jump_instability`: 9 (0.0849)
  - `tonic_neighborhood_ambiguity`: 18 (0.1698)
  - `expected_key_not_tied_for_max_no_support`: 3 (0.0283)
  - `inactive_or_silence_related`: 0 (0.0000)
  - `collection_equivalent_wrong_mode_or_tonic_label`: 0 (0.0000)
- Mismatched-window predicted-key breakdown: C maj (76), E maj (21), A maj (5), A# maj (4)

### L4 — `full_piece` (expected E Minor)

- n_predictions=216, n_defined=216, n_undefined=0
- **strict_expected_key_proportion** = 0.0000 vs. **collection_equivalent_proportion** (expected key OR its relative G maj) = 0.4907
- expected-key-in-top-tie: tied for max score in 135 window(s); of those, actually selected in 0, lost to tie-break in 135
- Tie-loss taxonomy (proportion of mismatched windows only):
  - `expected_tied_for_max_but_lost_to_lower_index`: 135 (0.6250)
  - `large_jump_instability`: 16 (0.0741)
  - `tonic_neighborhood_ambiguity`: 36 (0.1667)
  - `expected_key_not_tied_for_max_no_support`: 29 (0.1343)
  - `inactive_or_silence_related`: 0 (0.0000)
  - `collection_equivalent_wrong_mode_or_tonic_label`: 0 (0.0000)
- Mismatched-window predicted-key breakdown: G maj (106), C maj (52), E maj (27), D maj (14), A maj (7), F maj (7), F# maj (3)

### L5 — `approx_first_half` (expected C Major)

- n_predictions=18, n_defined=18, n_undefined=0
- **strict_expected_key_proportion** = 0.7222 vs. **collection_equivalent_proportion** (expected key OR its relative A min) = 0.7222
- expected-key-in-top-tie: tied for max score in 13 window(s); of those, actually selected in 13, lost to tie-break in 0
- Tie-loss taxonomy (proportion of mismatched windows only):
  - `tonic_neighborhood_ambiguity`: 5 (1.0000)
  - `inactive_or_silence_related`: 0 (0.0000)
  - `expected_tied_for_max_but_lost_to_lower_index`: 0 (0.0000)
  - `collection_equivalent_wrong_mode_or_tonic_label`: 0 (0.0000)
  - `large_jump_instability`: 0 (0.0000)
  - `expected_key_not_tied_for_max_no_support`: 0 (0.0000)
- Mismatched-window predicted-key breakdown: G maj (5)

### L5 — `approx_second_half` (expected G Major)

- n_predictions=16, n_defined=16, n_undefined=0
- **strict_expected_key_proportion** = 0.3750 vs. **collection_equivalent_proportion** (expected key OR its relative E min) = 0.3750
- expected-key-in-top-tie: tied for max score in 16 window(s); of those, actually selected in 6, lost to tie-break in 10
- Tie-loss taxonomy (proportion of mismatched windows only):
  - `expected_tied_for_max_but_lost_to_lower_index`: 10 (1.0000)
  - `inactive_or_silence_related`: 0 (0.0000)
  - `collection_equivalent_wrong_mode_or_tonic_label`: 0 (0.0000)
  - `large_jump_instability`: 0 (0.0000)
  - `tonic_neighborhood_ambiguity`: 0 (0.0000)
  - `expected_key_not_tied_for_max_no_support`: 0 (0.0000)
- Mismatched-window predicted-key breakdown: C maj (10)

### L6 — `pre_384s` (expected C Major)

- n_predictions=768, n_defined=665, n_undefined=103
- **strict_expected_key_proportion** = 1.0000 vs. **collection_equivalent_proportion** (expected key OR its relative A min) = 1.0000
- expected-key-in-top-tie: tied for max score in 665 window(s); of those, actually selected in 665, lost to tie-break in 0
- Tie-loss taxonomy (proportion of mismatched windows only):
  - `inactive_or_silence_related`: 0 (n/a)
  - `expected_tied_for_max_but_lost_to_lower_index`: 0 (n/a)
  - `collection_equivalent_wrong_mode_or_tonic_label`: 0 (n/a)
  - `large_jump_instability`: 0 (n/a)
  - `tonic_neighborhood_ambiguity`: 0 (n/a)
  - `expected_key_not_tied_for_max_no_support`: 0 (n/a)

### L6 — `384_to_432s` (expected Eb Major)

- n_predictions=96, n_defined=96, n_undefined=0
- **strict_expected_key_proportion** = 0.8958 vs. **collection_equivalent_proportion** (expected key OR its relative C min) = 0.8958
- expected-key-in-top-tie: tied for max score in 93 window(s); of those, actually selected in 86, lost to tie-break in 7
- Tie-loss taxonomy (proportion of mismatched windows only):
  - `expected_key_not_tied_for_max_no_support`: 3 (0.3000)
  - `expected_tied_for_max_but_lost_to_lower_index`: 7 (0.7000)
  - `inactive_or_silence_related`: 0 (0.0000)
  - `collection_equivalent_wrong_mode_or_tonic_label`: 0 (0.0000)
  - `large_jump_instability`: 0 (0.0000)
  - `tonic_neighborhood_ambiguity`: 0 (0.0000)
- Mismatched-window predicted-key breakdown: C maj (10)

### L6 — `post_432s` (expected C Major)

- n_predictions=510, n_defined=510, n_undefined=0
- **strict_expected_key_proportion** = 0.9412 vs. **collection_equivalent_proportion** (expected key OR its relative A min) = 0.9412
- expected-key-in-top-tie: tied for max score in 480 window(s); of those, actually selected in 480, lost to tie-break in 0
- Tie-loss taxonomy (proportion of mismatched windows only):
  - `expected_key_not_tied_for_max_no_support`: 4 (0.1333)
  - `tonic_neighborhood_ambiguity`: 26 (0.8667)
  - `inactive_or_silence_related`: 0 (0.0000)
  - `expected_tied_for_max_but_lost_to_lower_index`: 0 (0.0000)
  - `collection_equivalent_wrong_mode_or_tonic_label`: 0 (0.0000)
  - `large_jump_instability`: 0 (0.0000)
- Mismatched-window predicted-key breakdown: G maj (24), D# maj (4), A# maj (2)

## Bach (L2) interpretation

Of 95 mismatched windows in the `full_piece` anchor (expected G Major), 95 (1.0000) predicted a key within Bach's own C/G/D tonic neighborhood (the predicted-key breakdown is: C maj (49), D maj (46)), and 0 predicted a distant key outside that neighborhood. By the tie-loss taxonomy, 0.2316 of mismatches fall in the `tonic_neighborhood_ambiguity` category and 0.7684 in `expected_tied_for_max_but_lost_to_lower_index`. **Conclusion:** Bach's errors are essentially all close-tonic confusion among G/C/D, not failures onto a distant or unrelated key -- consistent with genuine (if imprecise) tonal-neighborhood evidence rather than random or catastrophic misprediction.

## Clementi (L5) interpretation

Frozen Phase 3G-A predicted-key run sequence: C maj -> G maj -> C maj.

Tonic-dominant-tonic excursion (C major -> G major -> C major), not a clean monotonic single modulation. The predicted trajectory visits and returns from the dominant region rather than settling there, which is the musically expected shape for a short exposition that tonicizes the dominant only briefly before a cadential return -- but it does mean Clementi should not be used as a 'before/after' single-modulation anchor test without accounting for the return leg.

## Chopin (L4) silence audit

Silence region: t=95.36-99.64s (9 windows at window_sec=0.5). Of these: 5/9 have numerically-zero raw chroma and 4/9 do not; 9/9 have strictly-positive smoothed chroma; 9/9 are flagged `active` by Phase 3G-A's `analyze_piece`.

The true raw-chroma-silent span is narrower than the full documented region: t=97.00-99.00s (5 windows), vs. the documented t=95.36-99.64s.
- Sustain/pedal tail before the true silence: t=95.50s (raw_sum=360.0), t=96.00s (raw_sum=360.0), t=96.50s (raw_sum=360.0)
- Next phrase's onset at/before the region's end boundary: t=99.50s (raw_sum=540.0)

Decay-formula check (smoothed_t = 0.8*smoothed_{t-1} verified directly against the saved arrays wherever raw_t is numerically zero): max absolute error = 0.00e+00 across 5 checked window(s) -- confirms the EMA formula, not just its qualitative effect.

**Classification: `combined_boundary_granularity_and_smoothing_memory_convention_artifact`**

0/9 inactive windows is explained by two combined, verified mechanisms -- neither is a pipeline bug. (1) Boundary granularity: only 5/9 of the documented silence-region windows are numerically raw-silent at all. 3 window(s) at the start of the region (t=95.50-96.50s) still carry real, nonzero raw chroma energy (a sustain/pedal tail from before the pause), and 1 window(s) at the end of the region (t=99.50-99.50s) already contain the next phrase's note onset arriving at/before the documented end boundary. The documented t=95.36-99.64s silence is a score-level/perceptual marker (Phase 3F.8); at this pipeline's 0.5s window granularity, the true raw-chroma-silent stretch is narrower (97.00-99.00s, 5 windows) than the full documented region, so windows outside that narrower stretch are correctly flagged active because they genuinely contain sound. (2) Memory-decay carryover: for the 5 genuinely raw-silent windows (t=97.00-99.00s), smoothed chroma stays strictly positive throughout (confirmed directly against the saved arrays via the geometric 0.8-decay formula, max absolute error 0.00e+00). Because the 10%-of-max threshold in `analyze_piece`/`extract_chroma_sequence` is relative to each window's OWN max, pure geometric decay preserves the ratio between pitch classes exactly, so the thresholded pattern's nonzero support never disappears as the signal decays -- only its magnitude does. `active` (thresholded chroma's per-window max > 0) therefore stays True through this short a silent stretch regardless of the pause. Both mechanisms are documented, expected consequences of representing a perceptual pause with fixed real-time boundaries against 0.5s raw MIDI windows, and of the per-window-relative-threshold + EMA-memory 'active' convention established in Phase 3B and reused as-is in Phase 3G-A -- not a defect in analyze_piece, extract_chroma_sequence, or any Phase 3G-A code, and per this task's guardrails it is documented here, not changed.

Diagnostic plot: `05_Figures_Results/PHASE3G_B_Chopin_silence_raw_vs_smoothed_chroma.png`

## Scope note

This is Phase 3G-B only: tie-aware diagnostic interpretation added on top of the frozen Phase 3G-A results. No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement was run. No Phase 2C/2D/3B/3C script, no Phase 3G-A script, and no Phase 3G-A output file was modified. `np.argmax` tie-breaking behavior in the underlying baseline was not changed -- it is analyzed here, not altered. The Chopin silence 'active' behavior is documented as a smoothing-memory/threshold-convention mechanism, not 'fixed.'
