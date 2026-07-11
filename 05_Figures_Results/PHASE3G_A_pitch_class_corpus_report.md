# Phase 3G-A — Corpus-Aware Pitch-Class Baseline & Uncertainty Evaluation

Extends the Phase 2C (chroma extraction) -> Phase 2B/2D (pitch-class/scale-template baseline) -> Phase 3B (uncertainty diagnostics) pipeline to the full 6-level benchmark ladder, using a new, corpus-aware script (`evaluate_phase3g_pitch_class_corpus.py`) that reuses (imports, does not modify) the exact logic, formulas, and conventions of the frozen Twinkle-only scripts. **Stage 1 + Stage 3 only** -- no chord-id EMA/SRN comparison, no Chroma SRN, no Transformer, no neural refinement of any kind.

## Settings

- window_sec: 0.5
- chroma memory_decay: 0.8
- threshold_ratio: 0.1
- SCALE_TEMPLATES: imported from `pitch_class_baseline.py` (24x12 full major/natural-minor scales), not redefined
- top-1 key selection: plain `np.argmax` (never `np.argsort`), matching Phase 2B/3B exactly
- `low_margin` is not used as a standalone difficulty signal (structurally saturated by relative major/minor scale-template ties, per Phase 3B) -- `key_switch`, `large_jump`, `anchor_mismatch`, and `tie_count` are used instead

## L1 — Twinkle.mid

- Role: Monophonic sanity check, full piece.
- MIDI: `03_MIDI_Data/Twinkle.mid`
- Duration: 53.05s, 106 chroma windows, baseline_offset_windows=0

**Predicted key sequence:** 106 predictions, 1 unique keys.
Dominant predicted keys: C maj (100.0%)

**Circle-of-Fifths jumps (baseline sequence):** mean=0.00, max=0.00, large jumps (>=3): 0 (0.0000 of transitions)

**Uncertainty summary (full grid):** 106 predictions, 106 active (1.0000); 1 unique keys (C maj); key switches 0/105 (0.0000); large jumps 0 (0.0000).

**tie_count summary:** mean=5.23 tied keys/active window, max=14. (Structurally saturated by relative major/minor scale-template ties, per Phase 3B -- `low_margin` alone is not treated as a difficulty signal here.)

**Anchor mismatch summary:**

- `full_piece` (expected C Major, confirmed window): n=106, proportion_expected_key=1.0000, mismatch_count=0 across 0 interval(s)

**Inactive/silent window handling:** 0 inactive windows (0.0000 of grid).

Outputs: `05_Figures_Results/PHASE3G_A_Twinkle_key_trajectory.png`, `05_Figures_Results/PHASE3G_A_Twinkle_circle_of_fifths.png`

## L2 — Bach — Minuet in G Major, BWV Anh. 114

- Role: Non-C tonic + light accompaniment, full piece.
- MIDI: `03_MIDI_Data/candidate_intermediate_midi/J.S. Bach - Minuet in G Major, BWV Anh. 114.mid`
- Duration: 82.29s, 164 chroma windows, baseline_offset_windows=0

**Predicted key sequence:** 164 predictions, 3 unique keys.
Dominant predicted keys: G maj (42.1%), C maj (29.9%), D maj (28.0%)

**Circle-of-Fifths jumps (baseline sequence):** mean=0.08, max=2.00, large jumps (>=3): 0 (0.0000 of transitions)

**Uncertainty summary (full grid):** 164 predictions, 164 active (1.0000); 3 unique keys (C maj, D maj, G maj); key switches 12/163 (0.0736); large jumps 0 (0.0000).

**tie_count summary:** mean=3.15 tied keys/active window, max=6. (Structurally saturated by relative major/minor scale-template ties, per Phase 3B -- `low_margin` alone is not treated as a difficulty signal here.)

**Anchor mismatch summary:**

- `full_piece` (expected G Major, confirmed window): n=164, proportion_expected_key=0.4207, mismatch_count=95 across 6 interval(s)

**Inactive/silent window handling:** 0 inactive windows (0.0000 of grid).

Outputs: `05_Figures_Results/PHASE3G_A_Bach_Minuet_G_key_trajectory.png`, `05_Figures_Results/PHASE3G_A_Bach_Minuet_G_circle_of_fifths.png`

## L3 — Beethoven — Für Elise (opening excerpt, [0.0, 54.0]s)

- Role: Relative-major/minor ambiguity test. Excerpt is the source of truth (per Phase 3F.8), not the full 169.7s piece.
- MIDI: `03_MIDI_Data/candidate_intermediate_midi/excerpts/Fur_Elise_opening_0_54s.mid`
- Duration: 54.00s, 107 chroma windows, baseline_offset_windows=1

**Predicted key sequence:** 106 predictions, 4 unique keys.
Dominant predicted keys: C maj (71.7%), E maj (19.8%), A maj (4.7%), A# maj (3.8%)

**Circle-of-Fifths jumps (baseline sequence):** mean=0.59, max=6.00, large jumps (>=3): 15 (0.1429 of transitions)

**Uncertainty summary (full grid):** 107 predictions, 106 active (0.9907); 4 unique keys (C maj, E maj, A maj, A# maj); key switches 16/105 (0.1524); large jumps 15 (0.1429).

**tie_count summary:** mean=3.55 tied keys/active window, max=14. (Structurally saturated by relative major/minor scale-template ties, per Phase 3B -- `low_margin` alone is not treated as a difficulty signal here.)

**Anchor mismatch summary:**

- `full_excerpt` (expected A Minor, confirmed window): n=107, proportion_expected_key=0.0000, mismatch_count=106 across 1 interval(s)
  - tie-break check: expected key exactly tied for the max scale-template score in 76 active window(s); actually selected in 0 of those. **Structurally impossible for the expected minor key to win these ties** (major-key indices 0-11 always beat tied minor-key indices 12-23 under `np.argmax`'s leftmost-tie rule) -- verified, not assumed.

**Inactive/silent window handling:** 1 inactive windows (0.0093 of grid).

Outputs: `05_Figures_Results/PHASE3G_A_FurElise_excerpt_key_trajectory.png`, `05_Figures_Results/PHASE3G_A_FurElise_excerpt_circle_of_fifths.png`

## L4 — Chopin — Prelude in E minor, Op. 28 No. 4

- Role: Slow harmony + chromatic pressure, full piece. Contains a genuine dramatic silence (see silence_region).
- MIDI: `03_MIDI_Data/candidate_intermediate_midi/f-f-chopin-prelude-op-28-no-4.mid`
- Duration: 108.21s, 216 chroma windows, baseline_offset_windows=0

**Predicted key sequence:** 216 predictions, 7 unique keys.
Dominant predicted keys: G maj (49.1%), C maj (24.1%), E maj (12.5%), D maj (6.5%), A maj (3.2%)

**Circle-of-Fifths jumps (baseline sequence):** mean=0.44, max=6.00, large jumps (>=3): 20 (0.0930 of transitions)

**Uncertainty summary (full grid):** 216 predictions, 216 active (1.0000); 7 unique keys (C maj, D maj, E maj, F maj, F# maj, G maj, A maj); key switches 38/215 (0.1767); large jumps 20 (0.0930).

**tie_count summary:** mean=2.60 tied keys/active window, max=14. (Structurally saturated by relative major/minor scale-template ties, per Phase 3B -- `low_margin` alone is not treated as a difficulty signal here.)

**Anchor mismatch summary:**

- `full_piece` (expected E Minor, confirmed window): n=216, proportion_expected_key=0.0000, mismatch_count=216 across 1 interval(s)
  - tie-break check: expected key exactly tied for the max scale-template score in 135 active window(s); actually selected in 0 of those. **Structurally impossible for the expected minor key to win these ties** (major-key indices 0-11 always beat tied minor-key indices 12-23 under `np.argmax`'s leftmost-tie rule) -- verified, not assumed.

**Inactive/silent window handling:** 0 inactive windows (0.0000 of grid).
  Known compositional silence region t=95.36-99.64s: 0/9 windows inactive there. Genuine, compositionally-intentional silence (Chopin's famous dramatic pause near the piece's end), confirmed in Phase 3F.8 -- NOT file corruption or a pipeline bug. Expected to register as an inactive/undefined-key stretch under this workspace's active-window convention; do not misdiagnose as a baseline failure.

Outputs: `05_Figures_Results/PHASE3G_A_Chopin_Op28No4_key_trajectory.png`, `05_Figures_Results/PHASE3G_A_Chopin_Op28No4_circle_of_fifths.png`

## L5 — Clementi — Sonatina Op. 36 No. 1, I (exposition, [0.0, 17.3]s)

- Role: Short, clear, single modulation (C major -> G major). The exact transition time within the excerpt is NOT confirmed, so the two anchors below are an approximate first-half/second-half split, reported descriptively (tonic vs. dominant-region behavior) -- not a scored, confirmed boundary. Do not overclaim precision from these anchors.
- MIDI: `03_MIDI_Data/candidate_intermediate_midi/excerpts/Clementi_Op36_No1_I_exposition_norepeat_0_17p3s.mid`
- Duration: 17.30s, 34 chroma windows, baseline_offset_windows=0

**Predicted key sequence:** 34 predictions, 2 unique keys.
Dominant predicted keys: C maj (67.6%), G maj (32.4%)

**Circle-of-Fifths jumps (baseline sequence):** mean=0.06, max=1.00, large jumps (>=3): 0 (0.0000 of transitions)

**Uncertainty summary (full grid):** 34 predictions, 34 active (1.0000); 2 unique keys (C maj, G maj); key switches 2/33 (0.0606); large jumps 0 (0.0000).

**tie_count summary:** mean=4.29 tied keys/active window, max=14. (Structurally saturated by relative major/minor scale-template ties, per Phase 3B -- `low_margin` alone is not treated as a difficulty signal here.)

**Anchor mismatch summary:**

- `approx_first_half` (expected C Major, approximate/unconfirmed window): n=18, proportion_expected_key=0.7222, mismatch_count=5 across 1 interval(s)
- `approx_second_half` (expected G Major, approximate/unconfirmed window): n=16, proportion_expected_key=0.3750, mismatch_count=10 across 1 interval(s)

**Full predicted-key run sequence** (not just the half-split proportions above -- shows whether the trajectory is a clean monotonic modulation or something messier):

- C maj: t=0.00-6.00s (13 windows)
- G maj: t=6.50-11.50s (11 windows)
- C maj: t=12.00-16.50s (10 windows)

**Inactive/silent window handling:** 0 inactive windows (0.0000 of grid).

Outputs: `05_Figures_Results/PHASE3G_A_Clementi_excerpt_key_trajectory.png`, `05_Figures_Results/PHASE3G_A_Clementi_excerpt_circle_of_fifths.png`

## L6 — Twinkle 12.mid (Mozart 12 Variations)

- Role: High-stress ornamented variation / modulation stress test, full piece, real embedded key signatures C->Eb->C.
- MIDI: `03_MIDI_Data/Twinkle 12.mid`
- Duration: 687.47s, 1374 chroma windows, baseline_offset_windows=103

**Predicted key sequence:** 1271 predictions, 4 unique keys.
Dominant predicted keys: C maj (90.9%), D# maj (7.1%), G maj (1.9%), A# maj (0.2%)

**Circle-of-Fifths jumps (baseline sequence):** mean=0.02, max=3.00, large jumps (>=3): 5 (0.0039 of transitions)

**Uncertainty summary (full grid):** 1374 predictions, 1271 active (0.9250); 4 unique keys (C maj, D# maj, G maj, A# maj); key switches 11/1270 (0.0087); large jumps 5 (0.0039).

**tie_count summary:** mean=6.71 tied keys/active window, max=14. (Structurally saturated by relative major/minor scale-template ties, per Phase 3B -- `low_margin` alone is not treated as a difficulty signal here.)

**Anchor mismatch summary:**

- `pre_384s` (expected C Major, confirmed window): n=768, proportion_expected_key=1.0000, mismatch_count=0 across 0 interval(s)
- `384_to_432s` (expected Eb Major, confirmed window): n=96, proportion_expected_key=0.8958, mismatch_count=10 across 3 interval(s)
- `post_432s` (expected C Major, confirmed window): n=510, proportion_expected_key=0.9412, mismatch_count=30 across 3 interval(s)

**Inactive/silent window handling:** 103 inactive windows (0.0750 of grid).

Outputs: `05_Figures_Results/PHASE3G_A_Twinkle12_key_trajectory.png`, `05_Figures_Results/PHASE3G_A_Twinkle12_circle_of_fifths.png`

## Cross-piece findings

**1. Does the pitch-class baseline generalize beyond C major?** On L2 (Bach, G major, full piece), proportion_expected_key (G Major) = 0.4207. The baseline does not reliably recover a non-C tonic even on a clean, lightly-accompanied piece, suggesting Phase 2D's Twinkle.mid result may have been C-major-specific rather than evidence of general tonic recovery.

**2. Does it fail on minor-key pieces due to relative major/minor ambiguity?** On L3 (Für Elise excerpt, A minor), proportion_expected_key = 0.0000. Verified structurally, not just observed: the expected minor key exactly tied the max scale-template score in 76 active window(s), and was selected in 0 of them -- because major-key indices (0-11) always beat a tied minor-key index (12-23) under `np.argmax`'s leftmost-tie rule, the baseline is structurally incapable of ever choosing this minor tonic on a tied window, independent of how much real evidence favors it. On L4 (Chopin No. 4, E minor), proportion_expected_key = 0.0000. Verified structurally, not just observed: the expected minor key exactly tied the max scale-template score in 135 active window(s), and was selected in 0 of them -- because major-key indices (0-11) always beat a tied minor-key index (12-23) under `np.argmax`'s leftmost-tie rule, the baseline is structurally incapable of ever choosing this minor tonic on a tied window, independent of how much real evidence favors it. Both minor-key pieces show a proportion_expected_key of exactly 0.0 -- the baseline never once predicts the true minor tonic for either piece, always resolving to a relative-major or other major-key candidate instead. This goes beyond Phase 3B's structural observation that scale-template scores tie across relative major/minor pairs: the tie-break checks above confirm the resolution is not close-but-wrong noise, it is a deterministic consequence of `np.argmax` + the major-keys-first index ordering, which makes any tied minor key categorically unselectable.

**3. Does Clementi show a usable single-modulation challenge?** The exact transition time within the excerpt is not confirmed. The actual predicted-key run sequence (not just coarse half-split proportions) is: C maj (0.0-6.0s) -> G maj (6.5-11.5s) -> C maj (12.0-16.5s). This is **not** a clean monotonic C major -> G major modulation -- the baseline oscillates back to C major after visiting G major, rather than settling on the dominant key. Real dominant-key responsiveness is present (G major is reached), but the excerpt (at this window/threshold setting) does not cleanly isolate a single one-way modulation the way a simple before/after anchor split would assume. Any future use of Clementi as a single-modulation benchmark should account for this oscillation rather than treating the coarse half-split proportions reported above at face value.

**4. Which pieces create local Stage 1 failures suitable for later Stage 4?** Pieces with nonzero key-switch or large-jump activity: L2 (Bach — Minuet in G Major, BWV Anh. 114: key_switch_proportion=0.0736, large_jump_proportion=0.0000); L3 (Beethoven — Für Elise (opening excerpt, [0.0, 54.0]s): key_switch_proportion=0.1524, large_jump_proportion=0.1429); L4 (Chopin — Prelude in E minor, Op. 28 No. 4: key_switch_proportion=0.1767, large_jump_proportion=0.0930); L5 (Clementi — Sonatina Op. 36 No. 1, I (exposition, [0.0, 17.3]s): key_switch_proportion=0.0606, large_jump_proportion=0.0000); L6 (Twinkle 12.mid (Mozart 12 Variations): key_switch_proportion=0.0087, large_jump_proportion=0.0039). These, together with any anchor_mismatch intervals reported per piece above, are the natural first candidates for a future targeted local refinement, per Phase 3D's original recommendation to look for intermediate-difficulty examples rather than the two globally-easy/globally-biased original pieces.

**5. Should Stage 4 remain deferred after Phase 3G-A?** This script performs Stage 1 + Stage 3 only (pitch-class baseline + uncertainty diagnostics) -- no chord-id EMA/SRN disagreement comparison (Phase 3C-style) has been run on this new corpus, so the question of whether disagreement with a recurrent chord-id model is local or global (Phase 3D's actual deciding factor for the original two pieces) remains unanswered here. Per HANDOFF_PHASE3G.md's explicit scope, this task does not implement, and does not recommend implementing, any Chroma SRN, Transformer, or neural refinement. Whether Stage 4 should be reconsidered is a Phase 3G-B/3H question, contingent on running that disagreement comparison on the newly-identified candidate regions from finding 4 above.

## Scope note

This is Phase 3G-A only: pitch-class/chroma baseline evaluation + uncertainty diagnostics on the full corpus. No chord-id EMA/SRN comparison, no Chroma SRN, no Transformer, and no neural refinement has been implemented. No existing Phase 1/1.5/2/3 script was modified, and no existing Phase 1/1.5/2/3 output was regenerated -- all outputs here are new, under a `PHASE3G_A_` prefix or in a new `03_MIDI_Data/derived_phase3g_corpus/` directory.
