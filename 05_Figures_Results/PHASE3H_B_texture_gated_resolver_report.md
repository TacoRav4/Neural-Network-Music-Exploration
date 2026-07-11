# Phase 3H-B — Texture-Gated Non-Neural Tonic/Mode Resolver

Tests whether Phase 3H-A's weighted key-profile resolver (variant D) can be applied *conditionally* -- only when local evidence is dense and stable enough to trust -- so its minor-key-recovery benefit (Für Elise, Chopin) is captured without its monophonic-stability cost (Twinkle, Twinkle 12). **Still not a neural-modeling phase.** Phase 3G-A, 3G-B, and 3H-A are treated as frozen and are not modified or overwritten; their scripts are only imported from, their output files only read.

## Variant E's design

**Default backbone: variant C (tie-aware continuity).** Documented choice, not incidental -- Phase 3H-A validated C as switch-reducing with zero effect on minor-key recovery, making it the natural stable base to gate weighted swaps on top of (rather than raw, ungated A).

**Three predeclared, label-free gates, ALL of which must pass for a window to use the weighted (D) prediction instead of C's:**

1. **Density**: active pitch-class count (nonzero entries in thresholded smoothed chroma) > 2. Exactly the threshold specified by the task.
2. **Collection stability**: C's own diatonic collection (relative major/minor collapsed) must be constant over the 4 predictions (2.0s) immediately preceding this window -- causal, no lookahead, uses only C's own history, never a label.
3. **Weighted margin**: the weighted profile's own normalized top1-vs-top2 margin >= 0.2 (pitch_class_uncertainty_diagnostics.DEFAULT_LOW_MARGIN_THRESHOLD (reused verbatim, not re-tuned)).

None of these three numbers were adjusted after seeing this run's expected-key results -- gate 1's threshold is specified directly by the task, gate 3 reuses an existing Phase 3B constant verbatim, and gate 2's window length was fixed at a round, conservative value before running the piece corpus.

## Per-piece, per-variant results

### L1 — Twinkle.mid

**A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)**
  - piece-level: 106 predictions, 106 active (1.0000), 1 unique keys; dominant: C maj (100.0%)
  - key switches: 0/105 (0.0000); jumps: mean=0.00, max=0.00, large=0 (0.0000)
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_piece` (expected C Major): n_defined=106, strict=1.0000, collection_equiv=1.0000, tied_for_max=106 (selected=106, lost_to_tiebreak=0), minor_mode_fraction=0.0000

**D: weighted key-profile matcher (Phase 3H-A's fixed profile, unchanged)**
  - piece-level: 106 predictions, 106 active (1.0000), 7 unique keys; dominant: C maj (35.8%), D min (17.9%), G maj (13.2%), E min (11.3%), F maj (10.4%)
  - key switches: 45/105 (0.4286); jumps: mean=0.78, max=5.00, large=8 (0.0762)
  - minor-mode predictions: 43 (0.4057 of defined windows)
  - `full_piece` (expected C Major): n_defined=106, strict=0.3585, collection_equiv=0.4245, tied_for_max=38 (selected=38, lost_to_tiebreak=0), minor_mode_fraction=0.4057

**E: texture-gated resolver (default C, weighted-profile swap only when density+stability+margin gates all pass)**
  - piece-level: 106 predictions, 106 active (1.0000), 2 unique keys; dominant: C maj (99.1%), D min (0.9%)
  - key switches: 2/105 (0.0190); jumps: mean=0.04, max=2.00, large=0 (0.0000)
  - minor-mode predictions: 1 (0.0094 of defined windows)
  - `full_piece` (expected C Major): n_defined=106, strict=0.9906, collection_equiv=0.9906, tied_for_max=105 (selected=105, lost_to_tiebreak=0), minor_mode_fraction=0.0094
  - **gate usage**: gate1(density)=100/106 (0.9434), gate2(stability)=102/106 (0.9623), gate3(margin)=1/106 (0.0094), all-3-pass=1 (0.0094), **actual swaps from C default=1** (0.0094), of which minor-mode=1

*(Context only -- C, E's default backbone: 0 key switches, 0.0 minor-mode fraction.)*

### L2 — Bach — Minuet in G Major, BWV Anh. 114

**A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)**
  - piece-level: 164 predictions, 164 active (1.0000), 3 unique keys; dominant: G maj (42.1%), C maj (29.9%), D maj (28.0%)
  - key switches: 12/163 (0.0736); jumps: mean=0.08, max=2.00, large=0 (0.0000)
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_piece` (expected G Major): n_defined=164, strict=0.4207, collection_equiv=0.4207, tied_for_max=142 (selected=69, lost_to_tiebreak=73), minor_mode_fraction=0.0000
  - Bach tonic-neighborhood: 95/95 mismatches within C/G/D (1.0000)

**D: weighted key-profile matcher (Phase 3H-A's fixed profile, unchanged)**
  - piece-level: 164 predictions, 164 active (1.0000), 8 unique keys; dominant: G maj (60.4%), D maj (25.6%), C maj (5.5%), A min (3.7%), B min (1.8%)
  - key switches: 50/163 (0.3067); jumps: mean=0.50, max=4.00, large=10 (0.0613)
  - minor-mode predictions: 13 (0.0793 of defined windows)
  - `full_piece` (expected G Major): n_defined=164, strict=0.6037, collection_equiv=0.6159, tied_for_max=99 (selected=99, lost_to_tiebreak=0), minor_mode_fraction=0.0793
  - Bach tonic-neighborhood: 51/65 mismatches within C/G/D (0.7846)

**E: texture-gated resolver (default C, weighted-profile swap only when density+stability+margin gates all pass)**
  - piece-level: 164 predictions, 164 active (1.0000), 3 unique keys; dominant: G maj (76.2%), D maj (15.2%), C maj (8.5%)
  - key switches: 7/163 (0.0429); jumps: mean=0.04, max=1.00, large=0 (0.0000)
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_piece` (expected G Major): n_defined=164, strict=0.7622, collection_equiv=0.7622, tied_for_max=141 (selected=125, lost_to_tiebreak=16), minor_mode_fraction=0.0000
  - **gate usage**: gate1(density)=164/164 (1.0000), gate2(stability)=145/164 (0.8841), gate3(margin)=9/164 (0.0549), all-3-pass=9 (0.0549), **actual swaps from C default=1** (0.0061), of which minor-mode=0
  - Bach tonic-neighborhood: 39/39 mismatches within C/G/D (1.0000)

*(Context only -- C, E's default backbone: 5 key switches, 0.0 minor-mode fraction.)*

### L3 — Beethoven — Für Elise (opening excerpt, [0.0, 54.0]s)

**A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)**
  - piece-level: 107 predictions, 106 active (0.9907), 4 unique keys; dominant: C maj (71.7%), E maj (19.8%), A maj (4.7%), A# maj (3.8%)
  - key switches: 16/105 (0.1524); jumps: mean=0.59, max=6.00, large=15 (0.1429)
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_excerpt` (expected A Minor): n_defined=106, strict=0.0000, collection_equiv=0.7170, tied_for_max=76 (selected=0, lost_to_tiebreak=76), minor_mode_fraction=0.0000

**D: weighted key-profile matcher (Phase 3H-A's fixed profile, unchanged)**
  - piece-level: 107 predictions, 106 active (0.9907), 8 unique keys; dominant: A min (41.5%), E min (30.2%), E maj (16.0%), C maj (4.7%), A maj (3.8%)
  - key switches: 44/105 (0.4190); jumps: mean=0.58, max=5.00, large=7 (0.0667)
  - minor-mode predictions: 78 (0.7358 of defined windows)
  - `full_excerpt` (expected A Minor): n_defined=106, strict=0.4151, collection_equiv=0.4623, tied_for_max=48 (selected=44, lost_to_tiebreak=4), minor_mode_fraction=0.7358

**E: texture-gated resolver (default C, weighted-profile swap only when density+stability+margin gates all pass)**
  - piece-level: 107 predictions, 106 active (0.9907), 4 unique keys; dominant: C maj (71.7%), E maj (19.8%), A maj (4.7%), A# maj (3.8%)
  - key switches: 16/105 (0.1524); jumps: mean=0.59, max=6.00, large=15 (0.1429)
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_excerpt` (expected A Minor): n_defined=106, strict=0.0000, collection_equiv=0.7170, tied_for_max=76 (selected=0, lost_to_tiebreak=76), minor_mode_fraction=0.0000
  - **gate usage**: gate1(density)=104/107 (0.9720), gate2(stability)=67/107 (0.6262), gate3(margin)=1/107 (0.0093), all-3-pass=0 (0.0000), **actual swaps from C default=0** (0.0000), of which minor-mode=0

*(Context only -- C, E's default backbone: 16 key switches, 0.0 minor-mode fraction.)*

### L4 — Chopin — Prelude in E minor, Op. 28 No. 4

**A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)**
  - piece-level: 216 predictions, 216 active (1.0000), 7 unique keys; dominant: G maj (49.1%), C maj (24.1%), E maj (12.5%), D maj (6.5%), A maj (3.2%)
  - key switches: 38/215 (0.1767); jumps: mean=0.44, max=6.00, large=20 (0.0930)
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_piece` (expected E Minor): n_defined=216, strict=0.0000, collection_equiv=0.4907, tied_for_max=135 (selected=0, lost_to_tiebreak=135), minor_mode_fraction=0.0000
  - Chopin silence region: 0/9 inactive windows

**D: weighted key-profile matcher (Phase 3H-A's fixed profile, unchanged)**
  - piece-level: 216 predictions, 216 active (1.0000), 12 unique keys; dominant: E min (40.7%), A min (24.1%), B maj (12.5%), E maj (6.0%), D min (5.1%)
  - key switches: 51/215 (0.2372); jumps: mean=0.34, max=4.00, large=10 (0.0465)
  - minor-mode predictions: 158 (0.7315 of defined windows)
  - `full_piece` (expected E Minor): n_defined=216, strict=0.4074, collection_equiv=0.4074, tied_for_max=90 (selected=88, lost_to_tiebreak=2), minor_mode_fraction=0.7315
  - Chopin silence region: 0/9 inactive windows

**E: texture-gated resolver (default C, weighted-profile swap only when density+stability+margin gates all pass)**
  - piece-level: 216 predictions, 216 active (1.0000), 9 unique keys; dominant: G maj (51.9%), C maj (16.2%), E maj (12.5%), D maj (6.9%), A maj (3.2%)
  - key switches: 38/215 (0.1767); jumps: mean=0.50, max=6.00, large=26 (0.1209)
  - minor-mode predictions: 10 (0.0463 of defined windows)
  - `full_piece` (expected E Minor): n_defined=216, strict=0.0185, collection_equiv=0.5370, tied_for_max=133 (selected=4, lost_to_tiebreak=129), minor_mode_fraction=0.0463
  - **gate usage**: gate1(density)=214/216 (0.9907), gate2(stability)=129/216 (0.5972), gate3(margin)=10/216 (0.0463), all-3-pass=10 (0.0463), **actual swaps from C default=10** (0.0463), of which minor-mode=10
  - Chopin silence region: 0/9 inactive windows

*(Context only -- C, E's default backbone: 32 key switches, 0.0 minor-mode fraction.)*

### L5 — Clementi — Sonatina Op. 36 No. 1, I (exposition, [0.0, 17.3]s)

**A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)**
  - piece-level: 34 predictions, 34 active (1.0000), 2 unique keys; dominant: C maj (67.6%), G maj (32.4%)
  - key switches: 2/33 (0.0606); jumps: mean=0.06, max=1.00, large=0 (0.0000)
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `approx_first_half` (expected C Major): n_defined=18, strict=0.7222, collection_equiv=0.7222, tied_for_max=13 (selected=13, lost_to_tiebreak=0), minor_mode_fraction=0.0000
  - `approx_second_half` (expected G Major): n_defined=16, strict=0.3750, collection_equiv=0.3750, tied_for_max=16 (selected=6, lost_to_tiebreak=10), minor_mode_fraction=0.0000
  - Clementi run sequence: C maj -> G maj -> C maj (exact C->G->C: True)

**D: weighted key-profile matcher (Phase 3H-A's fixed profile, unchanged)**
  - piece-level: 34 predictions, 34 active (1.0000), 5 unique keys; dominant: C maj (47.1%), A min (23.5%), D min (17.6%), D maj (8.8%), G maj (2.9%)
  - key switches: 11/33 (0.3333); jumps: mean=0.67, max=3.00, large=5 (0.1515)
  - minor-mode predictions: 14 (0.4118 of defined windows)
  - `approx_first_half` (expected C Major): n_defined=18, strict=0.8333, collection_equiv=1.0000, tied_for_max=15 (selected=15, lost_to_tiebreak=0), minor_mode_fraction=0.1667
  - `approx_second_half` (expected G Major): n_defined=16, strict=0.0625, collection_equiv=0.0625, tied_for_max=1 (selected=1, lost_to_tiebreak=0), minor_mode_fraction=0.6875
  - Clementi run sequence: C maj -> A min -> C maj -> A min -> D maj -> G maj -> A min -> D min -> A min -> C maj -> A min -> D min (exact C->G->C: False)

**E: texture-gated resolver (default C, weighted-profile swap only when density+stability+margin gates all pass)**
  - piece-level: 34 predictions, 34 active (1.0000), 2 unique keys; dominant: G maj (61.8%), C maj (38.2%)
  - key switches: 1/33 (0.0303); jumps: mean=0.03, max=1.00, large=0 (0.0000)
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `approx_first_half` (expected C Major): n_defined=18, strict=0.7222, collection_equiv=0.7222, tied_for_max=13 (selected=13, lost_to_tiebreak=0), minor_mode_fraction=0.0000
  - `approx_second_half` (expected G Major): n_defined=16, strict=1.0000, collection_equiv=1.0000, tied_for_max=16 (selected=16, lost_to_tiebreak=0), minor_mode_fraction=0.0000
  - **gate usage**: gate1(density)=32/34 (0.9412), gate2(stability)=27/34 (0.7941), gate3(margin)=0/34 (0.0000), all-3-pass=0 (0.0000), **actual swaps from C default=0** (0.0000), of which minor-mode=0
  - Clementi run sequence: C maj -> G maj (exact C->G->C: False)

*(Context only -- C, E's default backbone: 1 key switches, 0.0 minor-mode fraction.)*

### L6 — Twinkle 12.mid (Mozart 12 Variations)

**A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)**
  - piece-level: 1374 predictions, 1271 active (0.9250), 4 unique keys; dominant: C maj (90.9%), D# maj (7.1%), G maj (1.9%), A# maj (0.2%)
  - key switches: 11/1270 (0.0087); jumps: mean=0.02, max=3.00, large=5 (0.0039)
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `pre_384s` (expected C Major): n_defined=665, strict=1.0000, collection_equiv=1.0000, tied_for_max=665 (selected=665, lost_to_tiebreak=0), minor_mode_fraction=0.0000
  - `384_to_432s` (expected Eb Major): n_defined=96, strict=0.8958, collection_equiv=0.8958, tied_for_max=93 (selected=86, lost_to_tiebreak=7), minor_mode_fraction=0.0000
  - `post_432s` (expected C Major): n_defined=510, strict=0.9412, collection_equiv=0.9412, tied_for_max=480 (selected=480, lost_to_tiebreak=0), minor_mode_fraction=0.0000

**D: weighted key-profile matcher (Phase 3H-A's fixed profile, unchanged)**
  - piece-level: 1374 predictions, 1271 active (0.9250), 13 unique keys; dominant: C maj (56.2%), G maj (12.4%), D min (11.0%), G min (6.8%), F maj (6.1%)
  - key switches: 133/1270 (0.1047); jumps: mean=0.21, max=5.00, large=48 (0.0378)
  - minor-mode predictions: 271 (0.2132 of defined windows)
  - `pre_384s` (expected C Major): n_defined=665, strict=0.6511, collection_equiv=0.6632, tied_for_max=433 (selected=433, lost_to_tiebreak=0), minor_mode_fraction=0.0481
  - `384_to_432s` (expected Eb Major): n_defined=96, strict=0.0417, collection_equiv=0.1875, tied_for_max=4 (selected=4, lost_to_tiebreak=0), minor_mode_fraction=0.8229
  - `post_432s` (expected C Major): n_defined=510, strict=0.5451, collection_equiv=0.5569, tied_for_max=278 (selected=278, lost_to_tiebreak=0), minor_mode_fraction=0.3137

**E: texture-gated resolver (default C, weighted-profile swap only when density+stability+margin gates all pass)**
  - piece-level: 1374 predictions, 1271 active (0.9250), 7 unique keys; dominant: C maj (90.1%), D# maj (7.4%), G maj (1.9%), A# maj (0.2%), F min (0.2%)
  - key switches: 15/1270 (0.0118); jumps: mean=0.02, max=3.00, large=1 (0.0008)
  - minor-mode predictions: 3 (0.0024 of defined windows)
  - `pre_384s` (expected C Major): n_defined=665, strict=1.0000, collection_equiv=1.0000, tied_for_max=665 (selected=665, lost_to_tiebreak=0), minor_mode_fraction=0.0000
  - `384_to_432s` (expected Eb Major): n_defined=96, strict=0.9375, collection_equiv=0.9375, tied_for_max=91 (selected=90, lost_to_tiebreak=1), minor_mode_fraction=0.0208
  - `post_432s` (expected C Major): n_defined=510, strict=0.9333, collection_equiv=0.9333, tied_for_max=477 (selected=476, lost_to_tiebreak=1), minor_mode_fraction=0.0020
  - **gate usage**: gate1(density)=820/1374 (0.5968), gate2(stability)=1246/1374 (0.9068), gate3(margin)=23/1374 (0.0167), all-3-pass=22 (0.0160), **actual swaps from C default=5** (0.0036), of which minor-mode=3

*(Context only -- C, E's default backbone: 7 key switches, 0.0 minor-mode fraction.)*

## Cross-variant findings

### L3/L4 minor-key recovery by variant

- **L3_full_excerpt_A_minor**: A_control: strict=0.0000, collection=0.7170; D_weighted_profile: strict=0.4151, collection=0.4623; E_texture_gated: strict=0.0000, collection=0.7170
- **L4_full_piece_E_minor**: A_control: strict=0.0000, collection=0.4907; D_weighted_profile: strict=0.4074, collection=0.4074; E_texture_gated: strict=0.0185, collection=0.5370

### L1/L6 stability by variant

- **L1 full_piece (C major)**: A_control: strict=1.0000, switches=0, large_jumps=0; D_weighted_profile: strict=0.3585, switches=45, large_jumps=8; E_texture_gated: strict=0.9906, switches=2, large_jumps=0
- **L6_pre_384s**: A_control: strict=1.0000; D_weighted_profile: strict=0.6511; E_texture_gated: strict=1.0000
- **L6_384_to_432s**: A_control: strict=0.8958; D_weighted_profile: strict=0.0417; E_texture_gated: strict=0.9375
- **L6_post_432s**: A_control: strict=0.9412; D_weighted_profile: strict=0.5451; E_texture_gated: strict=0.9333
- **L6 overall**: A_control: switches=11, large_jumps=5; D_weighted_profile: switches=133, large_jumps=48; E_texture_gated: switches=15, large_jumps=1

### Gate usage by piece (variant E)

- **L1**: gate1=0.9434, gate2=0.9623, gate3=0.0094, all-3=0.0094, actual_swaps=1 (0.0094), minor-mode swaps=1
- **L2**: gate1=1.0000, gate2=0.8841, gate3=0.0549, all-3=0.0549, actual_swaps=1 (0.0061), minor-mode swaps=0
- **L3**: gate1=0.9720, gate2=0.6262, gate3=0.0093, all-3=0.0000, actual_swaps=0 (0.0000), minor-mode swaps=0
- **L4**: gate1=0.9907, gate2=0.5972, gate3=0.0463, all-3=0.0463, actual_swaps=10 (0.0463), minor-mode swaps=10
- **L5**: gate1=0.9412, gate2=0.7941, gate3=0.0000, all-3=0.0000, actual_swaps=0 (0.0000), minor-mode swaps=0
- **L6**: gate1=0.5968, gate2=0.9068, gate3=0.0167, all-3=0.0160, actual_swaps=5 (0.0036), minor-mode swaps=3

### Bach (L2) tonic-neighborhood behavior by variant

- A_control: 95/95 mismatches within C/G/D (1.0000)
- D_weighted_profile: 51/65 mismatches within C/G/D (0.7846)
- E_texture_gated: 39/39 mismatches within C/G/D (1.0000)

### Clementi (L5) run behavior by variant

- A_control: C maj -> G maj -> C maj (exact C->G->C: True)
- D_weighted_profile: C maj -> A min -> C maj -> A min -> D maj -> G maj -> A min -> D min -> A min -> C maj -> A min -> D min (exact C->G->C: False)
- E_texture_gated: C maj -> G maj (exact C->G->C: False)

### Chopin (L4) silence-window handling by variant

- A_control: 0/9 inactive
- D_weighted_profile: 0/9 inactive
- E_texture_gated: 0/9 inactive

## Gate bottleneck analysis

Across all 6 pieces, gate pass-rate ranges are: gate1 (density) 0.597-1.000, gate2 (stability) 0.597-0.962, gate3 (weighted margin) 0.000-0.055. **Gate 3 is overwhelmingly the limiting factor everywhere** -- density and stability pass on a majority of windows for most pieces, but the weighted-margin requirement (reusing Phase 3B's 0.20 threshold, applied to the weighted representation) is cleared on well under 6% of windows across the entire corpus. This means variant E, as gated here, ends up very close to variant C almost everywhere -- the weighted profile is being deferred to only in the rare windows where its own evidence is unusually decisive. This is a direct, mechanical consequence of reusing an existing, non-tuned threshold value rather than fitting one to this corpus (as instructed) -- a looser or differently-defined margin gate would likely permit more swaps (and might shift the L3/L4-vs-L1/L6 trade-off in either direction), but choosing such a threshold from this run's own results would be exactly the kind of tuning-against-expected-key-results the task explicitly prohibits. This finding is reported transparently rather than adjusted.

## Verdict

**L3 (Für Elise, A minor) strict proportion: A=0.0000 -> E=0.0000 (NOT improved). L4 (Chopin, E minor) strict proportion: A=0.0000 -> E=0.0185 (IMPROVED). L1 (Twinkle) strict proportion: A=1.0000 -> E=0.9906, switches: A=0 -> E=2 (DEGRADED). L6 per-anchor strict deltas (E - A): pre_384s: +0.0000, 384_to_432s: +0.0417, post_432s: -0.0078 (STABLE/preserved). Gate usage (actual swaps to weighted prediction): L1=1, L6=5 (destabilization sources in D), L3=0, L4=10 (where minor recovery must come from, if it improves). OVERALL: texture-gated resolution (E) does NOT achieve both improved strict minor-key recovery and preserved L1/L6 stability simultaneously -- see the deltas and gate-usage counts above for exactly where gating over- or under-fires.**

- minor_recovery_improved_L3_via_E = False
- minor_recovery_improved_L4_via_E = True
- L1_stability_preserved_via_E = False
- L6_stability_preserved_via_E = True
- overall_success_E = False

## Plots

- `05_Figures_Results/PHASE3H_B_Twinkle_variant_comparison_key_trajectory.png`
- `05_Figures_Results/PHASE3H_B_FurElise_excerpt_variant_comparison_key_trajectory.png`
- `05_Figures_Results/PHASE3H_B_Chopin_Op28No4_variant_comparison_key_trajectory.png`
- `05_Figures_Results/PHASE3H_B_Twinkle12_variant_comparison_key_trajectory.png`

## Scope note

This is Phase 3H-B only: a texture-gated combination of Phase 3H-A's already-existing A/C/D variants. No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement was run or implemented. `np.argmax`'s tie-break rule and Phase 3H-A's weighted-profile weights were never changed -- variant D is reproduced verbatim via `variant_D_weighted_profile` (imported, unmodified). Anchors were used exclusively inside the evaluation functions (`compute_anchor_metrics`, `bach_tonic_neighborhood`), never inside any gate or any variant's key_id computation -- `variant_E_texture_gated` and its three gate functions take no anchor or expected-key argument at all.
