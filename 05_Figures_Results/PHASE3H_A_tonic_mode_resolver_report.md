# Phase 3H-A — Non-Neural Tonic/Mode Resolver Ablations

Tests whether the pitch-class fast filter's tonic/mode ambiguity (Phase 3G-A/3G-B: it is a diatonic-*collection* resolver, not a tonic/mode resolver, because unweighted `SCALE_TEMPLATES` gives relative major/minor pairs identical rows and `np.argmax`'s leftmost-tie convention then always prefers the major-indexed key) can be improved by small, interpretable, non-neural decision-rule variants. **Still not a neural-modeling phase** -- no chord-id EMA/SRN, no Chroma SRN, no Transformer, no neural refinement. Phase 3G-A and Phase 3G-B are treated as frozen and are not modified or overwritten; their scripts are only imported from, their output files only read.

## Variants

- A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)
- C: tie-aware continuity rule (same unweighted evidence as A, continuity-preferring tie-break)
- D: weighted key-profile matcher (functional-role-weighted templates, plain argmax)
- B: collection-level evaluation is not a separate predictor -- it is the `strict` vs. `collection_equiv` proportion pair reported for every anchor of every variant below, using Phase 3G-B's own `collection_equivalent_key_id` (imported, not redefined).

## Variant D's weighted key-profile (fixed, pre-declared, not tuned on this corpus)

- Degree roles (in order): tonic, supertonic, mediant, subdominant, dominant, submediant, leading-tone/subtonic
- Weights (aligned to the roles above): [5, 1, 3, 2, 4, 1, 1]
- Major scale degree offsets (semitones from tonic): [0, 2, 4, 5, 7, 9, 11]; minor (natural): [0, 2, 3, 5, 7, 8, 10] -- both lists copied verbatim from `pitch_class_baseline.build_scale_templates()`, only the assigned value per degree changes (weight instead of a flat 1).
- Rationale: tonic defines the key center (highest weight); dominant is the primary harmonic pillar (second-highest); mediant is the single pitch that distinguishes major from minor by ear, and is also exactly the scale degree whose offset differs between major (+4) and minor (+3) -- the mechanism that breaks the relative-key tie; the remaining diatonic degrees (supertonic, subdominant, submediant, leading-tone/subtonic) are weighted equally and lowest, since none individually defines tonic/mode.
- **Structural verification (not corpus-dependent)**: of the 12 relative major/minor pairs, 0 still have identical weighted rows (expected: 0). All 24 rows of `WEIGHTED_TEMPLATES` are pairwise distinct: True (24/24 unique).

## Per-piece, per-variant results

### L1 — Twinkle.mid

(Control reproduction sanity check -- variant A's fresh key_switch/jump_distance recomputation exactly matches Phase 3G-A's own frozen arrays: True)

**A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)**
  - piece-level: 106 predictions, 106 active (1.0000), 1 unique keys; dominant: C maj (100.0%)
  - key switches: 0/105 (0.0000); jumps: mean=0.00, max=0.00, large=0 (0.0000); tie_count: mean=5.23, max=14
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_piece` (expected C Major): n_defined=106, strict=1.0000, collection_equiv=1.0000, tied_for_max=106 (selected=106, lost_to_tiebreak=0), minor_mode_fraction=0.0000

**C: tie-aware continuity rule (same unweighted evidence as A, continuity-preferring tie-break)**
  - piece-level: 106 predictions, 106 active (1.0000), 1 unique keys; dominant: C maj (100.0%)
  - key switches: 0/105 (0.0000); jumps: mean=0.00, max=0.00, large=0 (0.0000); tie_count: mean=5.23, max=14
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_piece` (expected C Major): n_defined=106, strict=1.0000, collection_equiv=1.0000, tied_for_max=106 (selected=106, lost_to_tiebreak=0), minor_mode_fraction=0.0000

**D: weighted key-profile matcher (functional-role-weighted templates, plain argmax)**
  - piece-level: 106 predictions, 106 active (1.0000), 7 unique keys; dominant: C maj (35.8%), D min (17.9%), G maj (13.2%), E min (11.3%), F maj (10.4%)
  - key switches: 45/105 (0.4286); jumps: mean=0.78, max=5.00, large=8 (0.0762); tie_count: mean=1.12, max=2
  - minor-mode predictions: 43 (0.4057 of defined windows)
  - `full_piece` (expected C Major): n_defined=106, strict=0.3585, collection_equiv=0.4245, tied_for_max=38 (selected=38, lost_to_tiebreak=0), minor_mode_fraction=0.4057
  - active mask matches frozen control exactly: True

### L2 — Bach — Minuet in G Major, BWV Anh. 114

(Control reproduction sanity check -- variant A's fresh key_switch/jump_distance recomputation exactly matches Phase 3G-A's own frozen arrays: True)

**A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)**
  - piece-level: 164 predictions, 164 active (1.0000), 3 unique keys; dominant: G maj (42.1%), C maj (29.9%), D maj (28.0%)
  - key switches: 12/163 (0.0736); jumps: mean=0.08, max=2.00, large=0 (0.0000); tie_count: mean=3.15, max=6
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_piece` (expected G Major): n_defined=164, strict=0.4207, collection_equiv=0.4207, tied_for_max=142 (selected=69, lost_to_tiebreak=73), minor_mode_fraction=0.0000
  - Bach tonic-neighborhood: 95/95 mismatches within C/G/D (1.0000)

**C: tie-aware continuity rule (same unweighted evidence as A, continuity-preferring tie-break)**
  - piece-level: 164 predictions, 164 active (1.0000), 3 unique keys; dominant: G maj (75.6%), D maj (15.2%), C maj (9.1%)
  - key switches: 5/163 (0.0307); jumps: mean=0.03, max=1.00, large=0 (0.0000); tie_count: mean=3.15, max=6
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_piece` (expected G Major): n_defined=164, strict=0.7561, collection_equiv=0.7561, tied_for_max=142 (selected=124, lost_to_tiebreak=18), minor_mode_fraction=0.0000
  - Bach tonic-neighborhood: 40/40 mismatches within C/G/D (1.0000)

**D: weighted key-profile matcher (functional-role-weighted templates, plain argmax)**
  - piece-level: 164 predictions, 164 active (1.0000), 8 unique keys; dominant: G maj (60.4%), D maj (25.6%), C maj (5.5%), A min (3.7%), B min (1.8%)
  - key switches: 50/163 (0.3067); jumps: mean=0.50, max=4.00, large=10 (0.0613); tie_count: mean=1.00, max=1
  - minor-mode predictions: 13 (0.0793 of defined windows)
  - `full_piece` (expected G Major): n_defined=164, strict=0.6037, collection_equiv=0.6159, tied_for_max=99 (selected=99, lost_to_tiebreak=0), minor_mode_fraction=0.0793
  - active mask matches frozen control exactly: True
  - Bach tonic-neighborhood: 51/65 mismatches within C/G/D (0.7846)

### L3 — Beethoven — Für Elise (opening excerpt, [0.0, 54.0]s)

(Control reproduction sanity check -- variant A's fresh key_switch/jump_distance recomputation exactly matches Phase 3G-A's own frozen arrays: True)

**A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)**
  - piece-level: 107 predictions, 106 active (0.9907), 4 unique keys; dominant: C maj (71.7%), E maj (19.8%), A maj (4.7%), A# maj (3.8%)
  - key switches: 16/105 (0.1524); jumps: mean=0.59, max=6.00, large=15 (0.1429); tie_count: mean=3.55, max=14
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_excerpt` (expected A Minor): n_defined=106, strict=0.0000, collection_equiv=0.7170, tied_for_max=76 (selected=0, lost_to_tiebreak=76), minor_mode_fraction=0.0000

**C: tie-aware continuity rule (same unweighted evidence as A, continuity-preferring tie-break)**
  - piece-level: 107 predictions, 106 active (0.9907), 4 unique keys; dominant: C maj (71.7%), E maj (19.8%), A maj (4.7%), A# maj (3.8%)
  - key switches: 16/105 (0.1524); jumps: mean=0.59, max=6.00, large=15 (0.1429); tie_count: mean=3.55, max=14
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_excerpt` (expected A Minor): n_defined=106, strict=0.0000, collection_equiv=0.7170, tied_for_max=76 (selected=0, lost_to_tiebreak=76), minor_mode_fraction=0.0000

**D: weighted key-profile matcher (functional-role-weighted templates, plain argmax)**
  - piece-level: 107 predictions, 106 active (0.9907), 8 unique keys; dominant: A min (41.5%), E min (30.2%), E maj (16.0%), C maj (4.7%), A maj (3.8%)
  - key switches: 44/105 (0.4190); jumps: mean=0.58, max=5.00, large=7 (0.0667); tie_count: mean=1.05, max=2
  - minor-mode predictions: 78 (0.7358 of defined windows)
  - `full_excerpt` (expected A Minor): n_defined=106, strict=0.4151, collection_equiv=0.4623, tied_for_max=48 (selected=44, lost_to_tiebreak=4), minor_mode_fraction=0.7358
  - active mask matches frozen control exactly: True

### L4 — Chopin — Prelude in E minor, Op. 28 No. 4

(Control reproduction sanity check -- variant A's fresh key_switch/jump_distance recomputation exactly matches Phase 3G-A's own frozen arrays: True)

**A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)**
  - piece-level: 216 predictions, 216 active (1.0000), 7 unique keys; dominant: G maj (49.1%), C maj (24.1%), E maj (12.5%), D maj (6.5%), A maj (3.2%)
  - key switches: 38/215 (0.1767); jumps: mean=0.44, max=6.00, large=20 (0.0930); tie_count: mean=2.60, max=14
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_piece` (expected E Minor): n_defined=216, strict=0.0000, collection_equiv=0.4907, tied_for_max=135 (selected=0, lost_to_tiebreak=135), minor_mode_fraction=0.0000
  - Chopin silence region: 0/9 inactive windows

**C: tie-aware continuity rule (same unweighted evidence as A, continuity-preferring tie-break)**
  - piece-level: 216 predictions, 216 active (1.0000), 7 unique keys; dominant: G maj (53.2%), C maj (19.4%), E maj (12.5%), D maj (6.9%), A maj (3.2%)
  - key switches: 32/215 (0.1488); jumps: mean=0.41, max=6.00, large=21 (0.0977); tie_count: mean=2.60, max=14
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `full_piece` (expected E Minor): n_defined=216, strict=0.0000, collection_equiv=0.5324, tied_for_max=135 (selected=0, lost_to_tiebreak=135), minor_mode_fraction=0.0000
  - Chopin silence region: 0/9 inactive windows

**D: weighted key-profile matcher (functional-role-weighted templates, plain argmax)**
  - piece-level: 216 predictions, 216 active (1.0000), 12 unique keys; dominant: E min (40.7%), A min (24.1%), B maj (12.5%), E maj (6.0%), D min (5.1%)
  - key switches: 51/215 (0.2372); jumps: mean=0.34, max=4.00, large=10 (0.0465); tie_count: mean=1.02, max=2
  - minor-mode predictions: 158 (0.7315 of defined windows)
  - `full_piece` (expected E Minor): n_defined=216, strict=0.4074, collection_equiv=0.4074, tied_for_max=90 (selected=88, lost_to_tiebreak=2), minor_mode_fraction=0.7315
  - active mask matches frozen control exactly: True
  - Chopin silence region: 0/9 inactive windows

### L5 — Clementi — Sonatina Op. 36 No. 1, I (exposition, [0.0, 17.3]s)

(Control reproduction sanity check -- variant A's fresh key_switch/jump_distance recomputation exactly matches Phase 3G-A's own frozen arrays: True)

**A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)**
  - piece-level: 34 predictions, 34 active (1.0000), 2 unique keys; dominant: C maj (67.6%), G maj (32.4%)
  - key switches: 2/33 (0.0606); jumps: mean=0.06, max=1.00, large=0 (0.0000); tie_count: mean=4.29, max=14
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `approx_first_half` (expected C Major): n_defined=18, strict=0.7222, collection_equiv=0.7222, tied_for_max=13 (selected=13, lost_to_tiebreak=0), minor_mode_fraction=0.0000
  - `approx_second_half` (expected G Major): n_defined=16, strict=0.3750, collection_equiv=0.3750, tied_for_max=16 (selected=6, lost_to_tiebreak=10), minor_mode_fraction=0.0000
  - Clementi run sequence: C maj -> G maj -> C maj (exact C->G->C: True)

**C: tie-aware continuity rule (same unweighted evidence as A, continuity-preferring tie-break)**
  - piece-level: 34 predictions, 34 active (1.0000), 2 unique keys; dominant: G maj (61.8%), C maj (38.2%)
  - key switches: 1/33 (0.0303); jumps: mean=0.03, max=1.00, large=0 (0.0000); tie_count: mean=4.29, max=14
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `approx_first_half` (expected C Major): n_defined=18, strict=0.7222, collection_equiv=0.7222, tied_for_max=13 (selected=13, lost_to_tiebreak=0), minor_mode_fraction=0.0000
  - `approx_second_half` (expected G Major): n_defined=16, strict=1.0000, collection_equiv=1.0000, tied_for_max=16 (selected=16, lost_to_tiebreak=0), minor_mode_fraction=0.0000
  - Clementi run sequence: C maj -> G maj (exact C->G->C: False)

**D: weighted key-profile matcher (functional-role-weighted templates, plain argmax)**
  - piece-level: 34 predictions, 34 active (1.0000), 5 unique keys; dominant: C maj (47.1%), A min (23.5%), D min (17.6%), D maj (8.8%), G maj (2.9%)
  - key switches: 11/33 (0.3333); jumps: mean=0.67, max=3.00, large=5 (0.1515); tie_count: mean=1.06, max=2
  - minor-mode predictions: 14 (0.4118 of defined windows)
  - `approx_first_half` (expected C Major): n_defined=18, strict=0.8333, collection_equiv=1.0000, tied_for_max=15 (selected=15, lost_to_tiebreak=0), minor_mode_fraction=0.1667
  - `approx_second_half` (expected G Major): n_defined=16, strict=0.0625, collection_equiv=0.0625, tied_for_max=1 (selected=1, lost_to_tiebreak=0), minor_mode_fraction=0.6875
  - active mask matches frozen control exactly: True
  - Clementi run sequence: C maj -> A min -> C maj -> A min -> D maj -> G maj -> A min -> D min -> A min -> C maj -> A min -> D min (exact C->G->C: False)

### L6 — Twinkle 12.mid (Mozart 12 Variations)

(Control reproduction sanity check -- variant A's fresh key_switch/jump_distance recomputation exactly matches Phase 3G-A's own frozen arrays: True)

**A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)**
  - piece-level: 1374 predictions, 1271 active (0.9250), 4 unique keys; dominant: C maj (90.9%), D# maj (7.1%), G maj (1.9%), A# maj (0.2%)
  - key switches: 11/1270 (0.0087); jumps: mean=0.02, max=3.00, large=5 (0.0039); tie_count: mean=6.71, max=14
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `pre_384s` (expected C Major): n_defined=665, strict=1.0000, collection_equiv=1.0000, tied_for_max=665 (selected=665, lost_to_tiebreak=0), minor_mode_fraction=0.0000
  - `384_to_432s` (expected Eb Major): n_defined=96, strict=0.8958, collection_equiv=0.8958, tied_for_max=93 (selected=86, lost_to_tiebreak=7), minor_mode_fraction=0.0000
  - `post_432s` (expected C Major): n_defined=510, strict=0.9412, collection_equiv=0.9412, tied_for_max=480 (selected=480, lost_to_tiebreak=0), minor_mode_fraction=0.0000

**C: tie-aware continuity rule (same unweighted evidence as A, continuity-preferring tie-break)**
  - piece-level: 1374 predictions, 1271 active (0.9250), 4 unique keys; dominant: C maj (90.3%), D# maj (7.6%), G maj (1.9%), A# maj (0.2%)
  - key switches: 7/1270 (0.0055); jumps: mean=0.01, max=3.00, large=1 (0.0008); tie_count: mean=6.71, max=14
  - minor-mode predictions: 0 (0.0000 of defined windows)
  - `pre_384s` (expected C Major): n_defined=665, strict=1.0000, collection_equiv=1.0000, tied_for_max=665 (selected=665, lost_to_tiebreak=0), minor_mode_fraction=0.0000
  - `384_to_432s` (expected Eb Major): n_defined=96, strict=0.9583, collection_equiv=0.9583, tied_for_max=93 (selected=92, lost_to_tiebreak=1), minor_mode_fraction=0.0000
  - `post_432s` (expected C Major): n_defined=510, strict=0.9392, collection_equiv=0.9392, tied_for_max=480 (selected=479, lost_to_tiebreak=1), minor_mode_fraction=0.0000

**D: weighted key-profile matcher (functional-role-weighted templates, plain argmax)**
  - piece-level: 1374 predictions, 1271 active (0.9250), 13 unique keys; dominant: C maj (56.2%), G maj (12.4%), D min (11.0%), G min (6.8%), F maj (6.1%)
  - key switches: 133/1270 (0.1047); jumps: mean=0.21, max=5.00, large=48 (0.0378); tie_count: mean=1.23, max=2
  - minor-mode predictions: 271 (0.2132 of defined windows)
  - `pre_384s` (expected C Major): n_defined=665, strict=0.6511, collection_equiv=0.6632, tied_for_max=433 (selected=433, lost_to_tiebreak=0), minor_mode_fraction=0.0481
  - `384_to_432s` (expected Eb Major): n_defined=96, strict=0.0417, collection_equiv=0.1875, tied_for_max=4 (selected=4, lost_to_tiebreak=0), minor_mode_fraction=0.8229
  - `post_432s` (expected C Major): n_defined=510, strict=0.5451, collection_equiv=0.5569, tied_for_max=278 (selected=278, lost_to_tiebreak=0), minor_mode_fraction=0.3137
  - active mask matches frozen control exactly: True

## Cross-variant findings

### L3/L4 minor-key recovery by variant

- **L3_full_excerpt_A_minor**: A_control: strict=0.0000, collection=0.7170; C_tie_aware_continuity: strict=0.0000, collection=0.7170; D_weighted_profile: strict=0.4151, collection=0.4623
- **L4_full_piece_E_minor**: A_control: strict=0.0000, collection=0.4907; C_tie_aware_continuity: strict=0.0000, collection=0.5324; D_weighted_profile: strict=0.4074, collection=0.4074

### L1/L6 stability by variant

- **L1 full_piece (C major)**: A_control: strict=1.0000, switches=0, large_jumps=0; C_tie_aware_continuity: strict=1.0000, switches=0, large_jumps=0; D_weighted_profile: strict=0.3585, switches=45, large_jumps=8
- **L6_pre_384s**: A_control: strict=1.0000; C_tie_aware_continuity: strict=1.0000; D_weighted_profile: strict=0.6511
- **L6_384_to_432s**: A_control: strict=0.8958; C_tie_aware_continuity: strict=0.9583; D_weighted_profile: strict=0.0417
- **L6_post_432s**: A_control: strict=0.9412; C_tie_aware_continuity: strict=0.9392; D_weighted_profile: strict=0.5451
- **L6 overall**: A_control: switches=11, large_jumps=5; C_tie_aware_continuity: switches=7, large_jumps=1; D_weighted_profile: switches=133, large_jumps=48

### Bach (L2) tonic-neighborhood behavior by variant

- A_control: 95/95 mismatches within C/G/D (1.0000); breakdown: {'C maj': 49, 'D maj': 46}
- C_tie_aware_continuity: 40/40 mismatches within C/G/D (1.0000); breakdown: {'C maj': 15, 'D maj': 25}
- D_weighted_profile: 51/65 mismatches within C/G/D (0.7846); breakdown: {'C maj': 9, 'D maj': 42, 'B min': 3, 'A min': 6, 'F# min': 2, 'E min': 2, 'A maj': 1}

### Clementi (L5) run behavior by variant

- A_control: C maj -> G maj -> C maj (exact C->G->C: True)
- C_tie_aware_continuity: C maj -> G maj (exact C->G->C: False)
- D_weighted_profile: C maj -> A min -> C maj -> A min -> D maj -> G maj -> A min -> D min -> A min -> C maj -> A min -> D min (exact C->G->C: False)

### Chopin (L4) silence-window handling by variant

Per the task's instruction, silence behavior is reported here only to confirm whether it changed mechanically -- not reinterpreted (Phase 3G-B's boundary-granularity + smoothing-memory mechanism explanation stands unless a variant's numbers actually differ from the frozen control's).
- A_control: 0/9 inactive
- C_tie_aware_continuity: 0/9 inactive
- D_weighted_profile: 0/9 inactive

## Verdict

**L3 (Für Elise, A minor) strict expected-key proportion: A=0.0000 -> D=0.4151 (IMPROVED). L4 (Chopin, E minor) strict expected-key proportion: A=0.0000 -> D=0.4074 (IMPROVED). L1 (Twinkle, C major) strict expected-key proportion: A=1.0000 -> D=0.3585, key switches: A=0 -> D=45 (DEGRADED). L6 (Twinkle 12) per-anchor strict proportion deltas (D - A): pre_384s: -0.3489, 384_to_432s: -0.8542, post_432s: -0.3961 (DEGRADED). OVERALL: the weighted key-profile variant (D) does NOT achieve both improved strict minor-key recovery and preserved L1/L6 stability simultaneously -- see the deltas above for exactly where it falls short.**

- minor_recovery_improved_L3 = True
- minor_recovery_improved_L4 = True
- L1_stability_preserved = False
- L6_stability_preserved = False
- overall_success = False

## Mechanism: why D helps L3/L4 but hurts L1/L6

L1 (Twinkle) is monophonic -- at any moment its EMA-smoothed, 10%-thresholded chroma is dominated by whichever one or two notes were most recently played, not a stable multi-note harmony. Under the **unweighted** control template, a single active pitch class ties across every one of the (typically 5-7) keys that contain it as ANY diatonic degree, and `np.argmax`'s low-index tie-break happens to land on C major disproportionately often for this piece -- Phase 3B already noted this behavior comes from broad, largely accidental ties, not real tonic disambiguation. Under the **weighted** profile, that same single pitch class instead scores highest for whichever key treats it as the HIGHEST-weighted degree (tonic=5), which is a different key depending on which note was just played -- so the weighted resolver tracks the melody's passing notes' own local tonic-implication rather than the piece's actual, stable tonic. L1's dominant predicted keys under D are C maj (35.8%), D min (17.9%), G maj (13.2%), E min (11.3%), F maj (10.4%) -- scattered across several tonics rather than concentrated on C major, confirming this mechanism directly. The same effect degrades L6's largely-monophonic melody-plus-light-accompaniment texture. By contrast, L3/L4 have enough real, simultaneously-sounding harmonic content (or enough EMA-accumulated note history) that the mediant-weighted tonic/dominant/mediant evidence more often correctly favors the true (minor) tonic over its relative major -- exactly the ambiguity D was designed to break. **Net reading: D is not a strict improvement over the control -- it trades away monophonic-melody stability for minor-key tonic/mode resolution, and does not dominate the control across the whole corpus.**

## Plots

- `05_Figures_Results/PHASE3H_A_FurElise_excerpt_variant_comparison_key_trajectory.png`
- `05_Figures_Results/PHASE3H_A_Chopin_Op28No4_variant_comparison_key_trajectory.png`
(Only the two minor-key pieces central to this phase's question are plotted, per the task's instruction to avoid excessive plotting.)

## Scope note

This is Phase 3H-A only: non-neural decision-rule and representation ablations on top of the frozen Phase 3G-A pitch-class baseline. No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement was run or implemented. `np.argmax`'s own tie-breaking behavior was not changed anywhere -- variant C adds a continuity preference that only applies when the previous key is itself among the tied keys (falling back to the same argmax rule otherwise), and variant D applies the unchanged argmax rule to a different, hand-specified (not trained, not tuned on this corpus) template. No dense per-timestep accuracy is claimed anywhere -- all anchor comparisons are the same window-level, documented-expected-key convention Phase 3G-A/3G-B already used. Anchors were used exclusively inside the evaluation functions (`compute_anchor_metrics`, `bach_tonic_neighborhood`), never inside any variant's key_id computation (`variant_A_control`, `variant_C_tie_aware_continuity`, `variant_D_weighted_profile` take no anchor or expected-key argument at all).
