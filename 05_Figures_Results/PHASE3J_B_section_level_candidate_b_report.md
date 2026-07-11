# Phase 3J-B — Section-Level Candidate B: Implementation and Evaluation

Implements and evaluates ONLY the revised Phase 3J-A Candidate B (mediant + asymmetric raised-leading-tone comparison, aggregated per stable diatonic-collection segment). Candidates A and C are explicitly NOT implemented. No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement was run. The frozen Stage 1 baseline and all Phase 3G/3H/3I/3J-A outputs are read-only references, never modified. No dense per-timestep accuracy is claimed anywhere below.

**Review process**: after the primary run completed, three independent, read-only subagent passes (implementation audit, methodology/leakage audit, music-theory critique) reviewed the actual script and actual results. None modified any file, ran any code, or altered any prediction, segment, or constant. All three independently confirmed a structural (algebraic, not coding-bug) property of the pre-registered mediant cue -- see "Mechanistic finding" below -- which is reported here as additive analysis on top of the unmodified primary-configuration results, not as a change to them.

## Pre-registered configuration (frozen before any anchor evaluation)

- `MIN_SEGMENT_WINDOWS`: 4
- `MIN_SEGMENT_DURATION_SEC`: 2.0
- `BRIEF_INTERRUPTION_MAX_WINDOWS`: 3
- `MERGE_RULE`: merge a brief interruption only when the stable collection on both sides is identical; a brief run between different collections becomes its own unresolved 'transition' segment
- `SEGMENT_DETECTION_MODE`: offline, whole-sequence, label-free run-length encoding of the Stage 2 collection sequence (Stage 1's key_id collapsed via collection_class) -- no anchor, no lookahead-into-the-future beyond the piece's own already-fully-computed Stage 1 output
- `AGGREGATION_SOURCE`: sum of raw (pre-threshold) chroma across all windows in a stable segment; verified numerically equivalent (see verify_chroma_score_linearity) to scoring each window from raw chroma and summing the per-window scores, by linearity of the fixed SCALE_TEMPLATES matmul -- thresholded chroma is never aggregated. NOTE: this equivalence is checked against raw-chroma-derived per-window scores computed in this script, NOT against the frozen derived_phase3g_corpus raw_scores.npy array, because that array is actually thresholded_chroma @ SCALE_TEMPLATES.T (Phase 3B's analyze_piece naming: 'raw' = un-normalized, not un-thresholded) -- see verify_chroma_score_linearity's docstring for the full explanation.
- `AMBIGUOUS_FALLBACK`: segment majority vote over the segment's own frozen Stage 1 key_id values (never index-order argmax, never an anchor)
- `MEDIANT_MARGIN_THRESHOLD`: 0.02
- `LEADING_TONE_PRESENCE_THRESHOLD`: 0.02
- `LEADING_TONE_ASYMMETRY`: presence of the raised leading tone is POSITIVE evidence for the minor candidate and, when present, settles the decision toward minor regardless of the mediant cue; ABSENCE of the raised leading tone is NEUTRAL -- it never itself counts as evidence for major, and the decision in that case rests entirely on the (symmetric) mediant cue
- `STAGE6_TRANSITION_UNDEFINED_CONVENTION`: transition and undefined segments are NOT resolved by Candidate B; their windows in key_id_section_resolved preserve Stage 1's own frozen key_id verbatim, and are additionally flagged False in a parallel is_section_resolved boolean mask so 'genuinely resolved by Candidate B' vs. 'fell back to frozen Stage 1' is always distinguishable downstream

Sensitivity grid (predeclared, descriptive only): `MIN_SEGMENT_WINDOWS` in [4, 6, 8], `BRIEF_INTERRUPTION_MAX_WINDOWS` in [1, 3].

## Per-piece results (primary configuration)

### L1 — Twinkle.mid

- **L1**: 1 stable / 0 transition / 0 undefined segments (of 1 total); 1.0000 of windows section-resolved; segment duration mean=53.00s, median=53.00s
- Stage 1 dominant: C maj (100.0%) | Section-resolved dominant: C maj (100.0%)
- Key switches: Stage1=0 -> Resolved=0; jumps: mean 0.00->0.00, max 0.00->0.00; large jumps: 0->0 (0.0000->0.0000)
- Anchor `full_piece` (expected C Major): strict Stage1=1.0000 -> Resolved=1.0000; collection-equiv Stage1=1.0000 -> Resolved=1.0000
- Chroma/raw_scores linearity check: 1 stable segments checked, all_passed=True, max_abs_error=0.00e+00

### L2 — Bach — Minuet in G Major, BWV Anh. 114

- **L2**: 13 stable / 0 transition / 0 undefined segments (of 13 total); 1.0000 of windows section-resolved; segment duration mean=6.31s, median=6.00s
- Stage 1 dominant: G maj (42.1%), C maj (29.9%), D maj (28.0%) | Section-resolved dominant: E min (28.0%), B min (28.0%), A min (26.8%)
- Key switches: Stage1=12 -> Resolved=12; jumps: mean 0.08->0.13, max 2.00->5.00; large jumps: 0->1 (0.0000->0.0061)
- Anchor `full_piece` (expected G Major): strict Stage1=0.4207 -> Resolved=0.1402; collection-equiv Stage1=0.4207 -> Resolved=0.4207
- **Minor-mode windows by decision source** (see Mechanistic finding below): 136 total minor-resolved windows -- leading_tone=0.0000, mediant_signal=1.0000, fallback=0.0000
- Chroma/raw_scores linearity check: 13 stable segments checked, all_passed=True, max_abs_error=0.00e+00

### L3 — Beethoven — Für Elise (opening excerpt, [0.0, 54.0]s)

- **L3**: 5 stable / 2 transition / 1 undefined segments (of 8 total); 0.9533 of windows section-resolved; segment duration mean=10.20s, median=5.00s
- Stage 1 dominant: C maj (71.7%), E maj (19.8%), A maj (4.7%) | Section-resolved dominant: C maj (50.0%), A min (29.2%), C# min (9.4%)
- Key switches: Stage1=16 -> Resolved=6; jumps: mean 0.59->0.28, max 6.00->6.00; large jumps: 15->5 (0.1429->0.0476)
- Anchor `full_excerpt` (expected A Minor): strict Stage1=0.0000 -> Resolved=0.2925; collection-equiv Stage1=0.7170 -> Resolved=0.7925
- **Minor-mode windows by decision source** (see Mechanistic finding below): 45 total minor-resolved windows -- leading_tone=0.6889, mediant_signal=0.3111, fallback=0.0000
- Chroma/raw_scores linearity check: 5 stable segments checked, all_passed=True, max_abs_error=0.00e+00

### L4 — Chopin — Prelude in E minor, Op. 28 No. 4

- **L4**: 20 stable / 5 transition / 0 undefined segments (of 25 total); 0.9491 of windows section-resolved; segment duration mean=5.12s, median=3.50s
- Stage 1 dominant: G maj (49.1%), C maj (24.1%), E maj (12.5%) | Section-resolved dominant: E min (50.0%), A min (15.7%), C maj (12.0%)
- Key switches: Stage1=38 -> Resolved=24; jumps: mean 0.44->0.27, max 6.00->5.00; large jumps: 20->12 (0.0930->0.0558)
- Anchor `full_piece` (expected E Minor): strict Stage1=0.0000 -> Resolved=0.5000; collection-equiv Stage1=0.4907 -> Resolved=0.5093
- **Minor-mode windows by decision source** (see Mechanistic finding below): 168 total minor-resolved windows -- leading_tone=0.8393, mediant_signal=0.1607, fallback=0.0000
- Chroma/raw_scores linearity check: 20 stable segments checked, all_passed=True, max_abs_error=0.00e+00
- Silence-region check (interpretation preserved, not redefined): 0/9 inactive

### L5 — Clementi — Sonatina Op. 36 No. 1, I (exposition, [0.0, 17.3]s)

- **L5**: 3 stable / 0 transition / 0 undefined segments (of 3 total); 1.0000 of windows section-resolved; segment duration mean=5.67s, median=5.50s
- Stage 1 dominant: C maj (67.6%), G maj (32.4%) | Section-resolved dominant: A min (38.2%), E min (32.4%), C maj (29.4%)
- Key switches: Stage1=2 -> Resolved=2; jumps: mean 0.06->0.15, max 1.00->4.00; large jumps: 0->1 (0.0000->0.0303)
- Anchor `approx_first_half` (expected C Major): strict Stage1=0.7222 -> Resolved=0.0000; collection-equiv Stage1=0.7222 -> Resolved=0.7222
- Anchor `approx_second_half` (expected G Major): strict Stage1=0.3750 -> Resolved=0.0000; collection-equiv Stage1=0.3750 -> Resolved=0.3750
- **Minor-mode windows by decision source** (see Mechanistic finding below): 24 total minor-resolved windows -- leading_tone=0.0000, mediant_signal=1.0000, fallback=0.0000
- Chroma/raw_scores linearity check: 3 stable segments checked, all_passed=True, max_abs_error=0.00e+00

### L6 — Twinkle 12.mid (Mozart 12 Variations)

- **L6**: 7 stable / 1 transition / 1 undefined segments (of 9 total); 0.9236 of windows section-resolved; segment duration mean=90.64s, median=48.00s
- Stage 1 dominant: C maj (90.9%), D# maj (7.1%), G maj (1.9%) | Section-resolved dominant: A min (67.7%), C maj (22.7%), D# maj (7.6%)
- Key switches: Stage1=11 -> Resolved=7; jumps: mean 0.02->0.02, max 3.00->6.00; large jumps: 5->4 (0.0039->0.0031)
- Anchor `pre_384s` (expected C Major): strict Stage1=1.0000 -> Resolved=0.0000; collection-equiv Stage1=1.0000 -> Resolved=1.0000
- Anchor `384_to_432s` (expected Eb Major): strict Stage1=0.8958 -> Resolved=0.9583; collection-equiv Stage1=0.8958 -> Resolved=0.9583
- Anchor `post_432s` (expected C Major): strict Stage1=0.9412 -> Resolved=0.5647; collection-equiv Stage1=0.9412 -> Resolved=0.9412
- **Minor-mode windows by decision source** (see Mechanistic finding below): 885 total minor-resolved windows -- leading_tone=0.0000, mediant_signal=1.0000, fallback=0.0000
- Chroma/raw_scores linearity check: 7 stable segments checked, all_passed=True, max_abs_error=0.00e+00

## Sensitivity audit (descriptive only -- no winner chosen)

| condition | L1 stable segs | L1 switches | L1 minor frac | L3 minor frac | L4 minor frac | L2 stable segs | L4 stable segs | L5 stable segs | L6 stable segs |
|---|---|---|---|---|---|---|---|---|---|
| min=4, interrupt<=1 | 1 | 0 | 0.000 | 0.726 | 0.699 | 13 | 20 | 3 | 9 |
| min=4, interrupt<=3 | 1 | 0 | 0.000 | 0.425 | 0.778 | 13 | 20 | 3 | 7 |
| min=6, interrupt<=1 | 1 | 0 | 0.000 | 0.689 | 0.616 | 12 | 13 | 3 | 9 |
| min=6, interrupt<=3 | 1 | 0 | 0.000 | 0.387 | 0.713 | 12 | 15 | 3 | 7 |
| min=8, interrupt<=1 | 1 | 0 | 0.000 | 0.689 | 0.491 | 8 | 7 | 3 | 9 |
| min=8, interrupt<=3 | 1 | 0 | 0.000 | 0.387 | 0.620 | 8 | 9 | 3 | 7 |

Across the 6 predeclared sensitivity conditions: L1 (Twinkle) key switches range 0-0, L1 minor-mode fraction ranges 0.000-0.000, L3 (Für Elise) minor-mode fraction ranges 0.387-0.726, L4 (Chopin) minor-mode fraction ranges 0.491-0.778, L2 (Bach) stable-segment count ranges 8-13, L4 stable-segment count ranges 7-20. No condition in this grid is selected, adopted, or treated as a winner -- these ranges are reported purely to characterize whether the primary configuration's qualitative conclusions (below) are robust or fragile across nearby, equally-defensible settings.

## Mechanistic finding: the mediant cue is structurally confounded with tonic-pitch prominence

Discovered during independent post-run review (three read-only subagent passes: implementation audit, methodology/leakage audit, music-theory critique) and confirmed algebraically, not a coding bug -- `resolve_segment_candidate_b` faithfully implements exactly the rule `PHASE3J_A_section_level_resolver_design.md` specifies. **This is a design-level property of the pre-registered Candidate B rule itself, reported here as a finding, not corrected by adjusting any constant** (per this task's guardrail against post-hoc parameter changes -- this is additive analysis of the existing, unmodified predictions, segments, and constants).

For any collection, `minor_tonic_pc = major_tonic_pc - 3` and the minor candidate's third scale degree is `minor_tonic_pc + 3` -- these cancel exactly: `minor_third_pc == major_tonic_pc`, algebraically, for all 12 relative pairs. So `minor_third_evidence_fraction` is not independent mode evidence at all -- it is literally the aggregate evidence at the MAJOR candidate's own tonic pitch class. Since essentially all tonal music (major or minor) emphasizes its own tonic pitch heavily, any segment that simply plays its tonic a lot will show high `minor_third_evidence`, mechanically dragging `mediant_signal` negative and the decision toward "minor" -- independent of whether the passage is actually minor. This fully explains L6's largest stable segment (669 windows, resolved A minor via `mediant_signal`, with `leading_tone_evidence_fraction=0.0` -- zero genuine harmonic-minor evidence) and is the dominant source of L6's anchor-proportion collapse.

By contrast, the raised-leading-tone cue (`leading_tone_positive_evidence`) is NOT subject to this confound: it queries a pitch class foreign to both collection members under the plain diatonic representation, so real evidence there reflects an actual chromatic (harmonic/melodic-minor-style) gesture, not tonic-pitch bookkeeping. The decision-source breakdowns in the per-piece sections above show this split concretely: Für Elise's core A-minor block and the majority of Chopin's E-minor windows are `leading_tone_positive_evidence`-driven (comparatively trustworthy), while L6's minor-resolved windows are essentially entirely `mediant_signal`-driven (the confounded path) with no leading-tone support anywhere. **Readers should treat this report's pooled "minor recovery" percentages as an upper bound of uncertain composition, and consult each piece's decision-source breakdown before concluding the rule detects genuine harmonic minor** -- the leading-tone-attributed fraction is the more defensible number.

## Interpretation checks

**L1 Twinkle**: remains one single stable C-major segment. Strict C-major proportion: Stage1=1.0000 -> Resolved=1.0000. **L2 Bach**: segmentation produced 13 stable segment(s) (0 transition). Per Phase 3G-B's finding that Bach's C/G/D spread is systematic tie-break bias, not real tonicization, this segment count is reported with that caveat -- it is not treated as evidence of real modulation. Strict G-major proportion: Stage1=0.4207 -> Resolved=0.1402. **L3 Für Elise**: strict A-minor proportion: Stage1=0.0000 -> Resolved=0.2925. 5 stable segment(s) formed. **L4 Chopin**: 20 stable segment(s) formed (of 25 total; 0.0509 of windows in unresolved transition segments) -- Phase 3J-A flagged Chopin's low (0.491) collection-equivalent proportion as a specific fragmentation risk. Strict E-minor proportion: Stage1=0.0000 -> Resolved=0.5000. Silence-region interpretation preserved unchanged (see per-piece section). **L5 Clementi**: 3 stable segment(s) formed. Compared descriptively against the previously-observed, musically plausible C->G->C baseline trajectory (Phase 3G-A/3G-B) -- this is NOT independently-verified dense ground truth, and no constant was adjusted based on this comparison. **L6 Twinkle 12**: 7 stable segment(s) formed (of 9 total). Anchor-window strict proportions: pre_384s: Stage1=1.0000 -> Resolved=0.0000; 384_to_432s: Stage1=0.8958 -> Resolved=0.9583; post_432s: Stage1=0.9412 -> Resolved=0.5647.

## Verdict

**Verdict code: 3**

**Section-level Candidate B recovers minor mode but introduces unacceptable segmentation/stability damage.** L3 strict A-minor: 0.0000 -> 0.2925. L4 strict E-minor: 0.0000 -> 0.5000. Stability failure: L6 anchor deltas exceed the -0.05 tolerance (['-1.0000', '+0.0625', '-0.3765']). Per the Mechanistic finding section above, this damage is traced almost entirely to the `mediant_signal` decision path (structurally confounded with tonic-pitch prominence), while the `leading_tone_positive_evidence` path -- responsible for most of Chopin's and Für Elise's core minor recovery -- does not exhibit the same failure mode.

## Scope note

This is Phase 3J-B only: implementation and evaluation of the revised Candidate B section-level resolver. Candidates A and C were not implemented. No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement was run. All governing constants (`PRE_REGISTERED_CONFIG` above) were fixed in this script's source before any anchor was loaded or compared, and were not adjusted after seeing results. Anchors were used exclusively for evaluation, in the Evaluation section, after every piece's segments and `key_id_section_resolved` array already existed -- never inside segmentation, aggregation, or the Candidate B decision rule. No dense per-timestep accuracy is claimed anywhere in this report.
