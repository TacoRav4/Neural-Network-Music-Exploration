# Phase 3I — Synthesis and Next-Architecture Decision Memo

**Type: writing/synthesis only.** No new modeling code was written for this
memo, no script was run, and no chord-id EMA/SRN, Chroma SRN, Transformer,
or neural refinement was implemented. This document synthesizes the
already-completed, already-verified results of Phase 3G-A, 3G-B, 3H-A,
3H-B, and 3H-C (all frozen, none re-run or reinterpreted here beyond
restating their own stated conclusions) and proposes — but does not
implement — the next phase. All anchors referenced below are the same
sparse, documented expected-key windows used throughout Phase 3G/3H; they
are evaluation references only, never used to choose a prediction, and
nothing here claims dense per-timestep accuracy (real MIDI has no such
ground truth anywhere in this workspace).

Full evidence trail: `PHASE3G_A_pitch_class_corpus_report.md`,
`PHASE3G_B_tie_aware_diagnostics_report.md`,
`PHASE3H_A_tonic_mode_resolver_report.md`,
`PHASE3H_B_texture_gated_resolver_report.md`,
`PHASE3H_C_gate_sensitivity_report.md`, and their corresponding
`*_metrics.json` files (all in this directory) and `STATUS.md`.

---

## 1. What Phase 3G-A/3G-B established

**The frozen pitch-class/scale-template baseline (`pitch_class_baseline.py`,
`SCALE_TEMPLATES`, plain `np.argmax`) is a reliable diatonic-*collection*
fast filter, not a full tonic/mode resolver.** Specifically:

- **It recovers large-scale structure well on the corpus's two reference
  pieces.** Twinkle.mid (L1): 100% C major, 0 key switches, 0 large jumps.
  Twinkle 12.mid (L6): its real, embedded C→Eb→C modulation is tracked
  closely (pre_384s 100% C major, 384–432s 89.6% Eb/D# major, post_432s
  94.1% C major) — both results reproduced exactly by the corpus-aware
  Phase 3G-A pipeline, confirming they generalize beyond the original
  single-file scripts.

- **It fails strict minor-key labels for a structural, deterministic
  reason, not noisy ambiguity.** `SCALE_TEMPLATES` places relative
  major/minor pairs (e.g. A minor / C major) at numerically *identical*
  rows, and `key_index` places all 12 major keys at indices 0–11, all 12
  minor keys at 12–23. `np.argmax`'s leftmost-tie convention therefore
  *always* resolves an exact tie toward the major-indexed key. Verified
  directly on the saved arrays: on Für Elise (L3, A minor), the expected
  key tied for the top score in 76/106 active windows and was selected in
  **zero** of them; on Chopin No. 4 (L4, E minor), it tied in 135/216 and
  was selected in **zero**. Both anchors report `strict_expected_key_proportion
  = 0.0` — but `collection_equivalent_proportion` (predicted key is the
  expected key *or* its relative major/minor) is 0.717 for L3 and 0.491
  for L4, showing the baseline usually finds the right 7-pitch-class
  collection and only mislabels the tonic/mode.

- **Bach's (L2, G major) errors are tonic-neighborhood ambiguity, not
  distant-key failure.** Of 95 mismatched windows, 100% predicted a key
  within the C/G/D neighborhood (C major 49, D major 46) — zero landed on
  an unrelated or distant key. The mechanism is the *same* tie-break
  bias as the minor-key pieces: 142/164 windows have G major tied for the
  max score, but only 69 actually select it, because a lower-index major
  key (C or D) is often tied too.

- **Clementi (L5, nominally "C major → G major") is better described as a
  tonic–dominant–tonic excursion than a monotonic modulation.** The
  actual predicted-key run sequence is C major (0–6.0s) → G major
  (6.5–11.5s) → C major (12.0–16.5s), not a one-way shift — a real but
  qualitatively different behavior than the corpus ladder's original
  framing assumed.

- **Chopin's documented silence (t≈95.36–99.64s) registers as 0/9
  inactive windows, and this is a boundary-granularity + smoothing-memory
  convention effect, not a bug.** Only 5 of those 9 windows are actually
  raw-silent (t=97.00–99.00s); the other 4 genuinely contain sound (a
  sustain/pedal tail into the pause, and the next phrase's onset arriving
  at/before the documented end boundary). For the 5 truly-silent windows,
  the smoothed chroma decays geometrically (verified against the
  `smoothed_t = 0.8·smoothed_{t-1}` formula with 0.00e+00 error) but never
  reaches zero, and the 10%-of-*that-window's-own*-max threshold means the
  thresholded pattern's nonzero support never disappears under pure decay
  — so `active` stays true regardless of the real silence underneath.

**Net reading of Phase 3G-A/3G-B**: the representation (full-scale-template
evidence) and the temporal-smoothing convention are both doing real,
explicable work, and the baseline's apparent minor-key "failures" are a
specific, fully-characterized artifact of one design choice
(`np.argmax`'s tie-break combined with the major-first index ordering) —
not a diffuse, unexplained weakness.

## 2. What Phase 3H-A/3H-B/3H-C established

Phase 3H asked whether that one specific artifact could be fixed with a
small, non-neural, interpretable change, and — critically — whether any
fix could be applied *safely*, without damaging the baseline's
demonstrated strength on monophonic/large-scale-structure pieces.

- **3H-A (ablations): weighted profiles can restore tonic/mode
  sensitivity, but at a real, measured cost to sparse-melody stability.**
  A hand-specified (not trained, not tuned to this corpus) weighted
  template — tonic=5, dominant=4, mediant=3, other diatonic degrees=1,
  applied to each scale's own degree offsets — breaks the relative
  major/minor tie structurally (verified: 0/12 relative pairs still
  identical, 24/24 rows pairwise distinct) and lifts L3's strict minor
  recovery from 0.0 to 0.415 and L4's from 0.0 to 0.407. But it collapses
  L1 (Twinkle) from a perfect 1.0/0-switches sanity check to 0.359/45
  switches, and degrades all three of L6's anchors (deltas of −0.35,
  −0.85, −0.40). The mechanism is understood, not just observed:
  monophonic melodies' EMA-smoothed evidence is often dominated by one or
  two recently-played notes, and under weighting, a single note now
  scores highest for whichever key treats it as *tonic* — so the resolver
  chases each note's own local tonic-implication instead of the piece's
  one stable tonic. A separate, non-weighted variant (tie-aware
  continuity) reduces key-switch counts broadly (e.g. Bach 12→5, Chopin
  38→32) with *zero* effect on minor-key recovery, since it only
  re-selects among already-tied keys.

- **3H-B (conservative texture gating): protects stability but recovers
  almost none of the benefit.** A gated resolver defaulting to the
  continuity variant and swapping in the weighted prediction only when
  density, collection-stability, and weighted-margin gates all pass
  (using an existing, non-tuned Phase 3B threshold for the margin gate)
  successfully protects L1 (1.0→0.9906, 0→2 switches — a small fraction
  of D's damage) and L6 (anchors preserved within tolerance). But the
  margin gate passes on well under 6% of windows corpus-wide, so almost
  no weighted swaps occur at all: L3 gets **zero** swaps (strict recovery
  stays 0.0) and L4 gets only 10/216 (strict recovery nudges to a token
  0.0185).

- **3H-C (gate sensitivity / Pareto audit): no free-lunch region exists
  within this gate family.** A predeclared 54-condition sweep over the
  same three gates' thresholds (margin: 0.00–0.20; density: >1/>2/>3;
  stability window: 2/4/6) produced a clean, monotonic Pareto frontier —
  16 non-dominated conditions running smoothly from
  (damage≈0.004, recovery≈0.014) up to (damage≈0.386, recovery≈0.339).
  Using two purely descriptive, predeclared-before-scanning bars
  (stability damage ≤0.02, minor recovery ≥0.10), **zero of the 54
  conditions met both simultaneously.** Every low-damage condition has
  low recovery; every meaningfully-recovering condition carries
  non-trivial damage. Phase 3H-B's own configuration was itself
  Pareto-dominated by other grid points — so this isn't a case of having
  picked a suboptimal threshold; the whole achievable frontier for this
  gate family lacks a region that satisfies both goals.

**Net reading of Phase 3H**: the weighted-profile idea is directionally
correct (it does contain real minor-mode-recovering evidence) but the
*timestep-level* gating mechanism — deciding per 0.5-second window whether
to trust the weighted evidence — cannot separate "this window is part of
a stable minor passage" from "this window is one ambiguous note in a
monophonic melody" well enough to help one without hurting the other.

## 3. Architectural conclusion

1. **Do not replace the frozen pitch-class baseline with the weighted
   profile.** Every non-neural attempt to make weighting the default
   (3H-A's variant D) damaged the corpus's two large-scale-structure
   reference pieces (L1, L6) more than it helped the two minor-key pieces
   (L3, L4) — and even the fully gated version (3H-B) that best contained
   this damage recovered almost none of the benefit.

2. **Preserve the frozen pitch-class baseline as the Stage 1 fast
   filter, unchanged.** Nothing in Phase 3G/3H found a superior
   *drop-in* replacement for `pitch_class_baseline.py`/`SCALE_TEMPLATES`/
   plain `np.argmax` at the timestep level. Its known behavior — reliable
   collection-level tracking, structurally-explained tonic/mode
   ambiguity on ties — remains the best-characterized non-neural option
   at this timescale.

3. **Do not continue timestep-level weighted-profile gate tuning.**
   Phase 3H-C's 54-condition sweep is not a partial result inviting a
   55th configuration — it is a characterization of the entire gate
   family's achievable trade-off space, and that space has no free
   region. Further sweeps within the same three gates (density,
   collection-stability, weighted-margin, applied per 0.5s window) would
   very likely land on or near the same frontier, not off of it.

4. **The next plausible resolver should operate at a longer timescale:
   section-level or phrase-level tonic/mode inference, not per-window.**
   The core problem 3H-A/3H-B/3H-C converged on is that a single 0.5s
   window's evidence is fundamentally insufficient to distinguish
   "ambiguous but locally correct minor-mode evidence" from "a passing
   note in an otherwise-stable major context" — because both look like
   sparse, tied evidence in isolation. Phase 3B already found (independent
   of Phase 3H) that the baseline's real accuracy on Twinkle/Twinkle 12
   comes from EMA evidence *accumulating over many windows*, not from any
   single window being individually decisive. A resolver that first
   segments the piece into stretches where the diatonic *collection* is
   already stable (which the frozen Stage 1 filter already computes
   reliably, per Phase 3G-A/3G-B), and only then asks the tonic/mode
   question once per segment using that segment's *aggregated* evidence,
   directly targets the actual bottleneck this whole phase sequence
   identified — rather than re-fighting it one window at a time.

## 4. Proposed Phase 3J options (not implemented here)

### Option A — Section-level non-neural tonic/mode resolver

Aggregate chroma (or `raw_scores`) over already-identified stable
diatonic-collection segments (segment boundaries determined by the frozen
Stage 1 filter's own collection-stability behavior, the same notion Phase
3H-B's gate 2 used at the window level — but here defining the *segment*,
not gating a per-window swap), and decide tonic/mode **once per segment**
using the segment's aggregated evidence rather than per 0.5s window. No
anchors used for prediction anywhere — anchors would remain evaluation-only,
exactly as throughout Phase 3G/3H. This directly targets the
insufficient-per-window-evidence problem 3H-A/B/C converged on, without
introducing any neural component.

### Option B — Chroma SRN, later

A learned recurrent model over chroma, explicitly deferred until *after*
a section-level non-neural baseline (Option A) exists and is evaluated.
Framed as a **tonic/mode refinement layer over chroma**, not a
replacement for Stage 1 — consistent with the staged architecture this
whole workspace has followed since Phase 3A (fast filter → anchors →
uncertainty diagnostics → conditional refinement → evaluation gate).
Attempting this before a non-neural section-level baseline exists would
make it impossible to tell whether any future gain came from recurrence
or from simply operating at the right timescale — the same
representation-vs-mechanism confound Phase 1.5B/Phase 2 already
resolved once for the chord-id vs. pitch-class question.

### Option C — Stop here for the current report/paper

Write up the completed Phase 3G/3H sequence as a finished finding about
representation and temporal scale: a non-neural, fully-interpretable
pitch-class fast filter reliably recovers large-scale tonal structure;
its minor-mode/tonic-neighborhood limitations are structurally explained,
not mysterious; and a systematic, predeclared search across
representation weighting and gating shows the limitation is a
timescale/evidence-density problem, not a fixable per-window decision
rule. This is a complete, defensible, and non-trivial result on its own.

## 5. Recommendation

**Prefer a design-only Phase 3J-A plan for the section-level non-neural
tonic/mode resolver (Option A). Do not code it yet.**

Rationale: Option C forecloses further investigation without exhausting
the most promising remaining direction Phase 3H's own results point to.
Option B (Chroma SRN) would reintroduce exactly the representation/
mechanism confound this workspace has been careful to avoid at every
prior transition (Phase 1→1.5, Phase 1.5→2, Phase 3D's original
decision to defer Stage 4) — jumping to a learned model before
establishing whether the *timescale* change alone (still non-neural)
already solves the problem observed in Phase 3H. Option A is the direct,
minimal next step implied by Section 3's conclusion, and only needs a
design document — not new code — before a decision to implement it.

---

## Verification

- Phase 3G-A, 3G-B, 3H-A, 3H-B, and 3H-C output files (`PHASE3G_A_*`,
  `PHASE3G_B_*`, `PHASE3H_A_*`, `PHASE3H_B_*`, `PHASE3H_C_*` under
  `05_Figures_Results/` and `03_MIDI_Data/derived_phase3g_corpus/`) were
  not opened for writing, modified, or overwritten by this task — this
  memo only reads and restates their already-published conclusions.
- No script in `04_Recurrent_Implementation/` was created, modified, or
  run by this task — this is a writing-only deliverable.
- No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement was
  run or implemented.
- This report exists at `05_Figures_Results/PHASE3I_synthesis_and_architecture_decision.md`
  and is non-empty.
- No dense per-timestep accuracy is claimed anywhere above — every
  quantitative claim is either a window-count/proportion against a
  documented sparse anchor (identical convention to Phase 3G/3H) or a
  descriptive count (switches, jumps, swap rates).
- **Timestep-level weighted-profile gating should not be pursued
  further**: Section 3, point 3 above states this explicitly, grounded in
  Phase 3H-C's finding that its entire 54-condition, predeclared
  trade-off frontier contains no region simultaneously preserving L1/L6
  stability and recovering meaningful L3/L4 minor-mode predictions.
