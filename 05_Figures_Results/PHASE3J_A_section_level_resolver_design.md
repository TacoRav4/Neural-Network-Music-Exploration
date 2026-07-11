# Phase 3J-A — Section-Level Non-Neural Tonic/Mode Resolver: Design Plan

**Type: design document only. No implementation code was written or run for
this task.** No chord-id EMA/SRN, Chroma SRN, Transformer, or neural
refinement is proposed or implemented anywhere below. Phase 3G, 3H, and 3I
outputs are treated as frozen and are not modified or overwritten. Anchors
(expected-key windows) are discussed throughout strictly as sparse,
after-the-fact evaluation references — never as a signal any prediction
rule may consult.

**Review process**: this draft was reviewed by three independent,
read-only critique passes (architecture, methodology/anchor-leakage,
music-theory) before being finalized. None of the three reviewers edited
any file, ran any code, or created any output — each returned a text
critique that is synthesized into this document. Their specific findings,
and how each was incorporated, are noted inline where relevant and
summarized in §9.

---

## 1. Problem statement

Phase 3G-A/3G-B established that the frozen Stage 1 pitch-class baseline
(`pitch_class_baseline.py`, `SCALE_TEMPLATES`, plain `np.argmax`) is a
reliable diatonic-*collection* fast filter but not a full tonic/mode
resolver: it recovers Twinkle (100% C major, 0 switches) and Twinkle 12's
real C→Eb→C modulation (100%/89.6%/94.1% across its three anchor windows)
well, but fails strict minor-key labels deterministically — Für Elise and
Chopin are both 0.0 strict, because `SCALE_TEMPLATES` gives relative
major/minor pairs identical rows and `np.argmax`'s tie-break always
prefers the major-indexed key (0.717 and 0.491 collection-equivalent,
respectively — the collection is usually right, only the tonic/mode label
is wrong). Bach's errors (100% within the C/G/D tonic neighborhood) are
the *same* tie-break mechanism showing up as neighborhood confusion, not
a distinct failure mode.

Phase 3H-A/B/C then established that this tie-break problem cannot be
fixed by changing *what evidence wins at each individual 0.5s window*. A
weighted key-profile (breaking the tie structurally) recovers minor-key
strict proportions (L3 0.0→0.415, L4 0.0→0.407) but destroys monophonic
stability (L1 1.0→0.359, 45 new switches). Gating the weighted profile in
only for dense, stable, high-margin windows (Phase 3H-B) protects
stability (L1 1.0→0.9906) but recovers almost none of the benefit (L3
stays 0.0, L4 only reaches 0.0185), because the margin gate passes on
under 6% of windows corpus-wide. A predeclared 54-condition sweep over
that entire gate family (Phase 3H-C) found a clean, monotonic
damage-vs-recovery frontier with **zero** conditions achieving both low
damage and meaningful recovery — a characterization of the whole family's
achievable space, not a tuning failure.

**Therefore**: the problem is not which per-window decision rule to use —
it is that a single 0.5-second window's evidence is too sparse to support
a tonic/mode decision at all, independent of how it is weighted or gated.
The next resolver should operate over *stable sections/phrases*,
aggregating evidence across many windows before making one tonic/mode
decision per section, rather than repeating the same underpowered
decision at every window.

## 2. Proposed architecture (six stages)

- **Stage 1 (frozen, unchanged)**: the existing pitch-class baseline
  produces, per 0.5s window: `key_id` (argmax over `SCALE_TEMPLATES`),
  `raw_scores` (24-way scale-template dot products), `active`, and the
  underlying raw/smoothed/thresholded chroma. Nothing about Stage 1
  changes, and every downstream stage reads only from its output —
  nothing writes back to it.
- **Stage 2**: collapse each window's Stage 1 `key_id` into its canonical
  diatonic *collection* (the shared id for a relative major/minor pair —
  reusing the exact relationship `evaluate_phase3g_b_tie_aware_
  diagnostics.collection_equivalent_key_id` and
  `evaluate_phase3h_b_texture_gated_resolver.collection_class` already
  encode; not a new definition).
- **Stage 3**: detect stable *segments* — contiguous window ranges where
  the Stage 2 collection stays constant for at least a predeclared
  minimum length, with an explicit rule for brief interruptions (§3).
  Label-free: uses only Stage 1/2's own output. **Explicitly reuses Phase
  3H-B's Gate 2 (collection-stability, 4-window/2.0s default) as its core
  stability criterion** — Stage 3's role is boundary-finding, not new
  mechanism; the design's actual novelty is Stage 4.
- **Stage 4**: aggregate evidence within each detected segment (§4). This
  *is* the design's core novelty relative to Phase 3H — see §4.
- **Stage 5**: apply one non-neural tonic/mode decision rule per segment
  to the aggregated evidence (candidates in §5) — one decision per
  segment, not per window.
- **Stage 6**: project the segment-level decision back onto the
  underlying windows *only* for plotting/evaluation, so existing
  trajectory-style plots and `compute_anchor_metrics`-based comparisons
  continue to work. **This projected sequence must be stored under a
  distinct field/name (e.g. `key_id_section_resolved`), never overwriting
  or aliasing Stage 1's own frozen `key_id`** — Stage 6 is a display step,
  not a new source of truth for Stage 1's output.

## 3. Segment detection (label-free)

Run-length-encode Stage 2's per-window collection sequence (already
forward-filled through inactive windows, exactly as Stage 1's `key_id`
is). A candidate segment is any maximal run of constant collection with
length ≥ `MIN_SEGMENT_WINDOWS` (proposed default: 4 windows / 2.0s,
matching Phase 3H-B's `STABILITY_WINDOW_WINDOWS` for consistency, not
because it is known to be optimal for this purpose).

**Brief-interruption handling**: a run shorter than `MIN_SEGMENT_WINDOWS`
that sits between two segments of the *same* collection on both sides is
merged into the surrounding segment (treated as noise). A short run
between two segments of *different* collections is harder: this design
proposes leaving it as its own short, explicitly-labeled "transition"
segment — excluded from Stage 5's per-segment decision and reported
separately — rather than silently absorbing it into either neighbor.
**A segment-level tie-break rule is also required** for the case where
Stage 5's aggregated evidence itself is close/ambiguous for an otherwise
well-formed segment (not just a boundary artifact): the default proposal
is to fall back to the segment's own most-frequent Stage 1 `key_id` (a
majority vote over the segment's frozen per-window predictions) rather
than an arbitrary index-order tie-break, since a near-tie at the segment
level is consequential over dozens of seconds, unlike a single window.

This segmentation pass is proposed as an **offline, whole-sequence**
operation — it may use windows both before and after a given point,
unlike Phase 3H-B's strictly causal gates. This is deliberate:
segmentation is a post-hoc structural analysis of Stage 1's already-fully-
computed output, not a real-time decision rule, and no anchor/label is
used either way, so non-causality does not reopen any leakage concern —
but it is a different design posture than Phase 3H's online-style gates
and is flagged here explicitly for future reviewers.

Explicitly NOT used for segment detection: raw per-window
`normalized_margin` (Phase 3B found this structurally saturated near 100%
"low margin" and non-discriminating on its own) or the weighted-margin
gate that was Phase 3H-C's own subject of study. Segmentation relies on
the *resolved* Stage 1 collection sequence, not raw per-window
confidence, to avoid re-importing the problem this design exists to
escape.

**Piece-specific caveat, stated here rather than only in §7**: Bach's
three predicted collections (C/G/D-major) are *different* collections
under Stage 2, not a within-collection tie — so segmentation on Bach risks
manufacturing several "segments" that look like real modulations, when
Phase 3G-B already showed this is a systematic tie-break/index-order
bias, not genuine tonicization. Segment *count* on Bach may be
structurally misleading independent of segment length, and should be
interpreted with that caveat rather than taken at face value.

## 4. Segment-level evidence aggregation

Proposed aggregation target: **sum of raw (unthresholded) chroma across
all windows in a segment**, then compute segment-level scores by the same
linear scale-template matmul Stage 1 already uses. This is mathematically
precise, not just directional: for a fixed linear template matrix
(`SCALE_TEMPLATES` or a weighted profile), `sum_t(chroma_t) @
TEMPLATES.T == sum_t(chroma_t @ TEMPLATES.T)` by linearity — so
aggregating raw chroma first and scoring once is *identical* to summing
Stage 1's own per-window `raw_scores`. This does **not** hold if
per-window *thresholding* happens first (thresholding is nonlinear), so
aggregation must use pre-threshold evidence (raw or smoothed chroma, or
equivalently unthresholded per-window scores), not Stage 1's already-
thresholded values.

Also record, per segment: total active-pitch-class count (expected to be
far higher than any single window's — directly addressing the
sparse-evidence mechanism Phase 3H-A documented), segment length in
windows/seconds, and — for Candidate A only — the aggregated
weighted-profile score.

**On why aggregation should help, stated as a hypothesis, not a
guarantee**: Phase 3H-A's mechanism finding was that a *single* recently-
played note dominates one window's thresholded chroma, and under a
tonic-weighted template that one note is treated as if it alone defines
the tonic. Over a multi-second segment, a real melody typically traverses
several scale degrees, so the aggregate chroma *should* reflect the tonic
and several supporting degrees rather than whichever single note sounded
last. **This is an empirical question, not a settled result** — it is
entirely possible for aggregation across a genuinely ambiguous or
modally-mixed passage to produce a confident-looking wrong or averaged
answer instead, and this design does not claim otherwise (this framing
was corrected during review — see §9). The structural difference from
Phase 3H-B's density *gate* is real (aggregation changes the evidence
itself rather than only deciding whether to trust already-sparse
evidence), but it is a hypothesis to be tested in Phase 3J-B, not a
conclusion already reached here.

## 5. Tonic/mode resolver candidates (design only, none implemented)

- **Candidate A — aggregate weighted profile over the segment.** Apply
  Phase 3H-A's exact, unchanged `WEIGHTED_TEMPLATES` to the segment's
  aggregated raw chroma, argmax once. Most direct translation of the
  existing weighted-profile idea to the new timescale; carries over all
  of variant D's other degree weights (tonic/dominant/supertonic/etc.),
  not just the mediant.

- **Candidate B — mode-defining scale-degree comparison (mediant +
  leading-tone) within the already-known collection.** Stage 3 already
  identifies which collection (relative major/minor pair) a segment
  belongs to, so only two candidate tonics remain. The original draft of
  this candidate compared aggregate evidence only at the major-third vs.
  minor-third pitch class (the one diatonic degree that differs between
  the two candidates). **Revised after music-theory review**: this alone
  misses the historically stronger mode cue — the raised leading tone of
  harmonic/melodic minor. For a minor tonic at pitch class `t`, its
  raised leading tone sits at `(t+11) % 12` — a pitch class that is
  chromatic to (foreign to) *both* members of the collection under the
  plain diatonic `SCALE_TEMPLATES` representation (it is not the
  relative major's own diatonic 7th, since the two tonics differ). Real
  aggregate chroma evidence at that specific pitch class is therefore a
  comparatively unambiguous, label-free, minor-tonic-confirming signal
  (evidence of an actual dominant-preparation gesture), distinguishing
  genuine harmonic/melodic-minor practice from a merely diatonic,
  ambiguous passage. Revised Candidate B compares: (i) aggregate evidence
  at the major-third vs. minor-third pitch class, and (ii) whether the
  minor candidate's raised-leading-tone pitch class shows meaningful
  aggregate presence — combining both into the segment's tonic/mode
  decision. This remains narrower and more directly traceable to a
  specific music-theoretic claim than Candidate A.

- **Candidate C — cadence/terminal evidence.** Weight evidence from a
  segment's final windows more heavily, reflecting cadential-resolution
  theory. **Revised after music-theory review**: the original
  "longest-duration/highest-chroma-value note near the segment's end"
  heuristic was flagged as capturing static presence, not the
  dominant→tonic resolution that actually defines a cadence, and as
  systematically weighting the *least* reliable windows in a segment
  (those nearest a boundary are, by construction, the ones where
  collection stability was about to break — the reason a boundary exists
  there at all). Revised proposal: restrict Candidate C to
  anchor-*confirmable* segments only (i.e., pieces/segments where an
  independent, already-established qualitative shape exists from prior
  phases, such as Clementi's known C→G→C excursion) rather than applying
  it corpus-wide, and replace the static heuristic with an explicit
  penultimate-window-to-final-window pitch-class transition check
  (approximating an actual harmonic resolution rather than a sustained
  pitch). Flagged as the most speculative of the three candidates and the
  lowest priority for Phase 3J-B.

All three are candidates for future comparison; **none is implemented or
selected in this document** (see §8 for the recommended first candidate
to test).

## 6. Evaluation plan

Reuse the exact sparse-anchor convention already established (same
anchor windows, same `strict_expected_key_proportion` /
`collection_equivalent_proportion` definitions as `compute_anchor_
metrics`), applied to Stage 6's window-projected output (`key_id_
section_resolved`, never Stage 1's own `key_id`), so results remain
directly comparable to every existing Phase 3G/3H number. **Anchors are
consulted only after Stage 5's segment-level decisions and all governing
constants (§3's `MIN_SEGMENT_WINDOWS`/interruption rule, §5's candidate
choice) are already fixed — never inside segmentation, aggregation, or
any resolver candidate, and never used to revise a constant after the
fact** (see §9's leakage-audit note on the Clementi sanity check, which
must not be treated as an exception to this rule).

Additional segment-level-only descriptive metrics (new, not accuracy
claims): number of segments per piece, mean/median segment length
(windows and seconds), fraction of a piece's windows covered by
unresolved "transition" segments, and — only where an anchor window
happens to overlap a detected segment — whether that segment's decision
matches the anchor's expected key/collection. Many segments will have no
anchor coverage at all (anchors are sparse — e.g. only 3 anchor windows
total across all of Twinkle 12's 1374 windows) and must be reported as
descriptively evaluated only, not scored. **No dense per-timestep
accuracy is claimed anywhere in this plan.**

**Governance, mirroring Phase 3H-C's own precedent**: before any
candidate is run against any anchor, `MIN_SEGMENT_WINDOWS`, the
interruption-merge rule, the segment-level tie-break rule, and the choice
of which resolver candidate(s) to test must all be fixed and stated in
writing — exactly as Phase 3H-C's 54-condition grid was "fixed before
running, not adjusted afterward." A small, predeclared sensitivity check
over these free parameters (in the spirit of, though not necessarily the
same scale as, Phase 3H-C's sweep) is recommended before treating any
single configuration's results as validated, since §7 identifies
over-segmentation as a live risk tied directly to these same parameters.

## 7. Risks and failure modes

- **Over-segmentation.** If `MIN_SEGMENT_WINDOWS` is too short or the
  interruption-merging rule too permissive, high-fragmentation pieces
  could produce many segments too short to aggregate meaningfully —
  reintroducing the sparse-evidence problem one level up. This risk is
  **not** well predicted by raw switch count alone: Chopin's
  `collection_equivalent_proportion` is only **0.491** (barely half its
  windows land in the right collection even after merging relative
  pairs) versus Für Elise's **0.717** — Chopin's fragmentation risk is
  qualitatively worse than its 38-switches/216-windows figure alone
  suggests, and worse than Für Elise's, which should mostly resolve to
  one dominant segment with short transition runs.
- **Segment-level wrong-tonic locking.** A per-window wrong prediction is
  self-limiting (Stage 1 already switches fairly often); a wrong
  *segment*-level decision applies the same wrong label to every window
  in that segment at once — which could look more confident/stable in a
  trajectory plot while being no more (or less) correct. A real
  interpretability trap for anyone reading a resulting plot without this
  caveat in mind.
- **Minor recovery at the cost of Twinkle/Twinkle 12 stability, in a new
  form.** Twinkle is already one long, correctly-resolved segment under
  Stage 1 (100% C major, 0 switches). Whether aggregating its full ~53s
  of monophonic evidence still unambiguously favors the correct tonic
  under Candidate A or B is a genuine empirical question this design
  cannot resolve on paper — it must be checked in Phase 3J-B before
  either candidate is trusted on monophonic single-segment pieces.
- **Twinkle 12's C→Eb→C boundaries.** Segment detection must find
  something close to the real modulation boundaries (~384s, ~432s)
  without over-fragmenting on the piece's substantial existing
  ornamentation-driven switching (11 key switches, 5 large jumps under
  the frozen control) — a genuine test of whether label-free segmentation
  can locate real structure without anchor information.
- **Bach's tonic-neighborhood ambiguity may not average out.**
  Aggregation helps with *random* sparse-evidence noise, but Bach's
  errors are a *systematic* tie-break bias (100% of mismatches land in
  C/G/D) — plausibly, though not certainly, this bias could persist at
  the aggregate level rather than being averaged away, since it is
  structural rather than noise-driven. This is offered as a hypothesis to
  watch for in Phase 3J-B, not a predicted or assumed outcome.
- **Clementi's C→G→C is a real, useful sanity check.** Because Phase
  3G-A/3G-B already established its true qualitative shape independent
  of exact anchor timing, segment detection should produce roughly three
  segments (C, G, C) on this piece. This comparison is useful precisely
  because it is cheap and informative — but per §6's governance rule, it
  must happen only *after* `MIN_SEGMENT_WINDOWS` and the interruption
  rule are already locked, and must never be used to revise those
  constants after the fact; observing it is a validity check on an
  already-fixed design, not a tuning signal.

## 8. Recommendation

**Recommend proceeding to a Phase 3J-B design-and-implement step, testing
the revised Candidate B (mediant + leading-tone comparison) first**,
subject to the governance rule in §6 (all constants fixed in writing
before any anchor comparison, including a small predeclared sensitivity
check on `MIN_SEGMENT_WINDOWS`/interruption handling).

Rationale: Candidate B is the most surgical of the three — it targets the
*specific*, already-diagnosed mechanism (relative major/minor tie +
major-first tie-break) with a minimal, fully interpretable rule, rather
than reintroducing all of variant D's other degree weights (Candidate A)
or resting on an unverified assumption about segment-boundary/cadence
alignment (Candidate C, which is recommended only as a secondary,
Clementi-scoped check, not a general-purpose resolver). Candidate B is
also the cheapest to falsify quickly: if aggregated third-degree and
leading-tone evidence still fails to recover Für Elise/Chopin's minor
tonics, that is a fast, clean signal that the sparse-evidence hypothesis
in §4 needs revision before any more elaborate candidate is worth
building.

This recommendation is explicitly **not** an authorization to write
implementation code — per this task's guardrails, Phase 3J-B (if
approved) is a separate, future step.

## 9. Review synthesis

Three independent, read-only critique passes reviewed the pre-review
draft of this document before it was finalized. None modified any file,
ran any code, or created any output — findings were incorporated
directly into §§2–8 above. Summary of what changed:

- **Architecture review**: flagged that Stage 3's stability criterion is
  a direct reuse of Phase 3H-B's Gate 2 (not new), that the design's real
  novelty is confined to Stage 4, that Stage 6's projected output needed
  an explicit distinct-field-name guarantee to avoid ever being confused
  with Stage 1's frozen `key_id`, and that a predeclared sensitivity
  sweep over the new free parameters (mirroring 3H-C) was missing. All
  four points incorporated (§2, §6).
- **Methodology / leakage audit**: confirmed the pipeline is
  label-free/anchor-free end-to-end (PASS on dense-accuracy claims and
  candidate label-freedom), but flagged the Clementi sanity check (§7) as
  a channel where known-answer structure could indirectly influence
  constant selection if not explicitly firewalled, and that the document
  lacked an explicit "constants fixed before running" governance
  statement analogous to Phase 3H-C's. Both incorporated (§6, §7's
  Clementi bullet reworded).
- **Music-theory critique**: identified that the original Candidate B
  (third-degree comparison only) structurally cannot see raised-
  leading-tone evidence and inherits variant D's own theoretical blind
  spot; identified that Chopin's fragmentation risk is better predicted
  by its 0.491 collection-equivalent proportion than its raw switch
  count; identified that Bach's segmentation risks manufacturing
  apparent modulations from what is actually tie-break bias; identified
  that §4's original aggregation-benefit claim was stated with more
  confidence than warranted; and identified that Candidate C's original
  heuristic weighted the least reliable evidence in a segment and
  conflated static pitch presence with cadential resolution. All five
  incorporated (§5's Candidate B revision, §3's Bach caveat, §7's Chopin
  figures, §4's hedged framing, §5's Candidate C revision).

---

## Verification

- No implementation code was written or run for this task. This document
  and its pre-review draft (a scratch file outside the project directory,
  used only to brief the three review subagents) are the only artifacts
  produced.
- No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement was
  run or implemented.
- No Phase 3G, 3H, or 3I output file was opened for writing, modified, or
  overwritten by this task.
- No old Phase 2C/2D/3B/3C script was modified.
- The three review subagents were structurally restricted from file
  modification (agent type without Edit/Write tool access) and were
  additionally instructed explicitly not to edit, write, or create any
  file; each returned only a text critique, confirmed by inspecting their
  responses (no file-modification tool calls appear in their output).
- This report exists at
  `05_Figures_Results/PHASE3J_A_section_level_resolver_design.md` and is
  non-empty.
- No dense per-timestep accuracy is claimed anywhere above — every
  quantitative reference is either a documented figure already published
  in a frozen Phase 3G/3H report, or an explicitly-labeled descriptive/
  segment-level metric with sparse anchor coverage.
- Anchors are discussed exclusively as post-hoc, sparse evaluation
  references throughout (§1, §6, §7's Clementi bullet, §9) — never as an
  input to segmentation, aggregation, or any resolver candidate.
- **Phase 3J-B implementation is recommended (§8), but only as a future,
  separate, explicitly-authorized step** — nothing in this task
  constitutes that authorization, and no implementation code was written
  here.
