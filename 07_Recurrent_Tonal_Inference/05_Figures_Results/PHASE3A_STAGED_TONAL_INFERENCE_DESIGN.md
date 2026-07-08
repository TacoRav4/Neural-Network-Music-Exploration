# Phase 3A — Staged Tonal-Inference Architecture Design

**Design document only. No code, model, or pipeline has been implemented. No existing scripts, notebooks, or Phase 1/1.5/2 outputs have been modified.**

This designs the QuickBin-like staged architecture proposed in `BRIAN_EXTENSION_IDEAS_PLAN.md` section 3 (Brian idea B), at the level of interfaces and responsibilities per stage — not implementation.

## 1. Research motivation

- **Phase 1** (`PHASE1_SYNTHETIC_EMA_vs_SRN_SUMMARY.md`): a learned Elman SRN improves over the hand-coded EMA baseline on clean synthetic chord sequences (overall accuracy 0.8101 vs. 0.7529; modulation accuracy 0.8010 vs. 0.7485). Recurrence helps, but only demonstrated so far on a clean, non-triadically-distorted representation.
- **Phase 1.5B** (`PHASE1_5B_MIDI_EMA_vs_SRN_SUMMARY.md`): on real MIDI-derived chord-id sequences, the SRN cannot rescue triadic-forced input. It becomes smoother/more confident than EMA but remains biased toward F major on both test pieces, and assigns *less* average probability to the true C-major tonic than EMA does. Recurrence over a bad representation does not fix the representation.
- **Phase 2D** (`PHASE2D_pitch_class_baseline_report.md`): the pitch-class/scale-template baseline — non-neural, no chord-template intermediate, no MLP, no recurrence of any kind — recovers Twinkle.mid's stable C-major tonic (100%) and Twinkle 12.mid's real, embedded C → Eb → C modulation (100% / 89.6% / 94.1% across the three key-signature-aligned windows) far better than either chord-id temporal-memory condition.
- **Therefore, the fast filter should be preserved, not replaced.** Phase 2D is not a stepping stone to be discarded once something "better" is built — it is the current best-performing component in this entire workspace for real-MIDI tonal tracking, non-neural or not. A staged architecture that keeps it as the default path, and only invokes additional (and currently unproven, on real MIDI) neural machinery where the fast filter demonstrably struggles, is the design that best respects what Phase 1/1.5B/2D have actually shown so far.

## 2. Overall architecture

Five stages, each with a narrow, inspectable responsibility. Stages 1-3 are fully specified below (interfaces only, no code). Stage 4 is deliberately left as "candidate, not implemented." Stage 5 is the evaluation discipline that governs all of the above.

### Stage 1: Pitch-class fast filter

- **Input:** raw and/or smoothed 12-dim chroma sequence, `(T, 12)` — reuses Phase 2C's extraction path and Phase 2B's `pitch_class_baseline.py` exactly (frozen, not modified).
- **Output:** a key-prediction sequence, `(T,)` int key ids (0..23), and optionally the raw `SCALE_TEMPLATES` dot-product scores per timestep, `(T, 24)` float (not currently returned by `midi_to_key_baseline`, but the underlying computation already exists inside it and could be exposed — see section 3).
- **Role:** the robust default tonal estimate for the entire piece. Every downstream stage treats Stage 1's output as the baseline to be checked, anchored against, and selectively refined — never silently overridden.

### Stage 2: Anchor / metadata layer

- **Input:** whatever weak ground truth exists for a given piece — key-signature events (Twinkle 12.mid: confirmed via `pretty_midi` in Phase 2C), a known/assumed tonic for simple pieces (Twinkle.mid: C major, established since the original COGS 202 baseline), section/phrase boundaries if available (not yet extracted for either piece), and in principle future human-annotated or symbolic score labels.
- **Output:** a small set of sparse anchor constraints or evaluation windows — e.g. Phase 2D's three-window split for Twinkle 12.mid (`pre_384s` / `384_to_432s` / `post_432s`), each tagged with an expected key.
- **Role:** weak ground-truth / sanity-check structure. This stage does not predict anything; it only supplies the (sparse, real, but non-dense) reference points that Stages 3 and 5 use to judge Stage 1's output.

### Stage 3: Uncertainty and disagreement diagnostics

- **Input:** Stage 1's key sequence (and raw scores, if exposed), Circle-of-Fifths jump statistics (already computed the same way in Phase 2D and Phase 1.5B), and — where available — the chord-id EMA/SRN outputs from Phase 1.5B (`PHASE1_5B_MIDI_EMA_vs_SRN_metrics.json`) for the same piece.
- **Output:** a list/mask of "difficult windows" — timestep ranges where Stage 1's estimate is judged less trustworthy (see section 4 for the exact candidate criteria).
- **Role:** identify *where* a neural refinement might plausibly help, before deciding *whether* to build one. This stage is diagnostic, not corrective — it does not change any prediction, it only flags regions for further attention.

### Stage 4: Neural refinement candidate

- **Not implemented yet.** A Chroma SRN or (later, per `BRIAN_EXTENSION_IDEAS_PLAN.md`) a Transformer, applied only to the difficult windows Stage 3 identifies — either as a targeted corrective pass, or purely for evaluation (measuring whether such a model would even change the prediction in those windows, before committing to training one end-to-end).
- This stage exists in the design so Stages 1-3's interfaces can be specified with it in mind, but no model, dataset generator, or training code for it is written in this document or elsewhere in this task.

### Stage 5: Evaluation gate

- **Compares:** Stage 1 alone, vs. any future staged/neural refinement, vs. the existing chord-id EMA/SRN (Phase 1.5B), on the same pieces.
- **Uses:** the same descriptive-metric discipline established in Phase 1.5B and Phase 2D — key-region stability (dominant-key proportion, unique-key count), confidence/entropy (where a probability distribution exists), Circle-of-Fifths jumpiness (mean/max jump, large-jump fraction), and key-signature-aligned window checks (proportion of expected key per anchor window).
- **Does not** claim dense per-timestep accuracy without labels — real MIDI still has none. This is a hard carry-over from every phase so far, not a new rule.

## 3. Interface design

### Stage 1: Pitch-class fast filter

- **Inputs:** a MIDI path (or, more efficiently, an already-extracted chroma array from Phase 2C's saved `.npy` files) plus `window_sec`/`memory_decay` settings — exactly `pitch_class_baseline.midi_to_key_baseline`'s existing signature.
- **Outputs:** `key_ids: (T,) int64`; optionally `raw_scores: (T, 24) float64` (the `SCALE_TEMPLATES.dot(working_chroma)` value per timestep, per key — currently computed and discarded inside the function's loop; exposing it would require a minimal, additive change to `pitch_class_baseline.py`, not a rewrite, and should be scoped as an explicit small Phase 3B task rather than assumed here).
- **File formats:** `.npy` for `key_ids` and (if exposed) `raw_scores`; `.json` for accompanying metadata, following the exact convention already used by `midi_chord_extraction.py` and `midi_chroma_extraction.py`.
- **Expected shapes:** `(T,)` for `key_ids`, `(T, 24)` for `raw_scores` if produced. `T` varies per piece and is generally slightly less than the raw chroma window count, due to leading-silence dropping (see the Phase 2D bug note below).
- **Required metadata:** `midi_path`, `window_sec`, `memory_decay`, `n_chroma_windows`, `n_key_predictions`, and critically **`offset_windows = n_chroma_windows - n_key_predictions`** — the leading-silent-window count that must be added back when mapping `key_ids` indices to real wall-clock time. This was a real bug caught and fixed during Phase 2D (`evaluate_pitch_class_phase2d.py`); any Stage 1 output consumed by later stages must carry this offset explicitly, not leave it to be silently rediscovered.
- **What should be saved to disk:** `key_ids`, metadata (including the offset), and `raw_scores` if the exposing change is made. This mirrors Phase 1.5A/2C's existing derived-data convention.
- **What should not be saved yet:** any softmax-normalized probability array (not yet decided whether/how to normalize — see section 4), and no "difficult window" mask (that is Stage 3's output, not Stage 1's).

### Stage 2: Anchor / metadata layer

- **Inputs:** `pretty_midi` `key_signature_changes` (already extracted read-only in Phase 2C), a manually-recorded known tonic for pieces without embedded key signatures (e.g. Twinkle.mid: C major), and, if later available, section/phrase boundary timestamps.
- **Outputs:** a small, structured list of anchor windows, each with `{start_sec, end_sec, expected_key_name, source}` (`source` distinguishing "embedded key signature" from "assumed tonic" from "manual annotation," so provenance is never ambiguous).
- **File formats:** `.json`, small enough to hand-edit/review directly (unlike Stage 1's per-timestep arrays).
- **Expected shapes:** not array-shaped — a short list, typically single-digit length per piece.
- **Required metadata:** which MIDI file the anchors belong to, and the `source` field above for each anchor.
- **What should be saved to disk:** the anchor list itself, in a `PHASE3_`-prefixed derived-data location (see section 6) — kept separate from Phase 2C's raw chroma/key-signature dumps even though the underlying facts (e.g. Twinkle 12.mid's key-signature timestamps) are the same, since Stage 2's role is to package them as *anchors for evaluation*, not just record them as *MIDI metadata*.
- **What should not be saved yet:** no attempt to interpolate or infer anchors where none exist (e.g. no guessed section boundaries) — only real, sourced anchors belong in this stage's output.

### Stage 3: Uncertainty and disagreement diagnostics

- **Inputs:** Stage 1's `key_ids` (+ `raw_scores` if available) and `offset_windows`; Stage 2's anchor list; optionally the corresponding Phase 1.5B chord-id EMA/SRN `probs` arrays (`(T', 24)`, on the chord-id path's own timestep grid, which is **not** the same `T` as Stage 1's chroma-grid timesteps — any cross-referencing must account for the two paths' independently-derived timestep counts, not assume index alignment).
- **Outputs:** a `(T,)` boolean or categorical "difficulty" array (or a list of `(start_idx, end_idx, reason)` windows), using the criteria in section 4.
- **File formats:** `.npy` for the per-timestep difficulty array, `.json` for the window-list form and any accompanying summary statistics.
- **Expected shapes:** `(T,)`, matching Stage 1's `key_ids` length exactly (same timestep grid, same offset convention).
- **Required metadata:** which criteria contributed to each flagged window (not just a single opaque boolean), so a future Stage 4 decision (or a human reviewer) can see *why* a window was flagged.
- **What should be saved to disk:** the difficulty array/window list and the per-criterion breakdown.
- **What should not be saved yet:** no refined/corrected key predictions — Stage 3 flags, it does not fix.

## 4. Difficult-window definition candidates

Proposed criteria for Stage 3 (not all need to be implemented at once in Phase 3B — see roadmap):

- **Low top1-top2 scale-template margin:** the gap between the highest and second-highest `SCALE_TEMPLATES.dot(working_chroma)` score at a timestep. A small margin means the fast filter's argmax choice was a close call, even though the current baseline reports it with the same apparent certainty as an unambiguous window.
- **High entropy after optional softmax over scale scores:** since `midi_to_key_baseline` currently returns raw dot-product scores, not probabilities, entropy requires an explicit (and currently undecided) softmax normalization step — flagged here as a design decision for Phase 3B, not resolved in this document.
- **Key switches:** timesteps where the Stage 1 argmax key changes from the previous timestep — natural candidates for refinement since transitions are inherently harder than stable regions, and this is exactly where Phase 1.5B's modulation-lag metric was also concentrated.
- **Large Circle-of-Fifths jumps:** reusing the existing `fifth_distance` jump-size metric from Phase 1.5B/2D (distance ≥ 3 flagged as "large" throughout this workspace) — a large jump between consecutive predictions is a stronger signal of instability than a same-neighborhood switch.
- **Disagreement with chord-id EMA/SRN outputs:** timesteps (after resolving the two paths' independent timestep grids per Stage 3's input note above) where Stage 1's key differs from Phase 1.5B's EMA and/or SRN prediction — since Phase 1.5B and Phase 2D used genuinely different representations, agreement is a positive signal and disagreement flags a window worth a closer look.
- **Mismatch with known key-signature anchors:** within a Stage 2 anchor window, timesteps where Stage 1's prediction is not the anchor's expected key — directly reuses Phase 2D's `proportion_expected_key` logic, but at the per-timestep level rather than aggregated per window.
- **Sudden drops in expected-key score near section boundaries:** for pieces with section/phrase boundary anchors (not yet available for either test piece), a drop in the expected key's raw score immediately around a boundary would flag exactly the kind of transition region a staged neural refinement is meant to target.

## 5. Minimal implementation roadmap

- **Phase 3A (this document):** design only. Done.
- **Phase 3B:** implement pitch-class uncertainty diagnostics — the first new code in this line of work, and still non-neural. Concretely: expose `raw_scores` from `pitch_class_baseline.py` (additive change, or a new wrapper function that doesn't discard them), compute top1-top2 margin and (a decided) entropy measure, key-switch boundaries, and Circle-of-Fifths jump statistics per timestep (reusing existing jump-computation logic rather than rewriting it).
- **Phase 3C:** compare Phase 3B's difficult windows against Phase 1.5B's chord-id EMA/SRN disagreement — i.e. actually compute the "disagreement with chord-id models" criterion from section 4, which requires resolving the two paths' timestep-grid mismatch first.
- **Phase 3D:** only after 3B and 3C, decide whether the first neural refinement should be a Chroma SRN or a Transformer (or neither, if the difficult-window analysis suggests refinement isn't warranted) — informed by what the diagnostics actually show, not decided upfront.

## 6. Guardrails

- **Do not replace the pitch-class baseline yet.** It remains the best-performing real-MIDI component found so far; Phase 3's job is to characterize its failure modes, not to supersede it prematurely.
- **Do not implement the Transformer before diagnostics.** Per `BRIAN_EXTENSION_IDEAS_PLAN.md`, it is explicitly a later (Phase 4/5) option, contingent on what Phase 3B/3C find.
- **Do not implement the Chroma SRN before diagnostics.** Same reasoning — Phase 3D's model choice should be evidence-driven, not assumed.
- **Keep all outputs prefixed `PHASE3_`** (or a more specific `PHASE3A_`/`PHASE3B_`/etc. as each sub-phase produces its own artifacts), so they are never confused with Phase 1/1.5/2 outputs.
- **Preserve Phase 1/1.5/2 outputs unchanged.** Nothing in this design implies modifying `shared_music_defs.py`, `sequence_dataset.py`, `mlp_baseline.py`, `srn_model.py`, `run_comparison.py`, `diagnose_srn_training.py`, `plotting_comparison.py`, `midi_chord_extraction.py`, `midi_chroma_extraction.py`, `pitch_class_baseline.py`, `run_midi_phase15_evaluation.py`, or `evaluate_pitch_class_phase2d.py` — Phase 3B's proposed "expose `raw_scores`" change (section 3) should be scoped, reviewed, and approved as its own small step when Phase 3B actually begins, not assumed as already authorized by this design document.
- **Do not claim dense MIDI accuracy without labels.** Stage 5's evaluation gate (section 2) and every diagnostic in section 4 are descriptive/comparative, not accuracy claims against ground truth that does not exist.
