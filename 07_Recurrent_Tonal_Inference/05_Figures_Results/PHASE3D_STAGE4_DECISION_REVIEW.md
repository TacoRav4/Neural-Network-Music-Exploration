# Phase 3D — Stage 4 Decision Review

**Planning/decision document only. No code, model, or corpus has been implemented. No existing scripts, notebooks, or Phase 1/1.5/2/3 outputs have been modified.**

Reviews Phase 3B and Phase 3C to decide whether Stage 4 (neural refinement, per `PHASE3A_STAGED_TONAL_INFERENCE_DESIGN.md`) is currently justified.

## 1. Phase 3B recap

- The pitch-class/scale-template baseline is **stable on Twinkle.mid**: 100% C major throughout, 0 key switches, 0 large Circle-of-Fifths jumps across all 106 predictions.
- The pitch-class baseline **tracks Twinkle 12.mid's real C → Eb → C structure**: 100% C major in the pre-384s window, 89.6% Eb major (reported as "D# maj," same key) in the 384–432s window, 94.1% C major in the post-432s window — closely matching the piece's real, embedded key-signature events.
- **`low_margin` is structurally saturated** (~100% for both pieces) due to `SCALE_TEMPLATES` giving relative major/minor pairs (e.g. C major / A minor) numerically identical rows — sparse, monophonic chroma windows routinely tie across ~5-7 keys on average (up to 14). This is a genuine property of the representation, not a bug, and makes `low_margin` non-discriminating on its own.
- The useful difficulty signals identified instead are **`key_switch`, `large_jump`, `anchor_mismatch`, and `high_tie_count`** (a fixed, documented threshold, not tuned).
- Taken together, Phase 3B's own report characterizes the pitch-class baseline precisely: it is best understood as a **diatonic-collection fast filter** — it reliably narrows the search to "which 7-note diatonic collection is in play" — not a full tonic/mode resolver that can, on its own, distinguish a major key from its relative minor within that collection. That distinction has to come from elsewhere (e.g. weighting scale degrees by function, or a model that tracks resolution/cadence), which the current template-matching approach does not attempt.

## 2. Phase 3C recap

- **Twinkle.mid disagreement (pitch-class baseline vs. chord-id models) is global**: EMA disagrees 87.7% of the time, SRN disagrees 100% of the time.
- **Twinkle 12.mid disagreement is also global**: EMA disagrees 90.6% of the time, SRN disagrees 89.1% of the time.
- Disagreement is **not meaningfully concentrated** in the pitch-class baseline's own flagged difficult windows — concentration ratios (disagreement rate inside vs. outside flagged windows) cluster near 1.0 across all four criteria and both models (range ≈0.66–1.20), rather than showing a clear spike inside flagged regions.
- On Twinkle.mid specifically, the pitch-class baseline has **zero** flagged difficult windows of any kind (no switches, jumps, or mismatches — it's simply always right and always confident by these measures), so its own difficulty criteria cannot explain any of the chord-id models' disagreement there; the disagreement is entirely attributable to the chord-id models' own behavior.
- Conclusion: chord-id EMA/SRN failures (per Phase 1.5B's F-major bias) are **representation-wide, not local** — a property of the triadic chord-id representation and/or the model trained on it, not a small number of hard transition points that a targeted fix could address.

## 3. Decision about Stage 4

**Targeted neural refinement (Stage 4) is not currently justified on these two pieces.** Reasoning:

- **Twinkle.mid has no local pitch-class failure to refine.** The pitch-class baseline is already perfect (by every Phase 3B measure) on this piece. There is nothing for a Stage 4 model to fix, and no difficult-window signal to train or evaluate it against.
- **Twinkle 12.mid's pitch-class baseline already captures the key-signature structure.** The one real modulation in this piece (C → Eb → C) is already tracked at 89.6%–100% accuracy by Stage 1 alone. The residual gap is not large, and per Phase 3C, what disagreement *does* exist between Stage 1 and the chord-id models is not concentrated at this modulation or at any other flagged region — so there is no clear local target here either.
- **Chord-id disagreement is too global for local repair.** Phase 3A's staged-architecture premise was: keep a validated fast filter as the default, and use a *targeted* neural pass only on the specific windows where it demonstrably struggles. Phase 3C shows the chord-id models' actual failure mode (representation-wide bias) does not match this premise — a refinement scoped to "fix the hard windows" would, by Phase 3C's own numbers, leave most of the disagreement untouched, because the disagreement isn't concentrated in those windows to begin with.
- **Chroma SRN / Transformer should remain deferred**, per `BRIAN_EXTENSION_IDEAS_PLAN.md`'s original guardrails — nothing in Phase 3B or 3C changes that recommendation; if anything, Phase 3C reinforces it, since a global bias is not the kind of problem a narrowly-targeted refinement (the original justification for trying a small model first) is suited to address.

## 4. Recommended next step

**Phase 3E / Phase 4A: intermediate-difficulty MIDI corpus design.** (Design only — no corpus implementation in this task.) The corpus should include:

- A simple monophonic melody in a **non-C key** (to test whether the pitch-class baseline's apparent strength is specific to C major or generalizes to other tonics).
- A simple **minor-key** melody (to directly probe the relative-major/minor ambiguity Phase 3B identified as the representation's core structural limitation).
- A melody **with accompaniment** (chordal texture, not purely monophonic — tests behavior when the sparse-chroma/high-tie-count regime from Phase 3B doesn't apply).
- A short piece with **one clear modulation** (simpler and more isolated than Twinkle 12.mid's ornamented, high-note-density context — a cleaner test of modulation-tracking specifically).
- An **ornamented but non-modulating** melody (isolates the "ornamentation collapse" variable from the "real modulation" variable, which Twinkle 12.mid currently conflates).
- Possibly a **second classical variation example**, less extreme than Mozart's 12 Variations (Twinkle 12.mid), to get a data point between "trivially simple" (Twinkle.mid) and "maximally stressful" (Twinkle 12.mid).

## 5. Why intermediate-difficulty examples are needed

- The **current two pieces are too extreme** to usefully test the staged architecture's Stage 4 premise: Twinkle.mid is too simple (the pitch-class baseline is already perfect, nothing to refine), and Twinkle 12.mid's chord-id disagreement is too global/pervasive (not the "rare local failure" pattern Stage 4 was designed for). Neither piece occupies the middle ground where a targeted refinement would actually have something specific and bounded to fix.
- What's needed is **examples that create local, diagnosable pitch-class baseline failures** — cases where Stage 1 mostly works but has an identifiable, bounded region of difficulty, so Stage 3's diagnostics (key_switch, large_jump, anchor_mismatch, high_tie_count) can meaningfully flag *something* to test Stage 4 against.
- **Minor-key cases are especially needed** to directly test the relative major/minor ambiguity Phase 3B surfaced as the representation's central structural limitation (identical `SCALE_TEMPLATES` rows for relative pairs) — neither current test piece is in a minor key, so this specific, well-understood failure mode has never actually been exercised.
- **Only after such cases exist and are analyzed** should Stage 4 neural refinement be reconsidered — with a corpus that can actually distinguish "the refinement helped" from "there was nothing to help," which the current two-piece evidence base cannot do.

## 6. Updated staged architecture interpretation

- **Stage 1** (pitch-class fast filter) remains the main path — validated on both current test pieces, unmodified, not being replaced.
- **Stage 2** (anchor/metadata layer) remains useful — key-signature and known-tonic anchors were exactly what made Phase 2D's and Phase 3B's evaluations possible, and will be equally necessary for any future corpus.
- **Stage 3** (uncertainty/disagreement diagnostics) remains useful — Phase 3B and 3C together demonstrate it can produce real, interpretable findings (the tie-count/saturated-margin discovery; the global-vs-local disagreement question), even though on the current two pieces it did not surface a case for Stage 4.
- **Stage 4** (neural refinement) is **conditional and currently deferred** — not cancelled, not abandoned, but explicitly waiting on evidence (from a more suitable corpus) that a local, bounded refinement target actually exists.
- **Stage 5** (evaluation gate) remains descriptive, not dense-accuracy-based — this discipline has held throughout Phases 1.5B, 2D, 3B, and 3C, and should continue to hold for any future corpus, since real MIDI still has no dense per-timestep ground truth.

## 7. Guardrails

- **No neural model yet.** No Chroma SRN, and no other learned refinement of any kind, is implemented in or authorized by this document.
- **No Transformer yet.** Per `BRIAN_EXTENSION_IDEAS_PLAN.md`, still explicitly a later (Phase 4/5+) consideration, now further deferred by this review's findings.
- **No new corpus implementation in this task.** Section 4 is a design recommendation only — creating or extracting any new MIDI files, chroma sequences, or derived data for the intermediate-difficulty corpus is out of scope here and should be proposed, reviewed, and approved as its own separate step.
- **Do not discard Phase 1/1.5.** They remain the evidence base establishing that learned recurrence helps only under a clean, non-triadically-distorted representation — a finding this review does not revisit or supersede, only builds on.
- **Preserve all existing outputs.** Nothing in this review implies modifying `shared_music_defs.py`, `sequence_dataset.py`, `mlp_baseline.py`, `srn_model.py`, `run_comparison.py`, `diagnose_srn_training.py`, `plotting_comparison.py`, `midi_chord_extraction.py`, `midi_chroma_extraction.py`, `pitch_class_baseline.py`, `run_midi_phase15_evaluation.py`, `evaluate_pitch_class_phase2d.py`, `pitch_class_uncertainty_diagnostics.py`, or `compare_phase3c_disagreement.py`, nor any prior Phase 1/1.5/2/3 output file.
