# Brian Extension Ideas — Planning Document

**Planning only. No Transformer, Chroma SRN, or QuickBin-like staged pipeline has been implemented. No existing scripts, notebooks, or Phase 1/1.5/2 outputs have been modified.**

## 1. Current state after Phase 2D

- **Phase 1** (`PHASE1_SYNTHETIC_EMA_vs_SRN_SUMMARY.md`): on clean synthetic chord sequences, a learned Elman SRN improves over the hand-coded EMA baseline on both overall accuracy (0.8101 vs. 0.7529) and modulation accuracy (0.8010 vs. 0.7485). Recurrence helps, when the input representation is clean.
- **Phase 1.5B** (`PHASE1_5B_MIDI_EMA_vs_SRN_SUMMARY.md`): on real MIDI-derived chord-id sequences, the SRN cannot rescue distorted triadic inputs. It becomes smoother and more confident (lower entropy, higher top-1 probability) than EMA, but remains biased toward F major on both Twinkle.mid and Twinkle 12.mid, and in fact assigns *less* average probability to C major than EMA does. Learned recurrence over a bad representation does not fix the representation.
- **Phase 2D** (`PHASE2D_pitch_class_baseline_report.md` / `PHASE2D_pitch_class_baseline_metrics.json`): the pitch-class/scale-template baseline (non-neural, bypasses triadic forcing entirely) strongly recovers the real MIDI key structure:
  - **Twinkle.mid: 100% C major** (0 unique alternative keys, 0 fifths-jumps).
  - **Twinkle 12.mid: 90.9% C major overall**, and critically, in the key-signature-aligned windows: **pre_384s = 100% C major**, **384-432s = 89.6% Eb major** (reported as "D# maj" — same key, sharp-spelled by this workspace's `decode_key` convention), **post_432s = 94.1% C major**. The baseline tracks the piece's real, embedded C → Eb → C modulation closely.
- **Therefore: the main bottleneck is representation, not simply temporal memory.** A non-learned model with a better input representation (full scale evidence instead of hard 24-way triads) outperforms both temporal-memory conditions (hand-coded EMA and learned SRN) built on the worse representation, on every descriptive measure available. This reframes where future modeling effort should go.

## 2. Brian idea A: Transformer approach

**What it would be used for:** a self-attention sequence model over chroma or pitch-class evidence, to test whether a richer, longer-range temporal architecture (vs. the SRN's single recurrent hidden state) captures tonal structure that recurrence alone misses — e.g. delayed resolution of an ambiguous passage using context that arrives many timesteps later.

**Possible input representations:**
- 12-dim chroma sequence (raw or smoothed, per Phase 2C's saved arrays).
- Symbolic pitch-class sequence (categorical, closer to the existing chord-id/key-id conventions).
- Hybrid chroma + event markers (e.g. concatenating a chroma vector with a flag for phrase boundaries or detected onsets).

**Possible outputs:**
- 24 key probabilities per timestep (consistent with everything else in this workspace).
- Section-level key labels (one label per detected/assumed section, coarser than per-timestep).
- Next-note / next-chroma prediction as a self-supervised pretraining objective, before any supervised key-labeling head is trained.

**Strengths:**
- Long-range context beyond a single recurrent hidden state's effective memory.
- Flexible temporal dependencies (attention can look arbitrarily far back or forward, unlike a strictly causal recurrent update).
- Could in principle model delayed modulation recognition and repeated motifs (e.g. recognizing that a passage echoes an earlier theme in a different key).

**Risks:**
- Needs meaningfully more training data than anything built so far in this workspace — the synthetic dataset (`sequence_dataset.py`) was sized for a small SRN, not a Transformer.
- More black-box than the current interpretable pipeline (chord/scale templates, explicit EMA formula, a 24→48→24 MLP) — harder to diagnose failure modes the way Phase 1.5B and 2D's descriptive metrics did.
- May overfit synthetic data even more readily than the SRN did, given more parameters and the same small, clean synthetic distribution — worsening the exact "helps on synthetic, fails on real MIDI" gap Phase 1 vs. 1.5B already exposed once.
- **Unnecessary before simpler pitch-class and SRN baselines are exhausted.** Phase 2D's single-baseline result already outperforms both temporal-memory conditions; there is no evidence yet that any additional temporal-modeling capacity (SRN or Transformer) is the limiting factor.

**Recommendation:** treat the Transformer as a **later Phase 4 or Phase 5** step, not an immediate next step. It should only be attempted after (a) the pitch-class baseline's own failure cases are understood (Phase 3B below) and (b) a staged/simpler neural refinement (Chroma SRN or the QuickBin-like staged approach) has been tried and found insufficient on its own.

## 3. Brian idea B: QuickBin-like staged tonal inference approach

Mapping the QuickBin-style "fast filter → anchor → refine" logic onto this project:

### Stage 1: Fast filter
- The pitch-class/scale-template baseline (`pitch_class_baseline.py`), already modularized (Phase 2B) and evaluated (Phase 2D).
- Quickly identifies the likely global/local key region using only 12-dim chroma evidence and a fixed template match — no training required.
- Robust to melody and ornamentation in practice: Phase 2D showed 100% C major on Twinkle.mid and correct tracking of Twinkle 12.mid's real modulation, despite neither piece being template-matched at the chord level.
- **Already validated by Phase 2D** — this stage does not need to be re-implemented, only reused.

### Stage 2: Symbolic / weak ground-truth anchors
- Key-signature events, where present (Twinkle 12.mid: C major at t=0s, Eb major at t=384s/392s, C major at t=432s/440s — confirmed via `pretty_midi`, read-only, in Phase 2C).
- Known tonic for simple pieces (Twinkle.mid's tonic is C major, established by the original COGS 202 baseline and reconfirmed by every phase since).
- Section markers / phrase boundaries, if available (not yet extracted for either piece in this workspace — a candidate for Phase 3A's design work, not something to build today).
- Possible human-annotated or symbolic score labels, if such annotations exist or are feasible to add later (out of scope for now; noted only as a future anchor source).
- **For Twinkle 12.mid specifically:** use the confirmed C → Eb → C key-signature markers as anchors, the same three-window split (`pre_384s` / `384_to_432s` / `post_432s`) already used in Phase 2D's evaluation.

### Stage 3: Neural refinement
- A learned model (Chroma SRN, later a Transformer, or another sequence model) — but critically, **only asked to improve difficult regions**, not to re-derive the whole key trajectory from scratch:
  - Ambiguous windows (chord/pitch-class evidence consistent with multiple keys).
  - Modulation boundaries (where the fast filter's confidence dips or its prediction changes).
  - Ornamented passages (where the fast filter may be more prone to distraction from passing tones, per the COGS 202 paper's "ornamentation collapse" framing).
  - Places where the fast filter is measurably uncertain or unstable (per Phase 3B's diagnostics below) — this is the key idea: the neural model's training signal and evaluation focus should be concentrated where Stage 1 already struggles, not spread uniformly across regions Stage 1 already handles well.

### Stage 4: Evaluation gate
- Compare against the pitch-class baseline (Stage 1 alone), the chord-id EMA/SRN (Phase 1.5B), and any neural refinement model from Stage 3, side by side.
- Use descriptive metrics consistent with every phase so far: key-region stability, confidence/entropy (where a probability distribution exists), Circle-of-Fifths jumpiness (mean/max jump, large-jump fraction), and response around known key-signature markers (the windowed proportion-of-expected-key check from Phase 2D).
- **Do not claim dense accuracy unless labels exist** — real MIDI still has no per-timestep ground truth; this gate evaluates the same way Phase 1.5B and Phase 2D did.

## 4. Compare Transformer vs. QuickBin-like approach

| | Transformer | QuickBin-like staged approach |
|---|---|---|
| Implementation cost | High — new architecture, new/adapted dataset generator, larger training runs | Low-to-moderate — Stage 1 already exists (Phase 2B/2D); mainly needs anchor extraction (Stage 2) and diagnostics (Stage 3B) before any new model |
| Interpretability | Low — attention weights are a weaker interpretability signal than this workspace's explicit templates/formulas | High — each stage has a clear, inspectable role; failures can be localized to a specific stage |
| Data requirement | High — needs more and more-realistic synthetic (or real, labeled) sequences than currently exist | Low for Stage 1 (none — non-learned); moderate for Stage 3 (only needs to cover the difficult-region subset, not the whole trajectory) |
| Risk | Higher — more black-box, more prone to overfitting the same small synthetic distribution that already caused a real-MIDI generalization gap once | Lower — reuses an already-validated, already-interpretable Stage 1; risk is concentrated in the (deliberately small, later) Stage 3 refinement |
| Alignment with current findings | Weak — nothing in Phase 1/1.5/2D suggests more temporal-modeling *capacity* is the bottleneck; the bottleneck identified so far is representation | Strong — directly operationalizes Phase 2D's finding by keeping the validated fast filter as the default and only invoking a neural model where it demonstrably struggles |
| Recommended timing | Later (Phase 4/5), after simpler approaches are exhausted | Sooner (Phase 3), as the next concrete step |

**Expected conclusion:** the QuickBin-like staged approach should come first, because Phase 2D already shows a simple fast filter is powerful — nearly all of the real-MIDI tonal-tracking problem this workspace has been chasing since Phase 1.5B appears to already be solved by Stage 1 alone, for both test pieces. The Transformer should be treated as a later neural-refinement option, not the immediate next step, and should only be reconsidered once Stage 1's actual failure regions are characterized (Phase 3B) and found to need more than a small, targeted neural refinement (Phase 3C) can provide.

## 5. Recommended near-term roadmap

- **Phase 2E:** create a Phase 2D summary / representation-conclusion document, if `PHASE2D_pitch_class_baseline_report.md` is not already judged sufficient as that record. (This may turn out to be unnecessary — Phase 2D's report and metrics are already fairly complete; this step should be scoped only if a reviewer determines a consolidated summary, in the style of `PHASE1_SYNTHETIC_EMA_vs_SRN_SUMMARY.md`, adds value beyond what already exists.)
- **Phase 3A:** design (not implement) the staged tonal-inference architecture: pitch-class fast filter → uncertainty/anchor analysis → neural refinement target. Output should be a design document specifying exact interfaces between stages (what Stage 1 hands to Stage 2, what Stage 2 hands to Stage 3), not code.
- **Phase 3B:** implement uncertainty diagnostics for the pitch-class baseline — this is the first piece of new *code* in this roadmap, and it is deliberately still non-neural:
  - Raw scale-template score margin (difference between the top-1 and top-2 `SCALE_TEMPLATES` dot-product scores per window — a cheap, immediate confidence proxy given the baseline currently has no softmax).
  - Entropy after optional softmax normalization of the raw scores (to put confidence on the same footing as Phase 1.5B's EMA/SRN entropy metrics, for comparability).
  - Key-switch boundaries (timesteps where the argmax key changes) — where Stage 3 refinement should concentrate.
  - Circle-of-Fifths jumps (already computed in Phase 2D; reuse, don't rebuild).
  - Disagreement with the chord-id models (EMA/SRN from Phase 1.5B) — timesteps where the pitch-class baseline and the chord-id path predict different keys are natural candidates for "difficult regions."
- **Phase 3C:** only after 3A and 3B are complete and reviewed, decide whether the first neural refinement should be a Chroma SRN, a Transformer, or a hybrid model — informed by what Phase 3B's diagnostics actually show about where and how the fast filter fails, rather than deciding upfront.

## 6. Guardrails

- **Do not discard Phase 1/1.5.** They remain the evidence base establishing that recurrence helps only under a clean representation — this finding motivated the entire Phase 2 line of work and should not be treated as superseded or irrelevant now that Phase 2D looks promising.
- **Do not replace the pitch-class baseline with a neural model before understanding its failure cases.** Phase 2D evaluated only two pieces; its apparent success could mask failure modes on other material. Phase 3B's diagnostics should come before any Phase 3C model decision, not after.
- **Do not implement the Transformer before simpler baselines and uncertainty diagnostics are complete.** Per section 4, nothing in the current evidence base justifies its cost/risk yet.
- **Do not claim accuracy on real MIDI without labels.** Every phase so far (1.5B, 2D) has correctly limited itself to descriptive metrics against sparse key-signature checkpoints, not dense per-timestep accuracy; any future staged or Transformer evaluation must preserve this discipline.
- **Keep all future Brian-extension outputs separate with a clear prefix** — e.g. `BRIAN_EXT_` or `PHASE3_` — so they are never confused with Phase 1/1.5/2 artifacts, the same convention that has kept every phase's outputs distinguishable so far.
