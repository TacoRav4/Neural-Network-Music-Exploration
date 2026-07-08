# Phase 1 Synthetic Summary — EMA+MLP Baseline vs. Elman SRN

Consolidated summary of the Phase 1, synthetic-only comparison. Synthesizes:
- `EMA_vs_SRN_synthetic_comparison_report.md` / `EMA_vs_SRN_synthetic_comparison_metrics.json`
- `SRN_training_diagnostic_report.md` / `SRN_training_diagnostic_metrics.json`
- `EMA_vs_SRN_visual_comparison_report.md` (+ its three PNGs)

No new runs, plots, or evaluation were performed to produce this document — it only restates and frames results that already exist in the files above.

## 1. Project question

Phase 1 asks a single, narrow question: does replacing the notebook's **hand-coded EMA/leaky hidden-state integration** (`sequence_key_tracking`, `h = (1-alpha)*h + alpha*h_in`, fixed scalar `alpha`) with a **learned Elman SRN hidden state** (`h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)`, weights learned end-to-end) improve tonal-key tracking over chord sequences — while holding the input representation fixed. Both models consume the identical 24-dim one-hot chord/triad vector the COGS 202 MLP was built on; only the temporal-memory mechanism differs. This isolates "hand-coded EMA vs. learned recurrence" as the only variable under test, per the workspace guardrails (`../STATUS.md` section 8).

## 2. Baseline and SRN settings

| | EMA+MLP baseline | Elman SRN |
|---|---|---|
| Input representation | 24-dim one-hot chord/triad | 24-dim one-hot chord/triad (identical) |
| Temporal mechanism | Hand-coded leaky integration, `alpha=0.20` (fixed, not tuned) | Learned recurrent hidden state, `hidden_size=48` |
| Training | `ChordToKeyMLP` (24→48→24), seed=269, 12 epochs, 60,000 synthetic single-chord samples | `epochs=25`, `lr=1e-3` (best condition from a 6-way epoch x lr diagnostic sweep), seed=269 |
| Dataset | Synthetic-only: train/val/test = 400/90/100 labeled chord sequences, `length_range=(16,48)`, includes no-modulation negative controls | Same construction, same seed base (269; train/val/test seeds 269/1269/2269) |

All numbers below are from the held-out **test** split (100 sequences), evaluated identically for both models.

## 3. Main quantitative results

| Metric | EMA+MLP (alpha=0.20) | Elman SRN (epochs=25, lr=1e-3) |
|---|---|---|
| Overall accuracy | 0.7529 | **0.8101** |
| Modulation accuracy | 0.7485 | **0.8010** |
| Mean modulation lag (timesteps) | **4.87** | 5.38 |
| Switch failure rate | **0.0250** | 0.0375 |

The SRN leads on both accuracy metrics; the EMA baseline leads on both modulation-lag/reliability metrics. Neither model dominates across all four.

## 4. Main visual interpretation

From `EMA_vs_SRN_visual_comparison_report.md` and its plots:

- **SRN probability trajectories are sharper and higher-confidence** — on the representative C major → G major sequence, P(G major) climbs to ~0.85-0.95 shortly after the pivot for the SRN, versus a flatter, more gradual climb for EMA.
- **EMA trajectories are more diffuse** — competing keys (E minor, A minor) stay non-trivially probable for longer under EMA than under the SRN.
- **The representative C→G Circle-of-Fifths plot shows a cleaner SRN transition**: the SRN's walk moves directly F → C → G, while EMA's walk swings wider early on (starting at Bb major, passing through A minor) before settling near C/G.
- **However, the aggregate lag/switch-failure metrics (section 3) still slightly favor EMA** across the full 100-sequence test set — the single C→G example is illustrative of the SRN's sharper confidence, not a stand-in for its average timing behavior across all modulation types and distances.

## 5. Research interpretation

Read together, sections 3 and 4 support a specific, bounded claim rather than a blanket "SRN wins" or "SRN loses":

- **The SRN improves learned tonal-state classification.** Both overall accuracy and modulation accuracy are meaningfully higher for the SRN (+0.057 and +0.052 respectively), and the training diagnostic confirmed this gain scales with training (mean test accuracy rose from 0.727 at 10 epochs to 0.782 at 50 epochs across learning rates) — i.e. this is a genuine capability gain from learned recurrence, not noise from a lucky seed.
- **EMA remains a strong hand-coded temporal baseline.** Despite having zero learned temporal parameters, the fixed-alpha leaky integration matches or beats the SRN on *when* the key estimate stabilizes after a modulation (lag) and on *how often* it fails to stabilize at all (switch failure rate). A hand-tuned constant is a low bar in principle, but it is not a low bar in practice here.
- **Learned recurrence is promising but not automatically superior on every temporal metric.** The SRN's advantage is concentrated in classification accuracy, not in switching speed/reliability. This is a meaningful nuance for the eventual comparison against real MIDI: a model that classifies more accurately in aggregate is not guaranteed to track a modulation as promptly or as reliably as the simpler baseline, and both properties matter for the project's original question about tonal inference during listening.

## 6. Current limitations

- **Synthetic sequences only.** No real MIDI or audio has been evaluated in Phase 1; all numbers above come from `sequence_dataset.py`'s generated modulation sequences.
- **Hard key labels.** Each timestep has exactly one ground-truth key label (pre-pivot = key_a, post-pivot = key_b), which is a simplification — real tonal ambiguity is graded, not binary.
- **Ambiguous pivot chords.** Chords diatonic to both key_a and key_b are flagged (`is_ambiguous`) but not excluded from primary metrics by default; masked-accuracy figures exist in the underlying JSON but were not the headline numbers used above.
- **Same triadic representation as baseline.** The SRN was deliberately restricted to the same 24-dim one-hot chord/triad input as the EMA+MLP baseline (phase 1 guardrail) — it has not been given any richer input (e.g. raw chroma) that might change its behavior.
- **No raw chroma or real MIDI evaluation yet.** The entire comparison to this point is synthetic; how either model behaves on real, noisy, template-matched chord sequences (like `Twinkle.mid`) remains untested.

## 7. Next approved phase

**Recommended: Phase 1.5** — evaluate the best SRN condition (epochs=25, lr=1e-3, hidden_size=48) against the EMA+MLP baseline (alpha=0.20) on the **existing MIDI-derived chord-id sequences already used elsewhere in this workspace**:

- `Twinkle.mid` (the documented true-Twinkle baseline, `05_Figures_Results/Twinkle_mid_EMA_MLP_Baseline_RunNotes.md`)
- `Twinkle 12.mid` (the historical Mozart sequence, currently only present as the mislabeled reference PNGs)

This still uses the same 24-dim one-hot chord/triad representation (chords already extracted via the notebook's `midi_to_chord_ids` template-matching, not raw chroma) — so it is a data-source change, not a representation change, and remains consistent with the phase 1 guardrail against bundling recurrence and representation changes together.

**This task does not start Phase 1.5.** It only records the recommendation. Phase 1.5 (and any raw-chroma phase 2 beyond it) should wait for explicit approval, per `../STATUS.md`.
