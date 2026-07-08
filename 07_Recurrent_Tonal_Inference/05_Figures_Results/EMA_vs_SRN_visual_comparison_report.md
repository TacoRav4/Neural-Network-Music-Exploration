# EMA+MLP vs. Elman SRN — Visual Comparison Report (Phase 1)

Synthetic-only Phase 1 visual comparison between the EMA+MLP baseline (alpha=0.2) and the best Elman SRN condition found by the training diagnostic (epochs=25, lr=0.001, hidden_size=48), both on the same 24-dim one-hot chord/triad input.

## Files generated

- `05_Figures_Results/EMA_vs_SRN_C_to_G_Probability_Tracking.png`
- `05_Figures_Results/EMA_vs_SRN_C_to_G_Circle_of_Fifths.png`
- `05_Figures_Results/EMA_vs_SRN_Synthetic_Metric_Summary.png`

## Best SRN settings used

- epochs=25, lr=0.001, hidden_size=48 (from `SRN_training_diagnostic_metrics.json` `best_condition`)
- EMA+MLP baseline: alpha=0.2 (from `EMA_vs_SRN_synthetic_comparison_metrics.json`)

## Interpretation

- **The SRN improves overall accuracy**: overall_accuracy is 0.8101 for the SRN vs. 0.7529 for EMA on the synthetic test set (matches the training diagnostic's finding).
- **EMA still has slightly lower lag and switch failure**: mean modulation lag is 4.87 timesteps for EMA vs. 5.38 for the SRN, and switch_failure_rate is 0.0250 for EMA vs. 0.0375 for the SRN. The probability-tracking and Circle-of-Fifths plots above should be read alongside this: the SRN's higher accuracy does not (yet) come with faster/more reliable modulation switching than the hand-coded EMA baseline.
- The Circle-of-Fifths side-by-side plot shows the most-likely-key walk for both models on the same C major → G major sequence -- compare clustering/stability visually, not just the summary numbers.
- **This remains synthetic-only Phase 1.** No MIDI or chroma input was used anywhere in this script; `Twinkle.mid` and `Twinkle 12.mid` were not evaluated, and the true-Twinkle baseline figures were not regenerated.
- **No MIDI/chroma Phase 2 has started.** Per the workspace guardrails, both models here use the identical 24-dim one-hot chord/triad representation; a raw-chroma end-to-end SRN remains a separate, later, explicitly-labeled phase.

## Scope note

These plots use one representative C major → G major modulation sequence and one no-modulation control, freshly generated (seed=269, dedicated rng offset) rather than drawn from the held-out test set used for the metrics summary panel. The metric summary panel (`EMA_vs_SRN_Synthetic_Metric_Summary.png`) reads directly from the already-verified `EMA_vs_SRN_synthetic_comparison_metrics.json` and `SRN_training_diagnostic_metrics.json` outputs rather than recomputing test-set metrics.
