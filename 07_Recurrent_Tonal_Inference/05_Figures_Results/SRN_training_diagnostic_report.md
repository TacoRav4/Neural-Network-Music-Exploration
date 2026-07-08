# SRN Training Diagnostic Sweep (Phase 1)

Focused diagnostic: does the Elman SRN's weaker showing in the first synthetic comparison (`EMA_vs_SRN_synthetic_comparison_report.md`, 10 epochs) reflect undertraining, or a more structural limitation? Sweeps epochs x learning rate only; `hidden_size` and the Elman architecture are held fixed. No plots, no MIDI/chroma, no raw-chroma SRN.

## Dataset

- Base seed: `269`
- Train sequences: 400 (rng_seed=269)
- Val sequences: 90 (rng_seed=1269)
- Test sequences: 100 (rng_seed=2269)
- Sequence length range (per segment): (16, 48)
- Identical construction to `run_comparison.py` (same seeds/sizes), so results here are directly comparable to the EMA+MLP baseline reference below.

## EMA+MLP baseline reference (alpha=0.20)

- Source: loaded_from_previous_run_comparison_json (`EMA_vs_SRN_synthetic_comparison_metrics.json`)
- overall_accuracy=0.7529, masked_accuracy_excl_ambiguous=0.7551, modulation_accuracy=0.7485, no_modulation_accuracy=0.7880, mean_lag=4.87, switch_failure_rate=0.0250

## SRN epoch x learning-rate sweep (hidden_size=48, architecture unchanged)

| epochs | lr | final_train_loss | final_val_loss | final_val_acc | test_overall_acc | test_masked_acc (excl. ambig.) | test_modulation_acc | test_no_mod_acc | test_mean_lag | test_switch_fail_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.001 | 0.8340 | 0.8559 | 0.7180 | 0.7260 | 0.7292 | 0.7199 | 0.7753 | 5.72 | 0.0125 |
| 10 | 0.003 | 0.7973 | 0.8025 | 0.7150 | 0.7270 | 0.7385 | 0.7269 | 0.7278 | 5.59 | 0.0250 |
| 25 | 0.001 | 0.6173 | 0.6622 | 0.7800 | 0.8101 | 0.8164 | 0.8010 | 0.8845 | 5.38 | 0.0375 |
| 25 | 0.003 | 0.9061 | 0.8653 | 0.6840 | 0.6602 | 0.6646 | 0.6567 | 0.6883 | 6.54 | 0.0750 |
| 50 | 0.001 | 0.4688 | 0.6489 | 0.7850 | 0.8041 | 0.8112 | 0.7953 | 0.8750 | 5.12 | 0.0500 |
| 50 | 0.003 | 0.6248 | 0.7128 | 0.7580 | 0.7600 | 0.7619 | 0.7593 | 0.7658 | 4.37 | 0.0125 |

## Best SRN condition

- epochs=25, lr=0.001 (selected by highest test overall_accuracy, tie-broken by lower mean modulation lag, then lower switch_failure_rate)
- test overall_accuracy=0.8101, mean_lag=5.38, switch_failure_rate=0.0375

## Answers

**(1) Epoch sensitivity:** across the swept learning rates, mean test overall_accuracy improves going from 10 to 50 epochs (0.7265 -> 0.7820, delta=0.0555). **(2) Vs. baseline:** the best swept SRN condition (epochs=25, lr=0.001) reaches test overall_accuracy=0.8101 against the EMA+MLP baseline's 0.7529 -- beats it by 0.0573. **(3) Main weakness:** comparing the best SRN condition to the EMA+MLP baseline, the largest relative gap is in **modulation_lag** (accuracy gap=-0.0573, normalized lag gap=0.1036, switch_failure_rate gap=0.0125). **(4) Recommended next step:** the best SRN condition is close enough to the baseline that the next reasonable step is plotting the best SRN vs. EMA (probability tracking + Circle-of-Fifths trajectories) to inspect qualitative behavior, rather than further retuning training.

## Scope note

This diagnostic only sweeps epochs and learning rate at a fixed `hidden_size=48` and unchanged Elman architecture. It does not sweep hidden size, does not add plots (`plotting_comparison.py`), and does not evaluate `Twinkle.mid`/`Twinkle 12.mid` (still synthetic-only, same as `run_comparison.py`).
