"""plotting_comparison.py

Synthetic-only Phase 1 visual comparison: EMA+MLP baseline (alpha=0.20) vs.
the best Elman SRN condition found by diagnose_srn_training.py
(epochs=25, lr=1e-3, hidden_size=48), both consuming the same 24-dim
one-hot chord/triad input.

This module trains both models fresh (deterministically, seed=269) so it
can plot per-timestep probability trajectories and Circle-of-Fifths walks
on specific representative synthetic sequences -- data that isn't stored in
the earlier metrics-only JSON outputs. The bar-chart metric summary (plot
C), by contrast, reads directly from the already-computed, already-verified
JSON outputs of run_comparison.py and diagnose_srn_training.py rather than
recomputing test-set metrics here.

Explicitly out of scope (per workspace guardrails, see ../STATUS.md):
- No MIDI or raw chroma input -- Twinkle.mid / Twinkle 12.mid are not
  touched here, and the true-Twinkle baseline figures are not regenerated.
- No raw-chroma SRN.
- No changes to shared_music_defs.py, sequence_dataset.py, mlp_baseline.py,
  srn_model.py, run_comparison.py, or diagnose_srn_training.py -- those are
  only imported from.

Run from either this directory or the workspace root:
    python plotting_comparison.py
    python 04_Recurrent_Implementation/plotting_comparison.py
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe; this script only saves PNGs, never shows interactively
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_FIGURES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "05_Figures_Results"))

from shared_music_defs import key_index, decode_key, FIFTH_POS
from sequence_dataset import make_modulation_sequence, make_no_modulation_sequence
from mlp_baseline import set_seed, DEFAULT_SEED, get_device, train_default_mlp, sequence_key_tracking
from srn_model import ElmanKeySRN, train_srn, predict_sequence_probs
import run_comparison as rc  # reused, not modified: build_datasets, stdout-capture helper

SEED = DEFAULT_SEED  # 269

PRIMARY_ALPHA = 0.20
SRN_EPOCHS = 25
SRN_LR = 1e-3
SRN_HIDDEN_SIZE = 48

EMA_METRICS_JSON = os.path.join(_FIGURES_DIR, "EMA_vs_SRN_synthetic_comparison_metrics.json")
SRN_DIAG_METRICS_JSON = os.path.join(_FIGURES_DIR, "SRN_training_diagnostic_metrics.json")

PROB_TRACKING_PNG = os.path.join(_FIGURES_DIR, "EMA_vs_SRN_C_to_G_Probability_Tracking.png")
FIFTHS_WALK_PNG = os.path.join(_FIGURES_DIR, "EMA_vs_SRN_C_to_G_Circle_of_Fifths.png")
METRIC_SUMMARY_PNG = os.path.join(_FIGURES_DIR, "EMA_vs_SRN_Synthetic_Metric_Summary.png")
REPORT_MD = os.path.join(_FIGURES_DIR, "EMA_vs_SRN_visual_comparison_report.md")


# ---------------------------------------------------------------------------
# Model training (deterministic, same 24-dim chord/triad input as always)
# ---------------------------------------------------------------------------

def train_models():
    device = get_device()

    print("Building synthetic train/val sequences (same construction as run_comparison.py)...")
    train_sequences, val_sequences, _test_sequences, dataset_info = rc.build_datasets()

    print("\nTraining EMA+MLP baseline...")
    (model_mlp, mlp_device), _mlp_log = rc._run_capturing_stdout(
        train_default_mlp, seed=SEED, n_samples=rc.MLP_N_SAMPLES, epochs=rc.MLP_EPOCHS, verbose=True
    )

    print(f"\nTraining Elman SRN (epochs={SRN_EPOCHS}, lr={SRN_LR}, hidden_size={SRN_HIDDEN_SIZE})...")
    set_seed(SEED)
    model_srn = ElmanKeySRN(input_size=24, hidden_size=SRN_HIDDEN_SIZE, output_size=24).to(device)
    model_srn, _srn_log = rc._run_capturing_stdout(
        train_srn,
        model_srn,
        train_sequences,
        val_sequences=val_sequences,
        epochs=SRN_EPOCHS,
        lr=SRN_LR,
        seed=SEED,
        device=device,
        verbose=True,
    )

    return model_mlp, mlp_device, model_srn, device, dataset_info


# ---------------------------------------------------------------------------
# Representative sequences
# ---------------------------------------------------------------------------

def build_representative_sequences():
    # Dedicated rng, offset well clear of the dataset-construction seeds
    # (269, 1269, 2269) used by run_comparison.build_datasets(), so this
    # doesn't silently reuse/collide with that sampling stream.
    plot_rng = np.random.default_rng(SEED + 9000)

    key_c_maj = key_index(0, "maj")
    key_g_maj = key_index(7, "maj")
    cg_seq = make_modulation_sequence(key_c_maj, key_g_maj, len_a=12, len_b=12, rng=plot_rng)

    no_mod_seq = make_no_modulation_sequence((0, "maj"), length=24, rng=plot_rng)

    # Optional distant-key sequence (C major -> F# major, fifth_distance=6):
    # generated for interpretation context only, not plotted as its own
    # figure, per "keep outputs limited."
    key_fsharp_maj = key_index(6, "maj")
    distant_seq = make_modulation_sequence(key_c_maj, key_fsharp_maj, len_a=12, len_b=12, rng=plot_rng)

    return cg_seq, no_mod_seq, distant_seq


# ---------------------------------------------------------------------------
# Plot A: probability tracking, EMA vs SRN, same sequence
# ---------------------------------------------------------------------------

def select_relevant_keys(probs_ema, probs_srn, must_include, top_k=4):
    combined_avg = (probs_ema.mean(axis=0) + probs_srn.mean(axis=0)) / 2.0
    top_keys = list(np.argsort(-combined_avg)[:top_k])
    for k in must_include:
        if k not in top_keys:
            top_keys.append(k)
    return sorted(set(top_keys))


def plot_probability_tracking(cg_seq, probs_ema, probs_srn, out_path):
    key_c_maj = key_index(0, "maj")
    key_g_maj = key_index(7, "maj")
    pivot_idx = cg_seq["pivot_idx"]

    keys_to_plot = select_relevant_keys(probs_ema, probs_srn, must_include=[key_c_maj, key_g_maj], top_k=4)
    colors = plt.cm.tab10(np.linspace(0, 1, len(keys_to_plot)))
    color_by_key = {k: colors[i] for i, k in enumerate(keys_to_plot)}

    T = probs_ema.shape[0]
    fig, ax = plt.subplots(figsize=(11, 5))

    for k in keys_to_plot:
        name, mode = decode_key(k)
        label = f"{name} {mode}"
        ax.plot(range(T), probs_ema[:, k], color=color_by_key[k], linestyle="-", linewidth=2, label=f"EMA: {label}")
        ax.plot(range(T), probs_srn[:, k], color=color_by_key[k], linestyle="--", linewidth=2, label=f"SRN: {label}")

    ax.axvline(pivot_idx, color="black", linestyle=":", linewidth=1.5, label=f"pivot (t={pivot_idx})")

    ax.set_title(
        f"EMA (alpha={PRIMARY_ALPHA}) vs. SRN (epochs={SRN_EPOCHS}, lr={SRN_LR}) — "
        f"C major → G major synthetic sequence\nP(key) over time, solid=EMA / dashed=SRN"
    )
    ax.set_xlabel("Chord index (time)")
    ax.set_ylabel("P(key)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot B: Circle of Fifths walks, side by side
#
# Reuses the same FIFTH_POS geometry / major-outer-minor-inner radius
# convention as the notebook's plot_fifths_walk (see mlp_baseline.py's
# docstring and shared_music_defs.FIFTH_POS), adapted to draw onto an
# existing polar Axes so EMA and SRN can sit side by side in one figure.
# ---------------------------------------------------------------------------

FIFTHS_LABEL_NAMES = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]


def _draw_fifths_walk(ax, probs_t, title):
    best_keys = np.argmax(probs_t, axis=1)

    angles, radii = [], []
    for k in best_keys:
        tonic = k % 12
        mode = "maj" if k < 12 else "min"
        pos = FIFTH_POS[tonic]
        angle = pos * (2 * np.pi / 12)
        radius = 1.0 if mode == "maj" else 0.75
        angles.append(angle)
        radii.append(radius)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(360)

    ax.plot(angles, radii, marker="o", markersize=5, linestyle="-", alpha=0.6, color="royalblue")
    ax.scatter(angles[0], radii[0], color="green", s=100, label="Start", zorder=5)
    ax.scatter(angles[-1], radii[-1], facecolors="none", edgecolors="red", linewidth=3, s=180, label="End", zorder=6)

    label_angles = [i * (2 * np.pi / 12) for i in range(12)]
    ax.set_xticks(label_angles)
    ax.set_xticklabels(FIFTHS_LABEL_NAMES, fontsize=9)
    ax.set_yticks([0.75, 1.0])
    ax.set_yticklabels(["Minor", "Major"], color="grey", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(True)
    ax.set_title(title, pad=20, fontsize=11)


def plot_fifths_walk_comparison(probs_ema, probs_srn, out_path):
    fig = plt.figure(figsize=(13, 6.5))
    ax_ema = fig.add_subplot(1, 2, 1, projection="polar")
    ax_srn = fig.add_subplot(1, 2, 2, projection="polar")

    _draw_fifths_walk(ax_ema, probs_ema, f"EMA (alpha={PRIMARY_ALPHA})\nCircle of Fifths Walk")
    _draw_fifths_walk(ax_srn, probs_srn, f"SRN (epochs={SRN_EPOCHS}, lr={SRN_LR})\nCircle of Fifths Walk")

    fig.suptitle("C major → G major synthetic sequence: most-likely-key walk", fontsize=13)
    ax_srn.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot C: metric summary bar chart, loaded from prior JSON outputs
# ---------------------------------------------------------------------------

def load_metric_summary_data():
    with open(EMA_METRICS_JSON) as f:
        ema_all = json.load(f)
    with open(SRN_DIAG_METRICS_JSON) as f:
        srn_all = json.load(f)

    ema_metrics = ema_all["baseline"]["primary_metrics"]
    ema_alpha = ema_all["baseline"]["primary_alpha"]

    best = srn_all["best_condition"]
    srn_metrics = best["test_metrics"]

    return {
        "ema": {"alpha": ema_alpha, "metrics": ema_metrics},
        "srn": {"epochs": best["epochs"], "lr": best["lr"], "hidden_size": best["hidden_size"], "metrics": srn_metrics},
    }


def plot_metric_summary(summary_data, out_path):
    ema_m = summary_data["ema"]["metrics"]
    srn_m = summary_data["srn"]["metrics"]

    panels = [
        ("Overall accuracy", ema_m["overall_accuracy"], srn_m["overall_accuracy"], (0, 1)),
        ("Modulation accuracy", ema_m["modulation_accuracy"], srn_m["modulation_accuracy"], (0, 1)),
        ("Mean modulation lag\n(timesteps)", ema_m["modulation_lag"]["mean_lag"], srn_m["modulation_lag"]["mean_lag"], None),
        ("Switch failure rate", ema_m["modulation_lag"]["switch_failure_rate"], srn_m["modulation_lag"]["switch_failure_rate"], (0, 1)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    ema_label = f"EMA (alpha={summary_data['ema']['alpha']})"
    srn_label = f"SRN (epochs={summary_data['srn']['epochs']}, lr={summary_data['srn']['lr']})"

    for ax, (title, ema_val, srn_val, ylim) in zip(axes, panels):
        bars = ax.bar([ema_label, srn_label], [ema_val, srn_val], color=["#4C72B0", "#DD8452"])
        ax.set_title(title, fontsize=10)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        for bar, val in zip(bars, [ema_val, srn_val]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("EMA+MLP vs. SRN — synthetic test-set metric summary (Phase 1)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_report_md(summary_data, generated_files):
    ema_m = summary_data["ema"]["metrics"]
    srn_m = summary_data["srn"]["metrics"]

    lines = []
    lines.append("# EMA+MLP vs. Elman SRN — Visual Comparison Report (Phase 1)")
    lines.append("")
    lines.append(
        "Synthetic-only Phase 1 visual comparison between the EMA+MLP baseline "
        f"(alpha={summary_data['ema']['alpha']}) and the best Elman SRN condition found by the training "
        f"diagnostic (epochs={summary_data['srn']['epochs']}, lr={summary_data['srn']['lr']}, "
        f"hidden_size={summary_data['srn']['hidden_size']}), both on the same 24-dim one-hot chord/triad input."
    )
    lines.append("")

    lines.append("## Files generated")
    lines.append("")
    for path in generated_files:
        lines.append(f"- `{os.path.relpath(path, os.path.join(_FIGURES_DIR, '..'))}`")
    lines.append("")

    lines.append("## Best SRN settings used")
    lines.append("")
    lines.append(
        f"- epochs={summary_data['srn']['epochs']}, lr={summary_data['srn']['lr']}, "
        f"hidden_size={summary_data['srn']['hidden_size']} (from `SRN_training_diagnostic_metrics.json` `best_condition`)"
    )
    lines.append(f"- EMA+MLP baseline: alpha={summary_data['ema']['alpha']} (from `EMA_vs_SRN_synthetic_comparison_metrics.json`)")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"- **The SRN improves overall accuracy**: overall_accuracy is {srn_m['overall_accuracy']:.4f} for the SRN "
        f"vs. {ema_m['overall_accuracy']:.4f} for EMA on the synthetic test set (matches the training diagnostic's finding)."
    )
    lines.append(
        f"- **EMA still has slightly lower lag and switch failure**: mean modulation lag is "
        f"{ema_m['modulation_lag']['mean_lag']:.2f} timesteps for EMA vs. {srn_m['modulation_lag']['mean_lag']:.2f} for the SRN, "
        f"and switch_failure_rate is {ema_m['modulation_lag']['switch_failure_rate']:.4f} for EMA vs. "
        f"{srn_m['modulation_lag']['switch_failure_rate']:.4f} for the SRN. The probability-tracking and Circle-of-Fifths "
        "plots above should be read alongside this: the SRN's higher accuracy does not (yet) come with faster/more reliable "
        "modulation switching than the hand-coded EMA baseline."
    )
    lines.append(
        "- The Circle-of-Fifths side-by-side plot shows the most-likely-key walk for both models on the same "
        "C major → G major sequence -- compare clustering/stability visually, not just the summary numbers."
    )
    lines.append(
        "- **This remains synthetic-only Phase 1.** No MIDI or chroma input was used anywhere in this script; "
        "`Twinkle.mid` and `Twinkle 12.mid` were not evaluated, and the true-Twinkle baseline figures were not regenerated."
    )
    lines.append(
        "- **No MIDI/chroma Phase 2 has started.** Per the workspace guardrails, both models here use the identical "
        "24-dim one-hot chord/triad representation; a raw-chroma end-to-end SRN remains a separate, later, explicitly-labeled phase."
    )
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "These plots use one representative C major → G major modulation sequence and one no-modulation "
        "control, freshly generated (seed=269, dedicated rng offset) rather than drawn from the held-out test "
        "set used for the metrics summary panel. The metric summary panel (`EMA_vs_SRN_Synthetic_Metric_Summary.png`) "
        "reads directly from the already-verified `EMA_vs_SRN_synthetic_comparison_metrics.json` and "
        "`SRN_training_diagnostic_metrics.json` outputs rather than recomputing test-set metrics."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_verification(png_paths, md_path, prob_arrays):
    checks = []

    for p in png_paths:
        checks.append((f"{os.path.basename(p)} exists", os.path.exists(p)))
        checks.append((f"{os.path.basename(p)} is non-empty", os.path.exists(p) and os.path.getsize(p) > 0))

    checks.append(("markdown report exists", os.path.exists(md_path)))
    checks.append(("markdown report is non-empty", os.path.exists(md_path) and os.path.getsize(md_path) > 0))

    any_nan = any(bool(np.isnan(arr).any()) for arr in prob_arrays)
    checks.append(("no NaNs in plotted probability data", not any_nan))

    print()
    print("plotting_comparison.py verification")
    print("-" * 50)
    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")
    print("-" * 50)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(_FIGURES_DIR, exist_ok=True)

    model_mlp, mlp_device, model_srn, srn_device, dataset_info = train_models()

    print("\nBuilding representative synthetic sequences...")
    cg_seq, no_mod_seq, distant_seq = build_representative_sequences()

    print("Running both models on the C major -> G major sequence...")
    probs_ema_cg = sequence_key_tracking(model_mlp, cg_seq["chord_ids"], alpha=PRIMARY_ALPHA, device=mlp_device)
    probs_srn_cg = predict_sequence_probs(model_srn, cg_seq, device=srn_device)

    print("Running both models on the no-modulation control sequence (sanity, not separately plotted)...")
    probs_ema_nomod = sequence_key_tracking(model_mlp, no_mod_seq["chord_ids"], alpha=PRIMARY_ALPHA, device=mlp_device)
    probs_srn_nomod = predict_sequence_probs(model_srn, no_mod_seq, device=srn_device)

    print("Running both models on the distant-key (C -> F#) sequence (context only, not separately plotted)...")
    probs_ema_distant = sequence_key_tracking(model_mlp, distant_seq["chord_ids"], alpha=PRIMARY_ALPHA, device=mlp_device)
    probs_srn_distant = predict_sequence_probs(model_srn, distant_seq, device=srn_device)

    print(f"\nPlot A: probability tracking -> {PROB_TRACKING_PNG}")
    plot_probability_tracking(cg_seq, probs_ema_cg, probs_srn_cg, PROB_TRACKING_PNG)

    print(f"Plot B: Circle of Fifths walk -> {FIFTHS_WALK_PNG}")
    plot_fifths_walk_comparison(probs_ema_cg, probs_srn_cg, FIFTHS_WALK_PNG)

    print(f"Plot C: metric summary -> {METRIC_SUMMARY_PNG}")
    summary_data = load_metric_summary_data()
    plot_metric_summary(summary_data, METRIC_SUMMARY_PNG)

    print(f"\nWriting markdown report -> {REPORT_MD}")
    generated_files = [PROB_TRACKING_PNG, FIFTHS_WALK_PNG, METRIC_SUMMARY_PNG]
    report_md = build_report_md(summary_data, generated_files)
    with open(REPORT_MD, "w") as f:
        f.write(report_md)

    run_verification(
        generated_files,
        REPORT_MD,
        prob_arrays=[probs_ema_cg, probs_srn_cg, probs_ema_nomod, probs_srn_nomod, probs_ema_distant, probs_srn_distant],
    )


if __name__ == "__main__":
    main()
