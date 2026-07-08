"""run_midi_phase15_evaluation.py

Phase 1.5B: evaluate the EMA+MLP baseline (alpha=0.20) and the best Elman
SRN condition (epochs=25, lr=1e-3, hidden_size=48) on the MIDI-derived
chord-id sequences extracted in Phase 1.5A (midi_chord_extraction.py):

    03_MIDI_Data/derived_chord_sequences/Twinkle_mid_chord_ids.npy
    03_MIDI_Data/derived_chord_sequences/Twinkle_12_mid_chord_ids.npy

This is a **data-source change only** relative to Phase 1: both models
still consume the identical 24-dim one-hot chord/triad representation used
throughout this workspace. Raw chroma is never fed to either model here.

Real MIDI has no clean per-timestep key ground truth, so this script does
NOT compute synthetic-style "accuracy" against a label array. Instead it
computes descriptive metrics (confidence, entropy, predicted-key
diversity, Circle-of-Fifths jump statistics) that characterize each
model's *stability and tonal-trajectory behavior* under real,
noisy, template-matched MIDI-derived chord sequences.

Reuses (imports only, does not modify): mlp_baseline.py, srn_model.py,
run_comparison.py (dataset construction + stdout-capture helper),
plotting_comparison.py (select_relevant_keys, _draw_fifths_walk),
shared_music_defs.py (key_index, decode_key, fifth_distance, key_tonic_pc).

Run from either this directory or the workspace root:
    python run_midi_phase15_evaluation.py
    python 04_Recurrent_Implementation/run_midi_phase15_evaluation.py
"""

import json
import math
import os
import sys
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_MIDI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "03_MIDI_Data"))
_DERIVED_DIR = os.path.join(_MIDI_DIR, "derived_chord_sequences")
_FIGURES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "05_Figures_Results"))

from shared_music_defs import key_index, decode_key, fifth_distance, key_tonic_pc
from mlp_baseline import set_seed, DEFAULT_SEED, get_device, train_default_mlp, sequence_key_tracking
from srn_model import ElmanKeySRN, train_srn, predict_sequence_probs
import run_comparison as rc  # reused, not modified: build_datasets, stdout-capture helper
from plotting_comparison import select_relevant_keys, _draw_fifths_walk  # reused, not modified

SEED = DEFAULT_SEED  # 269
PRIMARY_ALPHA = 0.20
SRN_EPOCHS = 25
SRN_LR = 1e-3
SRN_HIDDEN_SIZE = 48

TWINKLE_NPY = os.path.join(_DERIVED_DIR, "Twinkle_mid_chord_ids.npy")
TWINKLE12_NPY = os.path.join(_DERIVED_DIR, "Twinkle_12_mid_chord_ids.npy")

OUT = {
    "twinkle_prob_png": os.path.join(_FIGURES_DIR, "PHASE1_5B_Twinkle_EMA_vs_SRN_Probability_Tracking.png"),
    "twinkle_fifths_png": os.path.join(_FIGURES_DIR, "PHASE1_5B_Twinkle_EMA_vs_SRN_Circle_of_Fifths.png"),
    "twinkle12_prob_png": os.path.join(_FIGURES_DIR, "PHASE1_5B_Twinkle12_EMA_vs_SRN_Probability_Tracking.png"),
    "twinkle12_fifths_png": os.path.join(_FIGURES_DIR, "PHASE1_5B_Twinkle12_EMA_vs_SRN_Circle_of_Fifths.png"),
    "metrics_json": os.path.join(_FIGURES_DIR, "PHASE1_5B_MIDI_EMA_vs_SRN_metrics.json"),
    "report_md": os.path.join(_FIGURES_DIR, "PHASE1_5B_MIDI_EMA_vs_SRN_report.md"),
}


# ---------------------------------------------------------------------------
# Model training (identical settings to Phase 1 / the SRN diagnostic best condition)
# ---------------------------------------------------------------------------

def train_models():
    device = get_device()

    print("Building synthetic train/val sequences (same construction as run_comparison.py)...")
    train_sequences, val_sequences, _test_sequences, _dataset_info = rc.build_datasets()

    print("\nTraining EMA+MLP baseline...")
    (model_mlp, mlp_device), mlp_log = rc._run_capturing_stdout(
        train_default_mlp, seed=SEED, n_samples=rc.MLP_N_SAMPLES, epochs=rc.MLP_EPOCHS, verbose=True
    )
    mlp_final = rc._parse_last_epoch_line(mlp_log)

    print(f"\nTraining Elman SRN (epochs={SRN_EPOCHS}, lr={SRN_LR}, hidden_size={SRN_HIDDEN_SIZE})...")
    set_seed(SEED)
    model_srn = ElmanKeySRN(input_size=24, hidden_size=SRN_HIDDEN_SIZE, output_size=24).to(device)
    model_srn, srn_log = rc._run_capturing_stdout(
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
    srn_final = rc._parse_last_epoch_line(srn_log)

    return model_mlp, mlp_device, model_srn, device, mlp_final, srn_final


# ---------------------------------------------------------------------------
# Descriptive metrics (NOT synthetic-style accuracy -- no ground truth here)
# ---------------------------------------------------------------------------

def _to_native(x):
    if isinstance(x, dict):
        return {k: _to_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_native(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _to_native(x.tolist())
    return x


def compute_descriptive_metrics(probs):
    """
    probs: (T, 24) softmax probability array.
    Returns a dict of descriptive (non-accuracy) metrics characterizing
    confidence, key diversity, and Circle-of-Fifths trajectory stability.
    """
    T = probs.shape[0]
    preds = np.argmax(probs, axis=1)

    top_key_counts = Counter(preds.tolist())
    top_keys = [
        {"key": f"{decode_key(k)[0]} {decode_key(k)[1]}", "key_id": int(k), "count": int(c), "fraction": float(c) / T}
        for k, c in top_key_counts.most_common(5)
    ]

    key_c_maj = key_index(0, "maj")
    avg_prob_c_major = float(probs[:, key_c_maj].mean())

    top1_prob = probs.max(axis=1)
    mean_confidence = float(top1_prob.mean())

    eps = 1e-12
    entropy_per_t = -np.sum(probs * np.log(probs + eps), axis=1)
    mean_entropy = float(entropy_per_t.mean())
    max_possible_entropy = float(np.log(24))  # uniform-distribution entropy over 24 keys, for scale reference

    n_unique_predicted_keys = int(len(set(preds.tolist())))

    jumps = [fifth_distance(key_tonic_pc(preds[t]), key_tonic_pc(preds[t + 1])) for t in range(T - 1)]
    jumps = np.array(jumps, dtype=np.float64) if jumps else np.array([])
    mean_jump = float(jumps.mean()) if jumps.size > 0 else None
    max_jump = float(jumps.max()) if jumps.size > 0 else None
    large_jump_count = int(np.sum(jumps >= 3)) if jumps.size > 0 else 0
    large_jump_fraction = float(large_jump_count) / jumps.size if jumps.size > 0 else None

    return {
        "n_timesteps": T,
        "top_predicted_keys": top_keys,
        "n_unique_predicted_keys": n_unique_predicted_keys,
        "avg_prob_c_major": avg_prob_c_major,
        "mean_confidence_top1": mean_confidence,
        "mean_entropy": mean_entropy,
        "max_possible_entropy_uniform24": max_possible_entropy,
        "fifths_jump_stats": {
            "mean_jump": mean_jump,
            "max_jump": max_jump,
            "large_jump_count": large_jump_count,
            "large_jump_fraction": large_jump_fraction,
            "large_jump_threshold": 3,
        },
    }


# ---------------------------------------------------------------------------
# Plot A: probability tracking (no pivot line -- no ground-truth pivot for real MIDI)
# ---------------------------------------------------------------------------

def plot_midi_probability_tracking(probs_ema, probs_srn, must_include_keys, top_k, title, out_path):
    keys_to_plot = select_relevant_keys(probs_ema, probs_srn, must_include=must_include_keys, top_k=top_k)
    colors = plt.cm.tab10(np.linspace(0, 1, len(keys_to_plot)))
    color_by_key = {k: colors[i] for i, k in enumerate(keys_to_plot)}

    T = probs_ema.shape[0]
    fig, ax = plt.subplots(figsize=(12, 5))

    for k in keys_to_plot:
        name, mode = decode_key(k)
        label = f"{name} {mode}"
        ax.plot(range(T), probs_ema[:, k], color=color_by_key[k], linestyle="-", linewidth=1.6, label=f"EMA: {label}")
        ax.plot(range(T), probs_srn[:, k], color=color_by_key[k], linestyle="--", linewidth=1.6, label=f"SRN: {label}")

    ax.set_title(title)
    ax.set_xlabel("Chroma window index (time)")
    ax.set_ylabel("P(key)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return keys_to_plot


# ---------------------------------------------------------------------------
# Plot B: Circle of Fifths, side by side (reuses plotting_comparison._draw_fifths_walk)
# ---------------------------------------------------------------------------

def plot_midi_fifths_walk(probs_ema, probs_srn, title_ema, title_srn, suptitle, out_path):
    fig = plt.figure(figsize=(13, 6.5))
    ax_ema = fig.add_subplot(1, 2, 1, projection="polar")
    ax_srn = fig.add_subplot(1, 2, 2, projection="polar")

    _draw_fifths_walk(ax_ema, probs_ema, title_ema)
    _draw_fifths_walk(ax_srn, probs_srn, title_srn)

    fig.suptitle(suptitle, fontsize=13)
    ax_srn.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt(x, digits=4):
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _metrics_lines(name, m):
    lines = [f"**{name}**"]
    lines.append(f"- n_timesteps: {m['n_timesteps']}")
    lines.append(f"- n_unique_predicted_keys: {m['n_unique_predicted_keys']}")
    lines.append(f"- avg_prob_c_major: {_fmt(m['avg_prob_c_major'])}")
    lines.append(f"- mean_confidence (top-1 prob): {_fmt(m['mean_confidence_top1'])}")
    lines.append(f"- mean_entropy: {_fmt(m['mean_entropy'])} (uniform-24 reference: {_fmt(m['max_possible_entropy_uniform24'])})")
    fj = m["fifths_jump_stats"]
    lines.append(
        f"- fifths jump: mean={_fmt(fj['mean_jump'], 2)}, max={_fmt(fj['max_jump'], 2)}, "
        f"large jumps (>=3): {fj['large_jump_count']} ({_fmt(fj['large_jump_fraction'])} of transitions)"
    )
    lines.append("- Top predicted keys: " + ", ".join(f"{tk['key']} ({tk['fraction']:.1%})" for tk in m["top_predicted_keys"]))
    return lines


def build_report_md(results):
    lines = []
    lines.append("# Phase 1.5B — MIDI-Derived Chord-ID Evaluation Report")
    lines.append("")
    lines.append(
        "Evaluates the EMA+MLP baseline (alpha=0.20) and the best Elman SRN condition (epochs=25, lr=1e-3, "
        "hidden_size=48) on the MIDI-derived chord-id sequences extracted in Phase 1.5A "
        "(`midi_chord_extraction.py`). **This is a data-source change only** -- both models still consume "
        "the identical 24-dim one-hot chord/triad representation used throughout Phase 1; raw chroma is "
        "never fed to either model."
    )
    lines.append("")
    lines.append(
        "**This is not an accuracy test.** Real MIDI has no clean per-timestep key ground truth, so no "
        "synthetic-style accuracy is reported here. Instead, the metrics below describe each model's "
        "confidence, key diversity, and Circle-of-Fifths trajectory *stability* under real, noisy, "
        "template-matched chord sequences."
    )
    lines.append("")

    lines.append("## Model settings")
    lines.append("")
    lines.append(f"- EMA+MLP: alpha={PRIMARY_ALPHA}, MLP training seed={SEED}, epochs={rc.MLP_EPOCHS}, "
                  f"final val loss={_fmt(results['ema_settings']['mlp_training'].get('val_loss'))}, "
                  f"final val acc={_fmt(results['ema_settings']['mlp_training'].get('val_acc'))}")
    lines.append(f"- SRN: epochs={SRN_EPOCHS}, lr={SRN_LR}, hidden_size={SRN_HIDDEN_SIZE}, seed={SEED}, "
                  f"final val loss={_fmt(results['srn_settings']['training'].get('val_loss'))}, "
                  f"final val acc={_fmt(results['srn_settings']['training'].get('val_acc'))}")
    lines.append("")

    for section_key, title in [("twinkle_mid", "Twinkle.mid"), ("twinkle_12_mid", "Twinkle 12.mid")]:
        sec = results[section_key]
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- n_timesteps: {sec['n_timesteps']}")
        lines.append(f"- Plotted keys (probability tracking): {', '.join(sec['plotted_keys'])}")
        lines.append("")
        lines.append("### EMA+MLP")
        lines.append("")
        lines.extend(_metrics_lines("EMA", sec["ema"]))
        lines.append("")
        lines.append("### SRN")
        lines.append("")
        lines.extend(_metrics_lines("SRN", sec["srn"]))
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(build_interpretation(results))
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This is Phase 1.5B: real MIDI-derived chord-id sequences, still using the 24-dim one-hot "
        "chord/triad representation (no raw chroma as model input). **Raw-chroma Phase 2 has not started.** "
        "Any next step (a raw-chroma end-to-end SRN, or hidden-state PCA visualization) should be reviewed "
        "and approved separately before being implemented."
    )
    lines.append("")

    return "\n".join(lines)


def build_interpretation(results):
    parts = []

    tw = results["twinkle_mid"]
    ema_tw, srn_tw = tw["ema"], tw["srn"]

    c_region_keys = {key_index(0, "maj"), key_index(5, "maj"), key_index(7, "maj"), key_index(2, "maj")}  # C, F, G, D major

    def frac_in_region(m):
        return sum(tk["fraction"] for tk in m["top_predicted_keys"] if tk["key_id"] in c_region_keys)

    ema_region_frac = frac_in_region(ema_tw)
    srn_region_frac = frac_in_region(srn_tw)
    parts.append(
        f"**Twinkle.mid:** among each model's top-5 predicted keys, the fraction landing in the "
        f"C/F/G/D-major neighborhood is {ema_region_frac:.1%} for EMA and {srn_region_frac:.1%} for the SRN "
        f"-- {'both models' if ema_region_frac > 0.5 and srn_region_frac > 0.5 else 'the models differ in whether they'} "
        f"stay clustered around the expected tonic neighborhood for this simple, low-modulation piece."
    )

    ema_conf, srn_conf = ema_tw["mean_confidence_top1"], srn_tw["mean_confidence_top1"]
    more_confident = "SRN" if srn_conf > ema_conf else "EMA"
    parts.append(
        f"Mean top-1 confidence on Twinkle.mid: EMA={ema_conf:.3f}, SRN={srn_conf:.3f} -- the {more_confident} "
        f"is more confident on average here; combined with entropy (EMA={ema_tw['mean_entropy']:.3f}, "
        f"SRN={srn_tw['mean_entropy']:.3f}, vs. a uniform-24 reference of {ema_tw['max_possible_entropy_uniform24']:.3f}), "
        f"{'the SRN appears more overconfident rather than simply cleaner' if srn_conf > ema_conf + 0.1 else 'the SRN does not show dramatic overconfidence relative to EMA'} on this sequence."
    )

    tw12 = results["twinkle_12_mid"]
    ema_12, srn_12 = tw12["ema"], tw12["srn"]
    ema_jump = ema_12["fifths_jump_stats"]["mean_jump"]
    srn_jump = srn_12["fifths_jump_stats"]["mean_jump"]
    ema_large = ema_12["fifths_jump_stats"]["large_jump_fraction"]
    srn_large = srn_12["fifths_jump_stats"]["large_jump_fraction"]

    if srn_jump is not None and ema_jump is not None:
        direction = "reduces" if srn_jump < ema_jump else "does not reduce (and may increase)"
        parts.append(
            f"**Twinkle 12.mid:** mean Circle-of-Fifths jump distance is {ema_jump:.2f} for EMA vs. {srn_jump:.2f} "
            f"for the SRN, and the fraction of large jumps (distance >= 3) is {_fmt(ema_large)} for EMA vs. "
            f"{_fmt(srn_large)} for the SRN. The SRN {direction} the 'spiderweb'-like instability described for "
            "this piece in the historical mislabeled-figure notes, on this specific run."
        )
    parts.append(
        "Whether the SRN 'improves' Twinkle 12.mid depends on which axis is prioritized: if it reduces jump "
        "frequency/magnitude, that supports smoother tonal tracking; if predicted keys still concentrate "
        "heavily on a narrow set (see top predicted keys above) despite the piece's real embedded modulations, "
        "that is consistent with the same triadic-forcing / ornamentation-collapse limitation the COGS 202 "
        "paper describes for this piece, independent of which temporal-memory mechanism is used."
    )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _scan_for_nan(obj, path="root"):
    bad = []
    if isinstance(obj, float) and math.isnan(obj):
        bad.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad.extend(_scan_for_nan(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad.extend(_scan_for_nan(v, f"{path}[{i}]"))
    return bad


def run_verification(results, png_paths, json_path, md_path, prob_arrays, npy_paths, npy_hashes_before):
    checks = []

    for p in png_paths:
        checks.append((f"{os.path.basename(p)} exists", os.path.exists(p)))
        checks.append((f"{os.path.basename(p)} is non-empty", os.path.exists(p) and os.path.getsize(p) > 0))

    checks.append(("metrics JSON exists", os.path.exists(json_path)))
    checks.append(("report MD exists", os.path.exists(md_path)))

    nan_paths = _scan_for_nan(results)
    checks.append(("no NaNs in metrics", len(nan_paths) == 0))

    any_prob_nan = any(bool(np.isnan(arr).any()) for arr in prob_arrays)
    checks.append(("no NaNs in probability arrays", not any_prob_nan))

    for npy_path in npy_paths:
        after = np.load(npy_path)
        before = npy_hashes_before[npy_path]
        checks.append((f"{os.path.basename(npy_path)} unmodified", np.array_equal(before, after)))

    print()
    print("run_midi_phase15_evaluation.py verification")
    print("-" * 50)
    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")
    print("-" * 50)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    if nan_paths:
        print("NaN found at:", nan_paths)

    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(_FIGURES_DIR, exist_ok=True)

    # snapshot the Phase 1.5A .npy inputs before any processing, to verify
    # afterward that this script never modifies them.
    npy_paths = [TWINKLE_NPY, TWINKLE12_NPY]
    npy_before = {p: np.load(p).copy() for p in npy_paths}

    twinkle_chord_ids = np.load(TWINKLE_NPY)
    twinkle12_chord_ids = np.load(TWINKLE12_NPY)
    print(f"Loaded Twinkle.mid chord ids: {len(twinkle_chord_ids)} timesteps")
    print(f"Loaded Twinkle 12.mid chord ids: {len(twinkle12_chord_ids)} timesteps")

    model_mlp, mlp_device, model_srn, srn_device, mlp_final, srn_final = train_models()

    print("\nRunning both models on Twinkle.mid...")
    probs_ema_tw = sequence_key_tracking(model_mlp, twinkle_chord_ids, alpha=PRIMARY_ALPHA, device=mlp_device)
    probs_srn_tw = predict_sequence_probs(model_srn, twinkle_chord_ids, device=srn_device)

    print("Running both models on Twinkle 12.mid...")
    probs_ema_tw12 = sequence_key_tracking(model_mlp, twinkle12_chord_ids, alpha=PRIMARY_ALPHA, device=mlp_device)
    probs_srn_tw12 = predict_sequence_probs(model_srn, twinkle12_chord_ids, device=srn_device)

    key_c_maj = key_index(0, "maj")
    key_f_maj = key_index(5, "maj")
    key_g_maj = key_index(7, "maj")

    print(f"\nPlot: Twinkle.mid probability tracking -> {OUT['twinkle_prob_png']}")
    tw_plotted_keys_ids = plot_midi_probability_tracking(
        probs_ema_tw, probs_srn_tw,
        must_include_keys=[key_c_maj, key_f_maj, key_g_maj],
        top_k=4,
        title=(
            f"Phase 1.5B — Twinkle.mid: EMA (alpha={PRIMARY_ALPHA}) vs. SRN (epochs={SRN_EPOCHS}, lr={SRN_LR})\n"
            "P(key) over time, solid=EMA / dashed=SRN"
        ),
        out_path=OUT["twinkle_prob_png"],
    )

    print(f"Plot: Twinkle.mid Circle of Fifths -> {OUT['twinkle_fifths_png']}")
    plot_midi_fifths_walk(
        probs_ema_tw, probs_srn_tw,
        title_ema=f"EMA (alpha={PRIMARY_ALPHA})\nCircle of Fifths Walk",
        title_srn=f"SRN (epochs={SRN_EPOCHS}, lr={SRN_LR})\nCircle of Fifths Walk",
        suptitle="Phase 1.5B — Twinkle.mid: most-likely-key walk",
        out_path=OUT["twinkle_fifths_png"],
    )

    print(f"\nPlot: Twinkle 12.mid probability tracking -> {OUT['twinkle12_prob_png']}")
    tw12_plotted_keys_ids = plot_midi_probability_tracking(
        probs_ema_tw12, probs_srn_tw12,
        must_include_keys=[],
        top_k=4,
        title=(
            f"Phase 1.5B — Twinkle 12.mid: EMA (alpha={PRIMARY_ALPHA}) vs. SRN (epochs={SRN_EPOCHS}, lr={SRN_LR})\n"
            "P(key) over time, solid=EMA / dashed=SRN"
        ),
        out_path=OUT["twinkle12_prob_png"],
    )

    print(f"Plot: Twinkle 12.mid Circle of Fifths -> {OUT['twinkle12_fifths_png']}")
    plot_midi_fifths_walk(
        probs_ema_tw12, probs_srn_tw12,
        title_ema=f"EMA (alpha={PRIMARY_ALPHA})\nCircle of Fifths Walk",
        title_srn=f"SRN (epochs={SRN_EPOCHS}, lr={SRN_LR})\nCircle of Fifths Walk",
        suptitle="Phase 1.5B — Twinkle 12.mid: most-likely-key walk",
        out_path=OUT["twinkle12_fifths_png"],
    )

    print("\nComputing descriptive metrics (not accuracy -- no MIDI ground truth)...")
    results = {
        "phase": "phase_1_5b_midi_evaluation",
        "seed": SEED,
        "ema_settings": {
            "alpha": PRIMARY_ALPHA,
            "mlp_training": {"seed": SEED, "epochs": rc.MLP_EPOCHS, **mlp_final},
        },
        "srn_settings": {
            "epochs": SRN_EPOCHS,
            "lr": SRN_LR,
            "hidden_size": SRN_HIDDEN_SIZE,
            "seed": SEED,
            "training": srn_final,
        },
        "twinkle_mid": {
            "n_timesteps": int(len(twinkle_chord_ids)),
            "plotted_keys": [f"{decode_key(k)[0]} {decode_key(k)[1]}" for k in tw_plotted_keys_ids],
            "ema": compute_descriptive_metrics(probs_ema_tw),
            "srn": compute_descriptive_metrics(probs_srn_tw),
        },
        "twinkle_12_mid": {
            "n_timesteps": int(len(twinkle12_chord_ids)),
            "plotted_keys": [f"{decode_key(k)[0]} {decode_key(k)[1]}" for k in tw12_plotted_keys_ids],
            "ema": compute_descriptive_metrics(probs_ema_tw12),
            "srn": compute_descriptive_metrics(probs_srn_tw12),
        },
        "notes": (
            "Phase 1.5B: MIDI-derived chord-id evaluation (data-source change only, same 24-dim "
            "one-hot chord/triad representation as Phase 1). Not an accuracy test -- no MIDI ground "
            "truth labels exist. Raw-chroma Phase 2 has not started."
        ),
    }
    results = _to_native(results)

    with open(OUT["metrics_json"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT['metrics_json']}")

    report_md = build_report_md(results)
    with open(OUT["report_md"], "w") as f:
        f.write(report_md)
    print(f"Wrote {OUT['report_md']}")

    png_paths = [OUT["twinkle_prob_png"], OUT["twinkle_fifths_png"], OUT["twinkle12_prob_png"], OUT["twinkle12_fifths_png"]]
    run_verification(
        results, png_paths, OUT["metrics_json"], OUT["report_md"],
        prob_arrays=[probs_ema_tw, probs_srn_tw, probs_ema_tw12, probs_srn_tw12],
        npy_paths=npy_paths, npy_hashes_before=npy_before,
    )


if __name__ == "__main__":
    main()
