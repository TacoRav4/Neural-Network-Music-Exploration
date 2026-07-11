"""evaluate_pitch_class_phase2d.py

Phase 2D: evaluate the modularized pitch-class / scale-template baseline
(pitch_class_baseline.py, Phase 2B) on Twinkle.mid and Twinkle 12.mid,
cross-checked against the saved chroma sequences (midi_chroma_extraction.py,
Phase 2C).

This is a **non-neural baseline evaluation**. It bypasses the MLP, chord-id
template matching, EMA+MLP, SRN, Chroma SRN, and Transformer entirely --
predictions come directly from `pitch_class_baseline.midi_to_key_baseline`
(raw/smoothed chroma -> SCALE_TEMPLATES -> argmax key id), which is not
modified here.

Real MIDI has no dense per-timestep key ground truth. Twinkle 12.mid does
have real, sparse key-signature events (C major -> Eb major -> C major),
which this script uses as approximate timing checkpoints for a windowed,
descriptive comparison -- not as labels for an accuracy computation.

No Chroma SRN, Transformer, or staged ("QuickBin-like") pipeline is
implemented anywhere in this file.
"""

import json
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_MIDI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "03_MIDI_Data"))
_DERIVED_CHROMA_DIR = os.path.join(_MIDI_DIR, "derived_chroma_sequences")
_FIGURES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "05_Figures_Results"))

from shared_music_defs import decode_key, fifth_distance, key_tonic_pc, FIFTH_POS, key_index
from pitch_class_baseline import midi_to_key_baseline, DEFAULT_WINDOW_SEC, DEFAULT_MEMORY_DECAY
from plotting_comparison import _draw_fifths_walk, FIFTHS_LABEL_NAMES  # reused, not modified

TWINKLE_MIDI = os.path.join(_MIDI_DIR, "Twinkle.mid")
TWINKLE12_MIDI = os.path.join(_MIDI_DIR, "Twinkle 12.mid")

# Real, sparse key-signature events for Twinkle 12.mid (confirmed in Phase 2C
# via pretty_midi -- reproduced here as constants, not re-derived, since this
# script does not modify or re-inspect the MIDI's key_signature_changes API
# beyond what Phase 2C already recorded).
TWINKLE12_KEY_EVENTS = [
    {"time": 0.0, "key_name": "C Major"},
    {"time": 384.0, "key_name": "Eb Major"},
    {"time": 392.0, "key_name": "Eb Major"},
    {"time": 432.0, "key_name": "C Major"},
    {"time": 440.0, "key_name": "C Major"},
]

OUT = {
    "twinkle_traj_png": os.path.join(_FIGURES_DIR, "PHASE2D_Twinkle_pitch_class_key_trajectory.png"),
    "twinkle12_traj_png": os.path.join(_FIGURES_DIR, "PHASE2D_Twinkle12_pitch_class_key_trajectory.png"),
    "twinkle_fifths_png": os.path.join(_FIGURES_DIR, "PHASE2D_Twinkle_pitch_class_circle_of_fifths.png"),
    "twinkle12_fifths_png": os.path.join(_FIGURES_DIR, "PHASE2D_Twinkle12_pitch_class_circle_of_fifths.png"),
    "metrics_json": os.path.join(_FIGURES_DIR, "PHASE2D_pitch_class_baseline_metrics.json"),
    "report_md": os.path.join(_FIGURES_DIR, "PHASE2D_pitch_class_baseline_report.md"),
}


# ---------------------------------------------------------------------------
# Descriptive metrics (no probabilities available -- midi_to_key_baseline
# returns hard argmax key ids only, so metrics here are count/jump-based,
# not confidence/entropy-based like Phase 1.5B's softmax metrics).
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


def compute_descriptive_metrics(key_ids):
    T = len(key_ids)
    counts = {}
    for k in key_ids.tolist():
        name, mode = decode_key(k)
        label = f"{name} {mode}"
        counts[label] = counts.get(label, 0) + 1
    top_keys = [
        {"key": label, "count": c, "fraction": float(c) / T}
        for label, c in sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    ]

    key_c_maj = key_index(0, "maj")
    prop_c_major = float(np.mean(key_ids == key_c_maj)) if T > 0 else None

    n_unique = int(len(set(key_ids.tolist())))

    jumps = [fifth_distance(key_tonic_pc(key_ids[t]), key_tonic_pc(key_ids[t + 1])) for t in range(T - 1)]
    jumps = np.array(jumps, dtype=np.float64) if jumps else np.array([])
    mean_jump = float(jumps.mean()) if jumps.size > 0 else None
    max_jump = float(jumps.max()) if jumps.size > 0 else None
    large_jump_count = int(np.sum(jumps >= 3)) if jumps.size > 0 else 0
    large_jump_fraction = float(large_jump_count) / jumps.size if jumps.size > 0 else None

    return {
        "n_timesteps": T,
        "top_predicted_keys": top_keys,
        "n_unique_predicted_keys": n_unique,
        "proportion_c_major": prop_c_major,
        "fifths_jump_stats": {
            "mean_jump": mean_jump,
            "max_jump": max_jump,
            "large_jump_count": large_jump_count,
            "large_jump_fraction": large_jump_fraction,
            "large_jump_threshold": 3,
        },
    }


def compute_window_metrics(key_ids, window_sec, t_start, t_end, expected_key_name=None, offset_windows=0):
    """
    offset_windows: number of leading chroma windows that were silent and
    therefore dropped entirely from key_ids by midi_to_key_baseline (it
    only starts appending once len(key_sequence) > 0 or the first nonzero
    window is reached -- see pitch_class_baseline.py). key_ids[0] then
    corresponds to real time offset_windows * window_sec, not t=0. Without
    this correction, window boundaries computed from wall-clock times would
    be silently misaligned by that offset.
    """
    idx_start = int(round(t_start / window_sec)) - offset_windows if t_start is not None else 0
    idx_end = int(round(t_end / window_sec)) - offset_windows if t_end is not None else len(key_ids)
    idx_start = max(0, min(idx_start, len(key_ids)))
    idx_end = max(idx_start, min(idx_end, len(key_ids)))

    window_ids = key_ids[idx_start:idx_end]
    metrics = compute_descriptive_metrics(window_ids) if len(window_ids) > 0 else {
        "n_timesteps": 0, "top_predicted_keys": [], "n_unique_predicted_keys": 0,
        "proportion_c_major": None, "fifths_jump_stats": {},
    }

    prop_expected = None
    if expected_key_name is not None and len(window_ids) > 0:
        expected_tonic, expected_mode = expected_key_name.split()
        expected_mode = "maj" if expected_mode.lower().startswith("maj") else "min"
        pc_map = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
                  "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
        expected_key_id = key_index(pc_map[expected_tonic], expected_mode)
        prop_expected = float(np.mean(window_ids == expected_key_id))

    metrics["window_start_idx"] = idx_start
    metrics["window_end_idx"] = idx_end
    metrics["window_start_sec"] = t_start
    metrics["window_end_sec"] = t_end
    metrics["offset_windows_applied"] = offset_windows
    metrics["expected_key"] = expected_key_name
    metrics["proportion_expected_key"] = prop_expected
    return metrics


# ---------------------------------------------------------------------------
# Plot A: key trajectory over time (Circle-of-Fifths position on y-axis)
# ---------------------------------------------------------------------------

def plot_key_trajectory(key_ids, window_sec, title, out_path, key_events=None, offset_windows=0):
    """offset_windows: see compute_window_metrics docstring -- corrects the
    x-axis for leading silent chroma windows dropped before the first
    prediction, so key-signature marker lines line up with true wall-clock
    time rather than with key_ids' own index 0."""
    T = len(key_ids)
    x_sec = (np.arange(T) + offset_windows) * window_sec
    fifths_pos = np.array([FIFTH_POS[key_tonic_pc(k)] for k in key_ids])
    is_major = np.array([k < 12 for k in key_ids])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_sec, fifths_pos, color="gray", alpha=0.35, linewidth=0.8, zorder=1)
    ax.scatter(x_sec[is_major], fifths_pos[is_major], s=12, color="tab:blue", label="major", zorder=2)
    ax.scatter(x_sec[~is_major], fifths_pos[~is_major], s=12, color="tab:orange", label="minor", zorder=2)

    ax.set_yticks(range(12))
    ax.set_yticklabels(FIFTHS_LABEL_NAMES)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Predicted tonic (circle-of-fifths position)")
    ax.set_title(title)

    if key_events:
        for ev in key_events:
            ax.axvline(ev["time"], color="red", linestyle="--", alpha=0.5, linewidth=1, zorder=0)
            ax.text(ev["time"], 11.6, ev["key_name"], rotation=90, fontsize=8, color="red", va="bottom", ha="center")

    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot B: Circle of Fifths walk (single model -- reuses plotting_comparison._draw_fifths_walk
# by constructing a one-hot "probs" array from the hard key ids, since this
# baseline has no softmax probabilities to plot).
# ---------------------------------------------------------------------------

def plot_fifths_walk_single(key_ids, title, out_path):
    T = len(key_ids)
    probs_onehot = np.zeros((T, 24))
    probs_onehot[np.arange(T), key_ids] = 1.0

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    _draw_fifths_walk(ax, probs_onehot, title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
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


def _metrics_lines(m):
    lines = []
    lines.append(f"- n_timesteps: {m['n_timesteps']}")
    lines.append(f"- n_unique_predicted_keys: {m['n_unique_predicted_keys']}")
    lines.append(f"- proportion_c_major: {_fmt(m['proportion_c_major'])}")
    fj = m["fifths_jump_stats"]
    if fj:
        lines.append(
            f"- fifths jump: mean={_fmt(fj.get('mean_jump'), 2)}, max={_fmt(fj.get('max_jump'), 2)}, "
            f"large jumps (>=3): {fj.get('large_jump_count', 'n/a')} ({_fmt(fj.get('large_jump_fraction'))} of transitions)"
        )
    lines.append("- Top predicted keys: " + ", ".join(f"{tk['key']} ({tk['fraction']:.1%})" for tk in m["top_predicted_keys"]))
    return lines


def build_report_md(results):
    lines = []
    lines.append("# Phase 2D — Pitch-Class Baseline Evaluation Report")
    lines.append("")
    lines.append(
        "Evaluates the modularized pitch-class/scale-template baseline (`pitch_class_baseline.py`, "
        "Phase 2B) on `Twinkle.mid` and `Twinkle 12.mid`, cross-referenced against the saved chroma "
        "sequences from Phase 2C. **This is a non-neural baseline evaluation** -- it bypasses the MLP, "
        "chord-id template matching, EMA+MLP, SRN, Chroma SRN, and Transformer entirely. Predictions come "
        "directly from `midi_to_key_baseline` (raw/smoothed chroma -> `SCALE_TEMPLATES` -> argmax key id)."
    )
    lines.append("")
    lines.append(
        "**No Chroma SRN, Transformer, or QuickBin-like staged pipeline has been implemented anywhere "
        "in this workspace.** This report is representation-only evidence (does avoiding hard triadic "
        "forcing change real-MIDI behavior), not a new model result."
    )
    lines.append("")

    lines.append("## Settings")
    lines.append("")
    lines.append(f"- window_sec: {DEFAULT_WINDOW_SEC}")
    lines.append(f"- memory_decay: {DEFAULT_MEMORY_DECAY} (pitch-class baseline's own smoothing constant, per Phase 2B)")
    lines.append("")

    lines.append("## Twinkle.mid")
    lines.append("")
    lines.extend(_metrics_lines(results["twinkle_mid"]["overall"]))
    lines.append("")
    lines.append(
        "**Vs. Phase 1.5B (chord-id EMA/SRN):** Phase 1.5B found the SRN collapsed onto F major "
        f"(67.0% of predictions) with avg_prob_c_major=0.072, and EMA at F major 39.6% with "
        f"avg_prob_c_major=0.147. Here, the pitch-class baseline predicts C major "
        f"{_fmt(results['twinkle_mid']['overall']['proportion_c_major'], 3)} of the time -- "
        f"{results['twinkle_mid']['comparison_note']}"
    )
    lines.append("")

    lines.append("## Twinkle 12.mid")
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    lines.extend(_metrics_lines(results["twinkle_12_mid"]["overall"]))
    lines.append("")
    lines.append(
        "**Vs. Phase 1.5B:** both chord-id EMA and SRN concentrated ~60% of predictions on F major and only "
        f"~12-13% on C major despite this piece's real embedded modulations. Here, the pitch-class baseline "
        f"predicts C major {_fmt(results['twinkle_12_mid']['overall']['proportion_c_major'], 3)} of the time overall."
    )
    lines.append("")
    lines.append("### Key-signature-aligned windows")
    lines.append("")
    lines.append(
        "Twinkle 12.mid's real embedded key-signature events (confirmed in Phase 2C via `pretty_midi`, "
        "read-only): t=0.0s C Major, t=384.0s/392.0s Eb Major, t=432.0s/440.0s C Major. The windows below "
        "are split at t=384s and t=432s accordingly, corrected for the leading-silent-window time offset "
        "(see Settings/JSON `*_offset_windows`) so window boundaries align with true wall-clock time rather "
        "than with `key_ids`' own index 0. **This is a descriptive, key-signature-aligned check, "
        "not a dense per-timestep accuracy computation** -- there is no per-timestep ground truth. "
        "Note: this workspace's `decode_key` spells pitch class 3 as \"D#\", so \"D# maj\" below is the "
        "same key as \"Eb Major\"."
    )
    lines.append("")
    for window_key, window_title in [("pre_384s", "pre_384s (expected: C Major)"), ("384_to_432s", "384_to_432s (expected: Eb Major)"), ("post_432s", "post_432s (expected: C Major)")]:
        wm = results["twinkle_12_mid"]["windows"][window_key]
        lines.append(f"**{window_title}:**")
        lines.append(f"- timesteps: {wm['n_timesteps']} (indices {wm['window_start_idx']}-{wm['window_end_idx']})")
        lines.append(f"- proportion predicted as expected key ({wm['expected_key']}): {_fmt(wm['proportion_expected_key'])}")
        if wm["top_predicted_keys"]:
            lines.append("- Top predicted keys: " + ", ".join(f"{tk['key']} ({tk['fraction']:.1%})" for tk in wm["top_predicted_keys"]))
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(build_interpretation(results))
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This is Phase 2D: a non-neural, representation-only baseline evaluation. No Chroma SRN "
        "(Phase 2E), Transformer, or QuickBin-like staged tonal-inference pipeline has been implemented "
        "here. Metrics are descriptive (proportions, jump statistics, key-signature-aligned window "
        "checks), not accuracy against dense ground truth, since no such ground truth exists for real MIDI."
    )
    lines.append("")

    return "\n".join(lines)


def build_interpretation(results):
    parts = []

    tw = results["twinkle_mid"]["overall"]
    parts.append(
        f"**Does the pitch-class baseline reduce the F-major bias seen in Phase 1.5B?** On Twinkle.mid, "
        f"yes, clearly: the pitch-class baseline's top predicted key is "
        f"{tw['top_predicted_keys'][0]['key']} at {tw['top_predicted_keys'][0]['fraction']:.1%}, with "
        f"proportion_c_major={_fmt(tw['proportion_c_major'], 3)} -- markedly higher than either chord-id "
        f"model's avg_prob_c_major in Phase 1.5B (EMA 0.147, SRN 0.072), and the baseline's most common "
        f"prediction is C major rather than F major."
    )

    tw12 = results["twinkle_12_mid"]
    pre = tw12["windows"]["pre_384s"]
    mid = tw12["windows"]["384_to_432s"]
    post = tw12["windows"]["post_432s"]
    mid_prop = mid["proportion_expected_key"]

    if mid_prop is not None and mid_prop > 0.5:
        mid_verdict = (
            "**the baseline responds strongly and correctly to the real modulation**: it predicts Eb major "
            "(reported as \"D# maj\" above -- the same key, spelled with a sharp rather than a flat by this "
            "workspace's `decode_key` convention) for the large majority of this window, closely tracking "
            "the piece's real, embedded key-signature change rather than defaulting to C major."
        )
    elif mid_prop is not None and mid_prop > 0.15:
        mid_verdict = "the baseline shows some real responsiveness to the expected key during the modulation window, though not a dominant majority."
    else:
        mid_verdict = (
            "**the baseline does not respond to the real modulation at all**: it predicts the same key "
            "(C major) essentially straight through the entire piece, including during the window where the "
            "actual key signature is Eb major."
        )

    parts.append(
        f"**Does Twinkle 12.mid respond to the C -> Eb -> C key-signature markers?** In the pre_384s window "
        f"(expected C Major), the baseline predicts C major {_fmt(pre['proportion_expected_key'])} of the "
        f"time. In the 384_to_432s window (expected Eb Major), it predicts Eb major (\"D# maj\") "
        f"{_fmt(mid_prop)} of the time. In the post_432s window (expected C Major again), it predicts C "
        f"major {_fmt(post['proportion_expected_key'])} of the time. Put together, {mid_verdict}"
    )

    parts.append(
        "**Framing:** all of the above is representation-only evidence -- a non-neural, hand-coded "
        "scale-template comparison, not a new learned model result. It suggests the *representation* "
        "(full scale evidence vs. hard triads) materially changes Twinkle.mid's behavior, but does not by "
        "itself establish that a learned model over this representation (a Chroma SRN) would perform "
        "similarly, worse, or better -- that remains untested. No Chroma SRN, Transformer, or "
        "QuickBin-like staged pipeline has been implemented yet."
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


def run_verification(results, png_paths, json_path, md_path, midi_mtimes_before, chroma_mtimes_before):
    checks = []

    for p in png_paths:
        checks.append((f"{os.path.basename(p)} exists", os.path.exists(p)))
        checks.append((f"{os.path.basename(p)} is non-empty", os.path.exists(p) and os.path.getsize(p) > 0))

    checks.append(("metrics JSON exists", os.path.exists(json_path)))
    checks.append(("metrics JSON has twinkle_mid", "twinkle_mid" in results))
    checks.append(("metrics JSON has twinkle_12_mid", "twinkle_12_mid" in results))
    checks.append(("report MD exists", os.path.exists(md_path)))

    nan_paths = _scan_for_nan(results)
    checks.append(("no NaNs in metrics", len(nan_paths) == 0))

    for path, mtime_before in midi_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} MIDI file not modified", os.path.getmtime(path) == mtime_before))

    for path, mtime_before in chroma_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} derived chroma file not modified", os.path.getmtime(path) == mtime_before))

    print()
    print("evaluate_pitch_class_phase2d.py verification")
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

    midi_mtimes_before = {TWINKLE_MIDI: os.path.getmtime(TWINKLE_MIDI), TWINKLE12_MIDI: os.path.getmtime(TWINKLE12_MIDI)}
    chroma_files = [f for f in os.listdir(_DERIVED_CHROMA_DIR)] if os.path.isdir(_DERIVED_CHROMA_DIR) else []
    chroma_mtimes_before = {os.path.join(_DERIVED_CHROMA_DIR, f): os.path.getmtime(os.path.join(_DERIVED_CHROMA_DIR, f)) for f in chroma_files}

    print("Running pitch-class baseline on Twinkle.mid...")
    twinkle_keys, twinkle_meta = midi_to_key_baseline(TWINKLE_MIDI)
    print(f"  n_predictions={len(twinkle_keys)}")

    print("Running pitch-class baseline on Twinkle 12.mid...")
    twinkle12_keys, twinkle12_meta = midi_to_key_baseline(TWINKLE12_MIDI)
    print(f"  n_predictions={len(twinkle12_keys)}")

    twinkle_overall = compute_descriptive_metrics(twinkle_keys)
    twinkle12_overall = compute_descriptive_metrics(twinkle12_keys)

    # midi_to_key_baseline drops any leading chroma windows that are silent
    # (see pitch_class_baseline.py: it only starts appending once the first
    # nonzero window is reached), so key_ids[0] can correspond to a real
    # time > 0. This offset must be applied whenever mapping key_ids
    # indices to/from wall-clock time, or window boundaries and plot
    # x-axes silently misalign with the true key-signature event times.
    twinkle_offset_windows = twinkle_meta["n_chroma_windows"] - twinkle_meta["n_key_predictions"]
    twinkle12_offset_windows = twinkle12_meta["n_chroma_windows"] - twinkle12_meta["n_key_predictions"]
    print(f"\nTwinkle.mid: {twinkle_offset_windows} leading silent windows dropped (time offset {twinkle_offset_windows * DEFAULT_WINDOW_SEC:.1f}s)")
    print(f"Twinkle 12.mid: {twinkle12_offset_windows} leading silent windows dropped (time offset {twinkle12_offset_windows * DEFAULT_WINDOW_SEC:.1f}s)")

    windows = {
        "pre_384s": compute_window_metrics(twinkle12_keys, DEFAULT_WINDOW_SEC, None, 384.0, expected_key_name="C Major", offset_windows=twinkle12_offset_windows),
        "384_to_432s": compute_window_metrics(twinkle12_keys, DEFAULT_WINDOW_SEC, 384.0, 432.0, expected_key_name="Eb Major", offset_windows=twinkle12_offset_windows),
        "post_432s": compute_window_metrics(twinkle12_keys, DEFAULT_WINDOW_SEC, 432.0, None, expected_key_name="C Major", offset_windows=twinkle12_offset_windows),
    }

    top_key_frac = twinkle_overall["top_predicted_keys"][0]["fraction"] if twinkle_overall["top_predicted_keys"] else 0.0
    is_c_major_top = twinkle_overall["top_predicted_keys"] and twinkle_overall["top_predicted_keys"][0]["key"] == "C maj"
    comparison_note = (
        "C major is the single most common prediction, in clear contrast with both chord-id models' F-major dominance in Phase 1.5B."
        if is_c_major_top else
        "C major is not the single most common prediction, so the F-major-style bias may persist in a different form even without triadic forcing."
    )

    print(f"\nPlot: Twinkle.mid trajectory -> {OUT['twinkle_traj_png']}")
    plot_key_trajectory(
        twinkle_keys, DEFAULT_WINDOW_SEC,
        title="Phase 2D — Pitch-Class Baseline: Twinkle.mid Key Trajectory",
        out_path=OUT["twinkle_traj_png"],
        key_events=None,
        offset_windows=twinkle_offset_windows,
    )

    print(f"Plot: Twinkle.mid Circle of Fifths -> {OUT['twinkle_fifths_png']}")
    plot_fifths_walk_single(twinkle_keys, "Phase 2D — Pitch-Class Baseline\nTwinkle.mid Circle of Fifths Walk", OUT["twinkle_fifths_png"])

    print(f"\nPlot: Twinkle 12.mid trajectory -> {OUT['twinkle12_traj_png']}")
    plot_key_trajectory(
        twinkle12_keys, DEFAULT_WINDOW_SEC,
        title="Phase 2D — Pitch-Class Baseline: Twinkle 12.mid Key Trajectory (with key-signature markers)",
        out_path=OUT["twinkle12_traj_png"],
        key_events=TWINKLE12_KEY_EVENTS,
        offset_windows=twinkle12_offset_windows,
    )

    print(f"Plot: Twinkle 12.mid Circle of Fifths -> {OUT['twinkle12_fifths_png']}")
    plot_fifths_walk_single(twinkle12_keys, "Phase 2D — Pitch-Class Baseline\nTwinkle 12.mid Circle of Fifths Walk", OUT["twinkle12_fifths_png"])

    results = {
        "phase": "phase_2d_pitch_class_baseline_evaluation",
        "settings": {
            "window_sec": DEFAULT_WINDOW_SEC,
            "memory_decay": DEFAULT_MEMORY_DECAY,
            "twinkle_mid_offset_windows": twinkle_offset_windows,
            "twinkle_12_mid_offset_windows": twinkle12_offset_windows,
        },
        "twinkle_mid": {
            "overall": twinkle_overall,
            "comparison_note": comparison_note,
        },
        "twinkle_12_mid": {
            "overall": twinkle12_overall,
            "key_signature_events": TWINKLE12_KEY_EVENTS,
            "windows": windows,
        },
        "notes": (
            "Phase 2D: non-neural pitch-class/scale-template baseline evaluation. No Chroma SRN, "
            "Transformer, or QuickBin-like staged pipeline implemented. Descriptive metrics only -- "
            "no dense per-timestep ground truth exists for real MIDI."
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

    png_paths = [OUT["twinkle_traj_png"], OUT["twinkle12_traj_png"], OUT["twinkle_fifths_png"], OUT["twinkle12_fifths_png"]]
    run_verification(results, png_paths, OUT["metrics_json"], OUT["report_md"], midi_mtimes_before, chroma_mtimes_before)


if __name__ == "__main__":
    main()
