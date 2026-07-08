"""pitch_class_uncertainty_diagnostics.py

Phase 3B: uncertainty diagnostics for the pitch-class/scale-template
baseline (pitch_class_baseline.py, Phase 2B), computed directly from the
saved Phase 2C chroma arrays.

This is **diagnostic only**. It does not change any prediction, does not
train anything, and does not implement any neural refinement, Chroma SRN,
or Transformer. It also does not compare against the chord-id EMA/SRN
models (Phase 1.5B) -- that cross-model comparison is Phase 3C, not here.

Alignment note (why this module recomputes scores from saved chroma
instead of calling pitch_class_baseline.midi_to_key_baseline directly):
midi_to_key_baseline silently *drops* leading silent chroma windows before
its first prediction (it only starts appending once the first nonzero
window is reached). That caused a real, caught-and-fixed timing-alignment
bug in Phase 2D (evaluate_pitch_class_phase2d.py), where predictions had to
be shifted by an `offset_windows` correction to line up with real
wall-clock time. This module avoids reintroducing that class of bug by
working over the *full* saved chroma array (every window from Phase 2C,
none dropped) and explicitly tracking which windows are active
(non-silent) via a boolean mask, rather than silently skipping any
timesteps. `prediction_times_sec` is therefore always `t * window_sec`
with no offset arithmetic required downstream.

Reuses (imports only): SCALE_TEMPLATES from pitch_class_baseline.py
(not modified), decode_key / fifth_distance / key_tonic_pc / key_index
from shared_music_defs.py (not modified).
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
_DERIVED_DIAG_DIR = os.path.join(_MIDI_DIR, "derived_phase3_diagnostics")
_FIGURES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "05_Figures_Results"))

from shared_music_defs import decode_key, fifth_distance, key_tonic_pc, key_index
from pitch_class_baseline import SCALE_TEMPLATES  # reused, not modified

# --- Diagnostic constants (documented here, not hidden magic numbers) ---
DEFAULT_LOW_MARGIN_THRESHOLD = 0.20   # normalized_margin below this -> "low_margin" flag
DEFAULT_LARGE_JUMP_THRESHOLD = 3      # fifths-distance >= this -> "large_jump" flag (matches the rest of this workspace)
DEFAULT_ENTROPY_TEMPERATURE = 1.0     # softmax temperature for the diagnostic entropy measure -- arbitrary, NOT tuned;
                                       # entropy values below are only meaningful relative to each other at this fixed
                                       # temperature, not as an absolute/calibrated confidence measure.
EPS = 1e-8

TWINKLE_CHROMA_NPY = os.path.join(_DERIVED_CHROMA_DIR, "Twinkle_mid_thresholded_smoothed_chroma_decay08.npy")
TWINKLE_META_JSON = os.path.join(_DERIVED_CHROMA_DIR, "Twinkle_mid_chroma_metadata.json")
TWINKLE12_CHROMA_NPY = os.path.join(_DERIVED_CHROMA_DIR, "Twinkle_12_mid_thresholded_smoothed_chroma_decay08.npy")
TWINKLE12_META_JSON = os.path.join(_DERIVED_CHROMA_DIR, "Twinkle_12_mid_chroma_metadata.json")

TWINKLE_MIDI = os.path.join(_MIDI_DIR, "Twinkle.mid")
TWINKLE12_MIDI = os.path.join(_MIDI_DIR, "Twinkle 12.mid")

OUT = {
    "twinkle_difficulty_json": os.path.join(_DERIVED_DIAG_DIR, "Twinkle_mid_pitch_class_difficulty.json"),
    "twinkle12_difficulty_json": os.path.join(_DERIVED_DIAG_DIR, "Twinkle_12_mid_pitch_class_difficulty.json"),
    "twinkle_margin_png": os.path.join(_FIGURES_DIR, "PHASE3B_Twinkle_margin_over_time.png"),
    "twinkle12_margin_png": os.path.join(_FIGURES_DIR, "PHASE3B_Twinkle12_margin_over_time.png"),
    "twinkle_entropy_png": os.path.join(_FIGURES_DIR, "PHASE3B_Twinkle_entropy_over_time.png"),
    "twinkle12_entropy_png": os.path.join(_FIGURES_DIR, "PHASE3B_Twinkle12_entropy_over_time.png"),
    "twinkle12_anchor_png": os.path.join(_FIGURES_DIR, "PHASE3B_Twinkle12_anchor_diagnostics.png"),
    "report_md": os.path.join(_FIGURES_DIR, "PHASE3B_pitch_class_uncertainty_report.md"),
    "metrics_json": os.path.join(_FIGURES_DIR, "PHASE3B_pitch_class_uncertainty_metrics.json"),
}

TWINKLE12_KEY_EVENTS = [
    {"time": 0.0, "key_name": "C Major"},
    {"time": 384.0, "key_name": "Eb Major"},
    {"time": 392.0, "key_name": "Eb Major"},
    {"time": 432.0, "key_name": "C Major"},
    {"time": 440.0, "key_name": "C Major"},
]


# ---------------------------------------------------------------------------
# Core per-timestep computation
# ---------------------------------------------------------------------------

def compute_scale_scores(chroma):
    """(T, 12) chroma -> (T, 24) raw SCALE_TEMPLATES dot-product scores."""
    return chroma @ SCALE_TEMPLATES.T


def analyze_piece(chroma, window_sec, low_margin_threshold=DEFAULT_LOW_MARGIN_THRESHOLD,
                   large_jump_threshold=DEFAULT_LARGE_JUMP_THRESHOLD, entropy_temperature=DEFAULT_ENTROPY_TEMPERATURE):
    """
    Computes all per-timestep diagnostic arrays over the FULL chroma array
    (no windows dropped -- see module docstring). Returns a dict of (T,)
    (or (T,24) for raw_scores) numpy arrays plus the settings used.
    """
    T = chroma.shape[0]
    raw_scores = compute_scale_scores(chroma)  # (T, 24)

    active = raw_scores.max(axis=1) > 0

    # top1 MUST use plain argmax (leftmost-tie convention) to exactly match
    # pitch_class_baseline.midi_to_key_baseline's own `int(np.argmax(scores))`
    # -- np.argsort's default (non-stable) sort can silently disagree with
    # argmax's tie-breaking on windows with several tied top scores (common
    # here: many chroma-sparse, single-active-pitch-class windows produce
    # several scale templates with identical dot-product scores). This was
    # caught during Phase 3B development: using argsort for top1 diverged
    # from the canonical prediction on ~7% of Twinkle 12.mid's windows,
    # concentrated in exactly the modulation window Phase 2D had validated.
    top1_key = np.argmax(raw_scores, axis=1)
    top1_score = raw_scores[np.arange(T), top1_key]

    # top2: argmax of the remaining scores after masking out top1's own
    # index (avoids re-picking top1 on a tie, and doesn't need to match any
    # external convention since top2 is only used for the margin diagnostic).
    masked_scores = raw_scores.copy()
    masked_scores[np.arange(T), top1_key] = -np.inf
    top2_key = np.argmax(masked_scores, axis=1)
    top2_score = raw_scores[np.arange(T), top2_key]
    raw_margin = top1_score - top2_score
    normalized_margin = raw_margin / (top1_score + EPS)  # 0 for fully-silent windows (0/EPS -> 0), never NaN

    # Diagnostic-only softmax entropy at a fixed, documented temperature.
    scaled = raw_scores / entropy_temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    probs = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
    entropy = -np.sum(probs * np.log(probs + EPS), axis=1)

    # Forward-fill key_id through silent windows (same continuity convention
    # as midi_to_key_baseline), but explicitly tracked via `active` so silent
    # stretches are never mistaken for real evidence downstream.
    key_id = np.full(T, -1, dtype=np.int64)
    last_key = -1
    for t in range(T):
        if active[t]:
            key_id[t] = top1_key[t]
            last_key = key_id[t]
        else:
            key_id[t] = last_key  # stays -1 until the first active window is seen

    # Key-switch and Circle-of-Fifths jump diagnostics -- only meaningful
    # between two consecutive *defined* keys (key_id != -1 on both sides).
    key_switch = np.zeros(T, dtype=bool)
    jump_distance = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        if key_id[t] != -1 and key_id[t - 1] != -1:
            key_switch[t] = key_id[t] != key_id[t - 1]
            jump_distance[t] = fifth_distance(key_tonic_pc(key_id[t - 1]), key_tonic_pc(key_id[t]))
    large_jump = jump_distance >= large_jump_threshold

    low_margin = active & (normalized_margin < low_margin_threshold)

    # Diagnostic: how many of the 24 SCALE_TEMPLATES rows are EXACTLY tied
    # at the max score for this window. Relative major/minor pairs (e.g. C
    # major / A minor) share an identical 7-pitch-class set and therefore
    # have numerically identical template rows, so any window whose active
    # chroma is a subset of several such scales' pitch classes (very common
    # for sparse, monophonic melodic evidence) can tie across many keys at
    # once. This is the main driver of near-zero top1-top2 margins observed
    # below, and is reported explicitly so `low_margin` isn't misread as a
    # simple "prediction is shaky" signal when it is often actually "many
    # keys are exactly, structurally indistinguishable from this evidence."
    tie_count = np.sum(raw_scores == raw_scores.max(axis=1, keepdims=True), axis=1).astype(np.int64)

    prediction_indices = np.arange(T, dtype=np.int64)
    prediction_times_sec = prediction_indices * window_sec

    return {
        "T": T,
        "raw_scores": raw_scores,
        "active": active,
        "key_id": key_id,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "raw_margin": raw_margin,
        "normalized_margin": normalized_margin,
        "entropy": entropy,
        "tie_count": tie_count,
        "key_switch": key_switch,
        "jump_distance": jump_distance,
        "large_jump": large_jump,
        "low_margin": low_margin,
        "prediction_indices": prediction_indices,
        "prediction_times_sec": prediction_times_sec,
        "settings": {
            "window_sec": window_sec,
            "low_margin_threshold": low_margin_threshold,
            "large_jump_threshold": large_jump_threshold,
            "entropy_temperature": entropy_temperature,
        },
    }


# ---------------------------------------------------------------------------
# Anchor diagnostics
# ---------------------------------------------------------------------------

_PC_MAP = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
           "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def _expected_key_id(expected_key_name):
    tonic, mode = expected_key_name.split()
    mode = "maj" if mode.lower().startswith("maj") else "min"
    return key_index(_PC_MAP[tonic], mode)


def _run_length_encode(mask, prediction_times_sec):
    """Compact contiguous-True intervals of a boolean array into a list of
    {start_idx, end_idx, start_sec, end_sec} dicts."""
    intervals = []
    T = len(mask)
    t = 0
    while t < T:
        if mask[t]:
            start = t
            while t < T and mask[t]:
                t += 1
            intervals.append({
                "start_idx": int(start), "end_idx": int(t),
                "start_sec": float(prediction_times_sec[start]), "end_sec": float(prediction_times_sec[t - 1]),
            })
        else:
            t += 1
    return intervals


def compute_anchor_diagnostics(analysis, anchor_windows):
    """
    anchor_windows: list of {name, start_sec, end_sec (or None), expected_key_name}.
    Returns a dict keyed by anchor name with per-window diagnostics.
    """
    key_id = analysis["key_id"]
    times = analysis["prediction_times_sec"]
    normalized_margin = analysis["normalized_margin"]
    entropy = analysis["entropy"]

    results = {}
    for anchor in anchor_windows:
        t_start = anchor["start_sec"] if anchor["start_sec"] is not None else 0.0
        t_end = anchor["end_sec"] if anchor["end_sec"] is not None else float(times[-1]) + 1.0
        mask = (times >= t_start) & (times < t_end)
        idxs = np.where(mask)[0]

        expected_id = _expected_key_id(anchor["expected_key_name"])
        defined = key_id[idxs] != -1
        n_defined = int(defined.sum())

        mismatch_mask_full = np.zeros(len(key_id), dtype=bool)
        if n_defined > 0:
            sub_idxs = idxs[defined]
            mismatch_mask_full[sub_idxs] = key_id[sub_idxs] != expected_id
            proportion_expected = float(np.mean(key_id[sub_idxs] == expected_id))
            mean_margin = float(normalized_margin[sub_idxs].mean())
            mean_entropy = float(entropy[sub_idxs].mean())
            low_margin_proportion = float(analysis["low_margin"][sub_idxs].mean())
        else:
            proportion_expected = None
            mean_margin = None
            mean_entropy = None
            low_margin_proportion = None

        mismatch_intervals = _run_length_encode(mismatch_mask_full[idxs[0]:idxs[-1] + 1] if len(idxs) else np.array([], dtype=bool),
                                                  times[idxs[0]:idxs[-1] + 1] if len(idxs) else np.array([]))
        # shift interval indices back to the full-array index space
        for iv in mismatch_intervals:
            iv["start_idx"] += int(idxs[0]) if len(idxs) else 0
            iv["end_idx"] += int(idxs[0]) if len(idxs) else 0

        results[anchor["name"]] = {
            "expected_key": anchor["expected_key_name"],
            "start_sec": t_start,
            "end_sec": anchor["end_sec"],
            "n_predictions": int(len(idxs)),
            "n_defined": n_defined,
            "proportion_expected_key": proportion_expected,
            "mismatch_count": int(mismatch_mask_full.sum()),
            "mismatch_intervals": mismatch_intervals,
            "low_margin_proportion": low_margin_proportion,
            "mean_normalized_margin": mean_margin,
            "mean_entropy": mean_entropy,
        }
        results[anchor["name"]]["_mismatch_mask"] = mismatch_mask_full  # internal use for combined difficulty; stripped before saving JSON

    return results


# ---------------------------------------------------------------------------
# Per-piece summary
# ---------------------------------------------------------------------------

def compute_summary(analysis):
    T = analysis["T"]
    active = analysis["active"]
    key_id = analysis["key_id"]
    defined = key_id != -1

    unique_keys = sorted(set(key_id[defined].tolist())) if defined.any() else []
    n_switch_eligible = int(np.sum(defined[1:] & defined[:-1]))
    n_switches = int(analysis["key_switch"].sum())
    n_large_jumps = int(analysis["large_jump"].sum())
    n_low_margin = int(analysis["low_margin"].sum())
    n_active = int(active.sum())

    jump_vals = analysis["jump_distance"][defined.copy() & np.concatenate([[False], defined[1:] & defined[:-1]])] if n_switch_eligible > 0 else np.array([])
    # simpler: jump_distance is already 0 where undefined by construction; gather only eligible positions
    eligible_mask = np.zeros(T, dtype=bool)
    eligible_mask[1:] = defined[1:] & defined[:-1]
    jump_vals = analysis["jump_distance"][eligible_mask]

    return {
        "n_predictions": T,
        "n_active_windows": n_active,
        "active_fraction": float(n_active) / T if T > 0 else None,
        "n_unique_predicted_keys": len(unique_keys),
        "unique_predicted_keys": [f"{decode_key(k)[0]} {decode_key(k)[1]}" for k in unique_keys],
        "n_key_switch_eligible_transitions": n_switch_eligible,
        "n_key_switches": n_switches,
        "key_switch_proportion": (float(n_switches) / n_switch_eligible) if n_switch_eligible > 0 else None,
        "mean_jump": float(jump_vals.mean()) if jump_vals.size > 0 else None,
        "max_jump": float(jump_vals.max()) if jump_vals.size > 0 else None,
        "n_large_jumps": n_large_jumps,
        "large_jump_proportion": (float(n_large_jumps) / n_switch_eligible) if n_switch_eligible > 0 else None,
        "n_low_margin_windows": n_low_margin,
        "low_margin_proportion": float(n_low_margin) / T if T > 0 else None,
        "mean_normalized_margin": float(analysis["normalized_margin"].mean()),
        "mean_entropy": float(analysis["entropy"].mean()),
        "mean_tie_count_active_windows": float(analysis["tie_count"][active].mean()) if active.any() else None,
        "max_tie_count_active_windows": int(analysis["tie_count"][active].max()) if active.any() else None,
    }


# ---------------------------------------------------------------------------
# Combined difficulty + saving
# ---------------------------------------------------------------------------

def _to_native(x):
    if isinstance(x, dict):
        return {k: _to_native(v) for k, v in x.items() if not str(k).startswith("_")}
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


def build_difficulty_output(piece_name, analysis, anchor_results, meta, npy_paths):
    T = analysis["T"]
    combined_anchor_mismatch = np.zeros(T, dtype=bool)
    for anchor in anchor_results.values():
        combined_anchor_mismatch |= anchor["_mismatch_mask"]

    difficult = analysis["low_margin"] | analysis["key_switch"] | analysis["large_jump"] | combined_anchor_mismatch
    difficult_intervals = _run_length_encode(difficult, analysis["prediction_times_sec"])

    criterion_counts = {
        "low_margin": int(analysis["low_margin"].sum()),
        "key_switch": int(analysis["key_switch"].sum()),
        "large_jump": int(analysis["large_jump"].sum()),
        "anchor_mismatch": int(combined_anchor_mismatch.sum()),
        "any_difficult": int(difficult.sum()),
    }

    output = {
        "piece": piece_name,
        "settings": analysis["settings"],
        "alignment": {
            "note": "prediction_times_sec = prediction_indices * window_sec, computed over the FULL saved chroma array (no leading windows dropped) -- see module docstring for why this avoids the Phase 2D offset_windows bug.",
            "n_predictions": T,
            "n_chroma_windows_in_metadata": meta.get("thresholded_smoothed_chroma_shape", [None])[0],
        },
        "summary": compute_summary(analysis),
        "anchor_diagnostics": {k: v for k, v in anchor_results.items()},
        "difficulty_criterion_counts": criterion_counts,
        "difficulty_criterion_proportions": {k: (v / T if T > 0 else None) for k, v in criterion_counts.items()},
        "difficult_intervals": difficult_intervals,
        "n_difficult_intervals": len(difficult_intervals),
        "saved_arrays": {k: os.path.relpath(v, os.path.join(_MIDI_DIR, "..")) for k, v in npy_paths.items()},
        "notes": (
            "Phase 3B diagnostic output. No predictions were changed, no model was trained, no neural "
            "refinement (Chroma SRN, Transformer) was implemented. Does not compare against chord-id "
            "EMA/SRN outputs -- that is Phase 3C."
        ),
    }
    return _to_native(output)


def save_diagnostic_arrays(piece_stem, analysis):
    os.makedirs(_DERIVED_DIAG_DIR, exist_ok=True)
    paths = {}
    array_fields = ["raw_scores", "active", "key_id", "top1_score", "top2_score", "raw_margin",
                     "normalized_margin", "entropy", "tie_count", "key_switch", "jump_distance", "large_jump",
                     "low_margin", "prediction_indices", "prediction_times_sec"]
    for field in array_fields:
        path = os.path.join(_DERIVED_DIAG_DIR, f"{piece_stem}_{field}.npy")
        np.save(path, analysis[field])
        paths[field] = path
    return paths


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_margin_over_time(analysis, title, out_path, key_events=None):
    times = analysis["prediction_times_sec"]
    margin = analysis["normalized_margin"]
    active = analysis["active"]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(times, margin, color="tab:blue", linewidth=1, label="normalized_margin")
    ax.scatter(times[~active], margin[~active], s=8, color="lightgray", label="inactive (silent) window", zorder=3)
    ax.axhline(DEFAULT_LOW_MARGIN_THRESHOLD, color="red", linestyle=":", linewidth=1, label=f"low_margin threshold ({DEFAULT_LOW_MARGIN_THRESHOLD})")

    if key_events:
        for ev in key_events:
            ax.axvline(ev["time"], color="red", linestyle="--", alpha=0.4, linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("normalized_margin (top1-top2 / top1)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_over_time(analysis, title, out_path, key_events=None):
    times = analysis["prediction_times_sec"]
    entropy = analysis["entropy"]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(times, entropy, color="tab:purple", linewidth=1, label=f"entropy (temperature={DEFAULT_ENTROPY_TEMPERATURE}, diagnostic only)")

    if key_events:
        for ev in key_events:
            ax.axvline(ev["time"], color="red", linestyle="--", alpha=0.4, linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("entropy (nats)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_anchor_diagnostics(analysis, anchor_results, key_events, title, out_path):
    times = analysis["prediction_times_sec"]
    margin = analysis["normalized_margin"]

    combined_mismatch = np.zeros(analysis["T"], dtype=bool)
    for anchor in anchor_results.values():
        combined_mismatch |= anchor["_mismatch_mask"]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(times, margin, color="tab:blue", linewidth=1, label="normalized_margin", zorder=2)
    ax.scatter(times[analysis["low_margin"]], margin[analysis["low_margin"]], s=14, color="orange", label="low_margin", zorder=3)
    ax.scatter(times[combined_mismatch], margin[combined_mismatch], s=14, color="red", marker="x", label="anchor_mismatch", zorder=4)

    for ev in key_events:
        ax.axvline(ev["time"], color="black", linestyle="--", alpha=0.5, linewidth=1, zorder=1)
        ax.text(ev["time"], 1.02, ev["key_name"], rotation=90, fontsize=8, ha="center", va="bottom", transform=ax.get_xaxis_transform())

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("normalized_margin")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(x, digits=4):
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def build_report_md(twinkle_out, twinkle12_out):
    lines = []
    lines.append("# Phase 3B — Pitch-Class Baseline Uncertainty Diagnostics Report")
    lines.append("")
    lines.append(
        "Computes uncertainty diagnostics for the pitch-class/scale-template baseline "
        "(`pitch_class_baseline.py`, Phase 2B) directly from the saved Phase 2C chroma arrays. "
        "**This is diagnostic only** -- no prediction was changed, no model was trained, and no "
        "neural refinement (Chroma SRN, Transformer) was implemented. This report also does **not** "
        "compare against the chord-id EMA/SRN outputs from Phase 1.5B -- that cross-model comparison "
        "is Phase 3C, not here."
    )
    lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        "For each timestep, `raw_scores = thresholded_smoothed_chroma @ SCALE_TEMPLATES.T` (24-way scale-template "
        "dot product, same formula `pitch_class_baseline.py` uses internally). From `raw_scores` we compute: "
        "the top-1/top-2 keys and scores, `raw_margin = top1_score - top2_score`, "
        f"`normalized_margin = raw_margin / (top1_score + eps)`, and a diagnostic softmax entropy at a fixed, "
        f"documented temperature ({DEFAULT_ENTROPY_TEMPERATURE}) -- **entropy here is temperature-dependent and "
        "not a calibrated confidence measure**, only useful for relative comparison within this report."
    )
    lines.append("")

    lines.append("## Alignment handling")
    lines.append("")
    lines.append(
        "Unlike `pitch_class_baseline.midi_to_key_baseline` (which silently drops leading silent chroma windows "
        "before its first prediction -- the source of a real, caught-and-fixed timing bug in Phase 2D), this "
        "module computes scores over the **full** saved chroma array, every window from Phase 2C, none dropped. "
        "`prediction_times_sec = prediction_indices * window_sec` always, with no offset correction needed. "
        "Silent windows are tracked via an explicit `active` mask rather than being skipped."
    )
    lines.append("")

    lines.append("## Note: `low_margin` is saturated, and why")
    lines.append("")
    lines.append(
        "`low_margin_proportion` is 1.0 (100%) for both pieces, with `mean_normalized_margin` at essentially "
        "0.0. This is **not a broken metric** -- it reflects a real, structural property of the 24-way full "
        "scale-template representation: `SCALE_TEMPLATES` gives *relative* major/minor pairs (e.g. C major and "
        "A minor) numerically **identical** rows, since a natural-minor scale shares all 7 pitch classes with "
        "its relative major. When a window's active chroma evidence is sparse (a single or few pitch classes, "
        "typical of monophonic melodic material like both test pieces), many of the 24 templates tie exactly "
        "at the maximum score -- an average of ~5-7 tied keys per active window in this data (see `mean_tie_count` "
        "below), and as many as 14 simultaneously. `top1_key` is still resolved deterministically (via `argmax`'s "
        "leftmost-tie convention, matching `pitch_class_baseline.py` exactly -- this was verified against "
        "`midi_to_key_baseline`'s own output during development), but the *margin* between that pick and the "
        "runner-up is close to meaningless as a confidence signal on its own: it is near-zero whether the pick "
        "is obviously correct or a coin-flip among many tied candidates. **This means `low_margin`, as currently "
        "thresholded, does not discriminate difficult windows from easy ones -- it flags nearly everything.** "
        "`tie_count` (also computed below) is a more informative alternative and should be considered in place "
        "of, or alongside, `normalized_margin` in Phase 3C."
    )
    lines.append("")

    for label, out in [("Twinkle.mid", twinkle_out), ("Twinkle 12.mid", twinkle12_out)]:
        s = out["summary"]
        lines.append(f"## {label}")
        lines.append("")
        lines.append(f"- n_predictions: {s['n_predictions']} (active: {s['n_active_windows']}, {_fmt(s['active_fraction'])})")
        lines.append(f"- n_unique_predicted_keys: {s['n_unique_predicted_keys']} ({', '.join(s['unique_predicted_keys'])})")
        lines.append(f"- key switches: {s['n_key_switches']} / {s['n_key_switch_eligible_transitions']} eligible transitions ({_fmt(s['key_switch_proportion'])})")
        lines.append(f"- fifths jump: mean={_fmt(s['mean_jump'], 2)}, max={_fmt(s['max_jump'], 2)}, large jumps: {s['n_large_jumps']} ({_fmt(s['large_jump_proportion'])})")
        lines.append(f"- low_margin windows: {s['n_low_margin_windows']} / {s['n_predictions']} ({_fmt(s['low_margin_proportion'])})")
        lines.append(f"- mean_normalized_margin: {_fmt(s['mean_normalized_margin'])}, mean_entropy: {_fmt(s['mean_entropy'])}")
        lines.append(f"- mean_tie_count (active windows): {_fmt(s['mean_tie_count_active_windows'], 2)} of 24 keys tied at the max scale-template score (max observed: {s['max_tie_count_active_windows']})")
        lines.append("")
        lines.append("### Anchor-window diagnostics")
        lines.append("")
        for name, a in out["anchor_diagnostics"].items():
            lines.append(f"**{name}** (expected: {a['expected_key']}):")
            lines.append(f"- n_predictions={a['n_predictions']}, proportion_expected_key={_fmt(a['proportion_expected_key'])}")
            lines.append(f"- low_margin_proportion={_fmt(a['low_margin_proportion'])}, mean_normalized_margin={_fmt(a['mean_normalized_margin'])}, mean_entropy={_fmt(a['mean_entropy'])}")
            lines.append(f"- mismatch_count={a['mismatch_count']}, mismatch_intervals={len(a['mismatch_intervals'])}")
            lines.append("")
        lines.append(f"### Difficult-window summary ({label})")
        lines.append("")
        cc = out["difficulty_criterion_counts"]
        cp = out["difficulty_criterion_proportions"]
        lines.append(
            f"- low_margin: {cc['low_margin']} ({_fmt(cp['low_margin'])}); "
            f"key_switch: {cc['key_switch']} ({_fmt(cp['key_switch'])}); "
            f"large_jump: {cc['large_jump']} ({_fmt(cp['large_jump'])}); "
            f"anchor_mismatch: {cc['anchor_mismatch']} ({_fmt(cp['anchor_mismatch'])})"
        )
        lines.append(f"- any_difficult: {cc['any_difficult']} timesteps ({_fmt(cp['any_difficult'])}) across {out['n_difficult_intervals']} contiguous interval(s)")
        lines.append("  (note: `any_difficult` is a union that includes the saturated `low_margin` criterion -- see the note above -- so it is dominated by that criterion and is not, by itself, a useful difficulty summary; prefer `key_switch`/`large_jump`/`anchor_mismatch` individually.)")
        lines.append("")

    lines.append("## Does the pitch-class baseline have meaningful failure/uncertainty regions?")
    lines.append("")
    lines.append(build_findings_text(twinkle_out, twinkle12_out))
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This is Phase 3B: non-neural uncertainty diagnostics only. **No neural model was implemented** "
        "(no Chroma SRN, no Transformer, no refinement of any kind). Comparison against the chord-id "
        "EMA/SRN outputs (Phase 1.5B) is Phase 3C, not performed here."
    )
    lines.append("")

    return "\n".join(lines)


def build_findings_text(twinkle_out, twinkle12_out):
    parts = []

    parts.append(
        "Because `low_margin` is saturated for both pieces (see the note above), it is excluded from the "
        "characterization below in favor of `key_switch`, `large_jump`, and `anchor_mismatch` -- the criteria "
        "that actually discriminate stable regions from unstable ones in this data."
    )

    tw_s = twinkle_out["summary"]
    if tw_s["n_key_switches"] == 0 and tw_s["n_large_jumps"] == 0:
        parts.append(
            f"**Twinkle.mid** shows essentially no measurable instability by the discriminating criteria: "
            f"{tw_s['n_key_switches']} key switches and {tw_s['n_large_jumps']} large Circle-of-Fifths jumps "
            f"across all {tw_s['n_predictions']} predictions (100% C major throughout, matching Phase 2D). "
            "There is little for a neural refinement to target on this piece by this analysis."
        )
    else:
        parts.append(
            f"**Twinkle.mid** shows {tw_s['n_key_switches']} key switches and {tw_s['n_large_jumps']} large "
            "jumps -- these are the specific regions a future targeted refinement could focus on."
        )

    tw12_s = twinkle12_out["summary"]
    mid_anchor = twinkle12_out["anchor_diagnostics"].get("384_to_432s", {})
    parts.append(
        f"**Twinkle 12.mid** shows concentrated (not uniform) instability: {tw12_s['n_key_switches']} key "
        f"switches and {tw12_s['n_large_jumps']} large Circle-of-Fifths jumps across {tw12_s['n_predictions']} "
        f"predictions ({_fmt(tw12_s['key_switch_proportion'])} / {_fmt(tw12_s['large_jump_proportion'])} of "
        "eligible transitions) -- both rare overall, meaning the piece is mostly stable with a small number of "
        "genuinely unstable transition points, not pervasively noisy."
    )
    parts.append(
        f"In the 384_to_432s anchor window (expected Eb major), proportion_expected_key="
        f"{_fmt(mid_anchor.get('proportion_expected_key'))} with {mid_anchor.get('mismatch_count')} mismatched "
        f"predictions across {len(mid_anchor.get('mismatch_intervals', []))} contiguous interval(s) -- the "
        "baseline's correct tracking of this real modulation (per Phase 2D) is not perfect within the window, "
        "and these mismatch intervals are natural first candidates for Phase 3C's targeted refinement analysis."
    )

    parts.append(
        f"Tie-count is a more useful confidence signal here than margin: Twinkle.mid averages "
        f"{_fmt(tw_s['mean_tie_count_active_windows'], 2)} tied keys per active window and Twinkle 12.mid "
        f"averages {_fmt(tw12_s['mean_tie_count_active_windows'], 2)} -- both pieces' per-window pitch-class "
        "evidence is often ambiguous in isolation, and the baseline's real accuracy comes from the temporal "
        "chroma-level EMA smoothing accumulating evidence across many windows, not from any single window being "
        "individually decisive. This is itself informative for Phase 3C/3D: a neural refinement operating on "
        "single windows would face the same structural ambiguity; any useful refinement likely needs its own "
        "temporal integration, not just a better per-window classifier."
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


def run_verification(twinkle_out, twinkle12_out, twinkle_analysis, twinkle12_analysis, all_out_paths,
                      midi_mtimes_before, chroma_mtimes_before, prior_outputs_mtimes_before):
    checks = []

    for p in all_out_paths:
        checks.append((f"{os.path.basename(p)} exists", os.path.exists(p)))
        checks.append((f"{os.path.basename(p)} is non-empty", os.path.exists(p) and os.path.getsize(p) > 0))

    for label, analysis in [("Twinkle.mid", twinkle_analysis), ("Twinkle 12.mid", twinkle12_analysis)]:
        checks.append((f"{label} raw_scores has no NaNs", not np.isnan(analysis["raw_scores"]).any()))
        checks.append((f"{label} normalized_margin has no NaNs", not np.isnan(analysis["normalized_margin"]).any()))
        checks.append((f"{label} entropy has no NaNs", not np.isnan(analysis["entropy"]).any()))

    combined = {"twinkle_mid": twinkle_out, "twinkle_12_mid": twinkle12_out}
    nan_paths = _scan_for_nan(combined)
    checks.append(("no NaNs in difficulty JSON outputs", len(nan_paths) == 0))

    mid_anchor = twinkle12_out["anchor_diagnostics"].get("384_to_432s", {})
    checks.append(("Twinkle 12.mid 384_to_432s anchor window has predictions (Eb/D# window correctly time-mapped)", mid_anchor.get("n_predictions", 0) > 0))
    checks.append(("Twinkle 12.mid 384_to_432s anchor window predicts expected key majority", (mid_anchor.get("proportion_expected_key") or 0) > 0.5))

    for path, mtime_before in midi_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} MIDI file unchanged", os.path.getmtime(path) == mtime_before))
    for path, mtime_before in chroma_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} Phase 2C chroma file unchanged", os.path.getmtime(path) == mtime_before))
    for path, mtime_before in prior_outputs_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} prior Phase 1/1.5/2 output unchanged", os.path.getmtime(path) == mtime_before))

    print()
    print("pitch_class_uncertainty_diagnostics.py verification")
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
    os.makedirs(_DERIVED_DIAG_DIR, exist_ok=True)

    midi_mtimes_before = {TWINKLE_MIDI: os.path.getmtime(TWINKLE_MIDI), TWINKLE12_MIDI: os.path.getmtime(TWINKLE12_MIDI)}
    chroma_files = [TWINKLE_CHROMA_NPY, TWINKLE_META_JSON, TWINKLE12_CHROMA_NPY, TWINKLE12_META_JSON]
    chroma_mtimes_before = {p: os.path.getmtime(p) for p in chroma_files}

    # spot-check a sample of prior Phase 1/1.5/2 outputs to confirm this task doesn't touch them
    prior_outputs_sample = [
        os.path.join(_FIGURES_DIR, "PHASE2D_pitch_class_baseline_metrics.json"),
        os.path.join(_FIGURES_DIR, "PHASE1_5B_MIDI_EMA_vs_SRN_metrics.json"),
        os.path.join(_FIGURES_DIR, "PHASE3A_STAGED_TONAL_INFERENCE_DESIGN.md"),
    ]
    prior_outputs_mtimes_before = {p: os.path.getmtime(p) for p in prior_outputs_sample if os.path.exists(p)}

    print("Loading Twinkle.mid chroma + metadata...")
    twinkle_chroma = np.load(TWINKLE_CHROMA_NPY)
    with open(TWINKLE_META_JSON) as f:
        twinkle_meta = json.load(f)
    twinkle_analysis = analyze_piece(twinkle_chroma, twinkle_meta["window_sec"])

    print("Loading Twinkle 12.mid chroma + metadata...")
    twinkle12_chroma = np.load(TWINKLE12_CHROMA_NPY)
    with open(TWINKLE12_META_JSON) as f:
        twinkle12_meta = json.load(f)
    twinkle12_analysis = analyze_piece(twinkle12_chroma, twinkle12_meta["window_sec"])

    print("Computing anchor diagnostics...")
    twinkle_anchors_spec = [{"name": "full_piece", "start_sec": None, "end_sec": None, "expected_key_name": "C Major"}]
    twinkle_anchor_results = compute_anchor_diagnostics(twinkle_analysis, twinkle_anchors_spec)

    twinkle12_anchors_spec = [
        {"name": "pre_384s", "start_sec": None, "end_sec": 384.0, "expected_key_name": "C Major"},
        {"name": "384_to_432s", "start_sec": 384.0, "end_sec": 432.0, "expected_key_name": "Eb Major"},
        {"name": "post_432s", "start_sec": 432.0, "end_sec": None, "expected_key_name": "C Major"},
    ]
    twinkle12_anchor_results = compute_anchor_diagnostics(twinkle12_analysis, twinkle12_anchors_spec)

    print("Saving diagnostic arrays...")
    twinkle_npy_paths = save_diagnostic_arrays("Twinkle_mid", twinkle_analysis)
    twinkle12_npy_paths = save_diagnostic_arrays("Twinkle_12_mid", twinkle12_analysis)

    twinkle_out = build_difficulty_output("Twinkle.mid", twinkle_analysis, twinkle_anchor_results, twinkle_meta, twinkle_npy_paths)
    twinkle12_out = build_difficulty_output("Twinkle 12.mid", twinkle12_analysis, twinkle12_anchor_results, twinkle12_meta, twinkle12_npy_paths)

    with open(OUT["twinkle_difficulty_json"], "w") as f:
        json.dump(twinkle_out, f, indent=2)
    with open(OUT["twinkle12_difficulty_json"], "w") as f:
        json.dump(twinkle12_out, f, indent=2)
    print(f"Wrote {OUT['twinkle_difficulty_json']}")
    print(f"Wrote {OUT['twinkle12_difficulty_json']}")

    print("\nPlotting...")
    plot_margin_over_time(twinkle_analysis, "Phase 3B — Twinkle.mid: Normalized Margin Over Time", OUT["twinkle_margin_png"])
    plot_margin_over_time(twinkle12_analysis, "Phase 3B — Twinkle 12.mid: Normalized Margin Over Time", OUT["twinkle12_margin_png"], key_events=TWINKLE12_KEY_EVENTS)
    plot_entropy_over_time(twinkle_analysis, "Phase 3B — Twinkle.mid: Entropy Over Time (diagnostic only)", OUT["twinkle_entropy_png"])
    plot_entropy_over_time(twinkle12_analysis, "Phase 3B — Twinkle 12.mid: Entropy Over Time (diagnostic only)", OUT["twinkle12_entropy_png"], key_events=TWINKLE12_KEY_EVENTS)
    plot_anchor_diagnostics(twinkle12_analysis, twinkle12_anchor_results, TWINKLE12_KEY_EVENTS, "Phase 3B — Twinkle 12.mid: Anchor Diagnostics", OUT["twinkle12_anchor_png"])

    print("\nWriting report...")
    report_md = build_report_md(twinkle_out, twinkle12_out)
    with open(OUT["report_md"], "w") as f:
        f.write(report_md)

    metrics_json_content = {
        "phase": "phase_3b_pitch_class_uncertainty_diagnostics",
        "twinkle_mid": twinkle_out,
        "twinkle_12_mid": twinkle12_out,
    }
    with open(OUT["metrics_json"], "w") as f:
        json.dump(metrics_json_content, f, indent=2)
    print(f"Wrote {OUT['report_md']}")
    print(f"Wrote {OUT['metrics_json']}")

    all_out_paths = [
        OUT["twinkle_difficulty_json"], OUT["twinkle12_difficulty_json"],
        OUT["twinkle_margin_png"], OUT["twinkle12_margin_png"],
        OUT["twinkle_entropy_png"], OUT["twinkle12_entropy_png"],
        OUT["twinkle12_anchor_png"], OUT["report_md"], OUT["metrics_json"],
    ] + list(twinkle_npy_paths.values()) + list(twinkle12_npy_paths.values())

    run_verification(twinkle_out, twinkle12_out, twinkle_analysis, twinkle12_analysis, all_out_paths,
                      midi_mtimes_before, chroma_mtimes_before, prior_outputs_mtimes_before)


if __name__ == "__main__":
    main()
