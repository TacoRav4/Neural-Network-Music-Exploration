"""compare_phase3c_disagreement.py

Phase 3C: compare Phase 3B's pitch-class difficult windows against
Phase 1.5B's chord-id EMA/SRN behavior, with careful real-time alignment.

This is **comparison/diagnostics only**. It does not change any prediction,
does not train anything new in the sense of tuning, and does not implement
any neural refinement, Chroma SRN, or Transformer. It reuses (imports only,
does not modify) mlp_baseline.py, srn_model.py, run_comparison.py, and
run_midi_phase15_evaluation.py's own `train_models()` to regenerate the
EMA+MLP and SRN models with the exact same settings Phase 1.5B used --
Phase 1.5B itself did not save per-timestep probability/key arrays to disk
(only PNGs, a metrics JSON, and a markdown report), so this script
regenerates them (deterministically, seed=269) purely to get per-timestep
key sequences to compare against, and saves the results as new, clearly
Phase-3C-labeled derived files. No old file is modified.

Per Phase 3B's finding: `low_margin` (top1-top2 scale-template margin) is
structurally saturated (~100%) for both pieces due to widespread exact ties
among SCALE_TEMPLATES rows (relative major/minor pairs share identical
pitch-class sets). It is therefore NOT used here as a standalone difficulty
criterion -- only key_switch, large_jump, anchor_mismatch, and (optionally)
high tie_count are used, per this task's explicit instruction.

Alignment: this script never assumes index alignment between the
pitch-class path (Phase 3B, computed over the FULL undropped chroma array)
and the chord-id path (Phase 1.5A, which -- like
pitch_class_baseline.midi_to_key_baseline -- drops leading silent windows
before its first prediction). Both are placed on a common real-time axis
via `prediction_times_sec` (pitch-class, from Phase 3B) and a freshly
computed `chord_prediction_times_sec` (chord-id, from Phase 1.5A's own
`offset_windows = n_chroma_windows - n_chord_ids`, the same convention
Phase 2D established and Phase 3B reused). Alignment is verified, not
assumed -- see `verify_alignment()`.
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
_DERIVED_CHORD_DIR = os.path.join(_MIDI_DIR, "derived_chord_sequences")
_DERIVED_DIAG_DIR = os.path.join(_MIDI_DIR, "derived_phase3_diagnostics")
_FIGURES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "05_Figures_Results"))

from shared_music_defs import decode_key, fifth_distance, key_tonic_pc, key_index, FIFTH_POS
from mlp_baseline import sequence_key_tracking
from srn_model import predict_sequence_probs
from run_midi_phase15_evaluation import train_models, PRIMARY_ALPHA, SRN_EPOCHS, SRN_LR, SRN_HIDDEN_SIZE  # reused, not modified
from plotting_comparison import FIFTHS_LABEL_NAMES  # reused, not modified

TWINKLE_MIDI = os.path.join(_MIDI_DIR, "Twinkle.mid")
TWINKLE12_MIDI = os.path.join(_MIDI_DIR, "Twinkle 12.mid")

HIGH_TIE_COUNT_THRESHOLD = 8  # documented, diagnostic-only: above both pieces' mean tie_count (~5.2 / ~6.7 per Phase 3B), not tuned

TWINKLE12_KEY_EVENTS = [
    {"time": 0.0, "key_name": "C Major"},
    {"time": 384.0, "key_name": "Eb Major"},
    {"time": 392.0, "key_name": "Eb Major"},
    {"time": 432.0, "key_name": "C Major"},
    {"time": 440.0, "key_name": "C Major"},
]
TWINKLE12_ANCHORS = [
    {"name": "pre_384s", "start_sec": None, "end_sec": 384.0, "expected_key_name": "C Major"},
    {"name": "384_to_432s", "start_sec": 384.0, "end_sec": 432.0, "expected_key_name": "Eb Major"},
    {"name": "post_432s", "start_sec": 432.0, "end_sec": None, "expected_key_name": "C Major"},
]
TWINKLE_ANCHORS = [{"name": "full_piece", "start_sec": None, "end_sec": None, "expected_key_name": "C Major"}]

_PC_MAP = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6,
           "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}


def _expected_key_id(expected_key_name):
    tonic, mode = expected_key_name.split()
    mode = "maj" if mode.lower().startswith("maj") else "min"
    return key_index(_PC_MAP[tonic], mode)


OUT = {
    "twinkle_report_md": os.path.join(_FIGURES_DIR, "PHASE3C_Twinkle_disagreement_report.md"),
    "twinkle12_report_md": os.path.join(_FIGURES_DIR, "PHASE3C_Twinkle12_disagreement_report.md"),
    "metrics_json": os.path.join(_FIGURES_DIR, "PHASE3C_disagreement_metrics.json"),
    "twinkle12_timeline_png": os.path.join(_FIGURES_DIR, "PHASE3C_Twinkle12_disagreement_timeline.png"),
    "twinkle_timeline_png": os.path.join(_FIGURES_DIR, "PHASE3C_Twinkle_disagreement_timeline.png"),
}


# ---------------------------------------------------------------------------
# Loading Phase 1.5A chord-id data + Phase 3B pitch-class diagnostics
# ---------------------------------------------------------------------------

def load_chord_data(piece_stem):
    chord_ids = np.load(os.path.join(_DERIVED_CHORD_DIR, f"{piece_stem}_chord_ids.npy"))
    with open(os.path.join(_DERIVED_CHORD_DIR, f"{piece_stem}_chord_extraction_metadata.json")) as f:
        meta = json.load(f)
    offset = meta["n_chroma_windows"] - meta["n_chord_ids"]
    times_sec = (np.arange(len(chord_ids)) + offset) * meta["window_sec"]
    return chord_ids, meta, offset, times_sec


def load_pc_diagnostics(piece_stem):
    fields = {}
    for field in ["key_id", "prediction_times_sec", "key_switch", "large_jump", "tie_count", "active"]:
        fields[field] = np.load(os.path.join(_DERIVED_DIAG_DIR, f"{piece_stem}_{field}.npy"))
    return fields


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_chord_to_pc_indices(chord_times_sec, pc_times_sec, window_sec):
    """
    For each chord-model timestep, finds the pitch-class-path index at the
    same real time. Returns (pc_indices, max_time_error_sec). Does not
    assume the two grids' array indices line up -- only that both are on
    the same window_sec real-time axis (verified via max_time_error_sec,
    which should be ~0).
    """
    pc_indices = np.round(chord_times_sec / window_sec).astype(np.int64)
    pc_indices = np.clip(pc_indices, 0, len(pc_times_sec) - 1)
    time_errors = np.abs(pc_times_sec[pc_indices] - chord_times_sec)
    return pc_indices, float(time_errors.max()) if len(time_errors) else 0.0


# ---------------------------------------------------------------------------
# Anchor mismatch (recomputed here from key_id + known anchors -- Phase 3B
# did not persist this as a standalone array, only as an internal mask)
# ---------------------------------------------------------------------------

def compute_anchor_mismatch_mask(key_id_array, times_sec, anchors):
    mask = np.zeros(len(key_id_array), dtype=bool)
    for anchor in anchors:
        t0 = anchor["start_sec"] if anchor["start_sec"] is not None else 0.0
        t1 = anchor["end_sec"] if anchor["end_sec"] is not None else float(times_sec[-1]) + 1.0
        sel = (times_sec >= t0) & (times_sec < t1)
        idxs = np.where(sel)[0]
        expected_id = _expected_key_id(anchor["expected_key_name"])
        defined = key_id_array[idxs] != -1
        mask[idxs[defined]] = key_id_array[idxs[defined]] != expected_id
    return mask


# ---------------------------------------------------------------------------
# Chord-model instability (its own key_switch / large_jump, on its own grid)
# ---------------------------------------------------------------------------

def compute_key_switch_and_jump(key_id_array):
    T = len(key_id_array)
    key_switch = np.zeros(T, dtype=bool)
    jump_distance = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        key_switch[t] = key_id_array[t] != key_id_array[t - 1]
        jump_distance[t] = fifth_distance(key_tonic_pc(key_id_array[t - 1]), key_tonic_pc(key_id_array[t]))
    large_jump = jump_distance >= 3
    return key_switch, jump_distance, large_jump


# ---------------------------------------------------------------------------
# Overlap statistics
# ---------------------------------------------------------------------------

def overlap_stats(disagreement, criterion_mask):
    overall_rate = float(disagreement.mean()) if len(disagreement) else None
    n_in = int(criterion_mask.sum())
    n_out = int((~criterion_mask).sum())
    rate_in = float(disagreement[criterion_mask].mean()) if n_in > 0 else None
    rate_out = float(disagreement[~criterion_mask].mean()) if n_out > 0 else None
    return {
        "overall_disagreement_rate": overall_rate,
        "n_criterion_windows": n_in,
        "criterion_fraction": float(criterion_mask.mean()) if len(criterion_mask) else None,
        "disagreement_rate_within_criterion": rate_in,
        "disagreement_rate_outside_criterion": rate_out,
        "concentration_ratio": (rate_in / rate_out) if (rate_in is not None and rate_out is not None and rate_out > 0) else None,
    }


# ---------------------------------------------------------------------------
# Per-piece analysis
# ---------------------------------------------------------------------------

def analyze_piece(piece_label, piece_stem, model_mlp, mlp_device, model_srn, srn_device, anchors):
    chord_ids, chord_meta, chord_offset, chord_times_sec = load_chord_data(piece_stem)
    pc = load_pc_diagnostics(piece_stem)
    window_sec = chord_meta["window_sec"]

    probs_ema = sequence_key_tracking(model_mlp, chord_ids, alpha=PRIMARY_ALPHA, device=mlp_device)
    probs_srn = predict_sequence_probs(model_srn, chord_ids, device=srn_device)
    ema_key_id = np.argmax(probs_ema, axis=1)
    srn_key_id = np.argmax(probs_srn, axis=1)

    ema_key_switch, ema_jump, ema_large_jump = compute_key_switch_and_jump(ema_key_id)
    srn_key_switch, srn_jump, srn_large_jump = compute_key_switch_and_jump(srn_key_id)

    pc_anchor_mismatch = compute_anchor_mismatch_mask(pc["key_id"], pc["prediction_times_sec"], anchors)

    pc_indices, max_time_error = align_chord_to_pc_indices(chord_times_sec, pc["prediction_times_sec"], window_sec)

    pc_key_id_aligned = pc["key_id"][pc_indices]
    pc_key_switch_aligned = pc["key_switch"][pc_indices]
    pc_large_jump_aligned = pc["large_jump"][pc_indices]
    pc_tie_count_aligned = pc["tie_count"][pc_indices]
    pc_anchor_mismatch_aligned = pc_anchor_mismatch[pc_indices]
    high_tie_count_aligned = pc_tie_count_aligned >= HIGH_TIE_COUNT_THRESHOLD

    valid = pc_key_id_aligned != -1  # exclude any (unexpected) undefined pitch-class predictions from disagreement
    disagreement_ema = (ema_key_id != pc_key_id_aligned) & valid
    disagreement_srn = (srn_key_id != pc_key_id_aligned) & valid

    criteria = {
        "pc_key_switch": pc_key_switch_aligned,
        "pc_large_jump": pc_large_jump_aligned,
        "pc_anchor_mismatch": pc_anchor_mismatch_aligned,
        "pc_high_tie_count": high_tie_count_aligned,
    }

    overlap = {"ema": {}, "srn": {}}
    for crit_name, crit_mask in criteria.items():
        overlap["ema"][crit_name] = overlap_stats(disagreement_ema, crit_mask)
        overlap["srn"][crit_name] = overlap_stats(disagreement_srn, crit_mask)

    result = {
        "piece": piece_label,
        "window_sec": window_sec,
        "n_chord_predictions": int(len(chord_ids)),
        "chord_offset_windows": int(chord_offset),
        "alignment_max_time_error_sec": max_time_error,
        "srn_settings": {"epochs": SRN_EPOCHS, "lr": SRN_LR, "hidden_size": SRN_HIDDEN_SIZE, "alpha_ema": PRIMARY_ALPHA},
        "high_tie_count_threshold": HIGH_TIE_COUNT_THRESHOLD,
        "overall_disagreement": {
            "ema": float(disagreement_ema.mean()),
            "srn": float(disagreement_srn.mean()),
        },
        "chord_model_own_instability": {
            "ema": {"n_key_switches": int(ema_key_switch.sum()), "n_large_jumps": int(ema_large_jump.sum()), "n_predictions": int(len(ema_key_id))},
            "srn": {"n_key_switches": int(srn_key_switch.sum()), "n_large_jumps": int(srn_large_jump.sum()), "n_predictions": int(len(srn_key_id))},
        },
        "overlap_with_pitch_class_difficulty_criteria": overlap,
        "note_on_low_margin": "low_margin is excluded as a standalone criterion (Phase 3B found it structurally saturated ~100% for both pieces due to SCALE_TEMPLATES ties); only key_switch, large_jump, anchor_mismatch, and high_tie_count are used here.",
    }

    arrays = {
        "chord_times_sec": chord_times_sec,
        "ema_key_id": ema_key_id,
        "srn_key_id": srn_key_id,
        "probs_ema": probs_ema,
        "probs_srn": probs_srn,
        "disagreement_ema": disagreement_ema,
        "disagreement_srn": disagreement_srn,
        "pc_key_id_aligned": pc_key_id_aligned,
        "pc_indices": pc_indices,
    }

    return result, arrays, pc


def save_phase3c_arrays(piece_stem, arrays):
    os.makedirs(_DERIVED_DIAG_DIR, exist_ok=True)
    paths = {}
    for field, arr in arrays.items():
        path = os.path.join(_DERIVED_DIAG_DIR, f"{piece_stem}_phase3c_{field}.npy")
        np.save(path, arr)
        paths[field] = path
    return paths


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_disagreement_timeline(result, arrays, pc, title, out_path, key_events=None):
    chord_times = arrays["chord_times_sec"]
    pc_pos = np.array([FIFTH_POS[key_tonic_pc(k)] if k != -1 else np.nan for k in arrays["pc_key_id_aligned"]])
    ema_pos = np.array([FIFTH_POS[key_tonic_pc(k)] for k in arrays["ema_key_id"]])
    srn_pos = np.array([FIFTH_POS[key_tonic_pc(k)] for k in arrays["srn_key_id"]])

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(chord_times, pc_pos, color="black", linewidth=1.5, label="pitch-class baseline (Stage 1)", zorder=3)
    ax.plot(chord_times, ema_pos, color="tab:blue", linewidth=1, alpha=0.8, label="chord-id EMA", zorder=2)
    ax.plot(chord_times, srn_pos, color="tab:orange", linewidth=1, alpha=0.8, label="chord-id SRN", zorder=2)

    both_disagree = arrays["disagreement_ema"] & arrays["disagreement_srn"]
    ax.scatter(chord_times[both_disagree], pc_pos[both_disagree], s=10, color="red", marker="x",
               label="EMA & SRN both disagree with Stage 1", zorder=4)

    if key_events:
        for ev in key_events:
            ax.axvline(ev["time"], color="gray", linestyle="--", alpha=0.5, linewidth=1, zorder=1)

    ax.set_yticks(range(12))
    ax.set_yticklabels(FIFTHS_LABEL_NAMES)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Predicted tonic (circle-of-fifths position)")
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


def build_report_md(piece_label, result):
    lines = []
    lines.append(f"# Phase 3C — {piece_label}: Pitch-Class vs. Chord-ID Disagreement Report")
    lines.append("")
    lines.append(
        "Compares Phase 3B's pitch-class difficult windows against Phase 1.5B's chord-id EMA/SRN behavior "
        f"on {piece_label}. **Comparison/diagnostics only** -- no prediction was changed, no model was tuned "
        "(EMA and SRN were regenerated deterministically with Phase 1.5B's exact settings, since Phase 1.5B "
        "did not persist per-timestep arrays to disk), and no neural refinement (Chroma SRN, Transformer) was "
        "implemented."
    )
    lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        f"EMA+MLP (alpha={result['srn_settings']['alpha_ema']}) and SRN (epochs={result['srn_settings']['epochs']}, "
        f"lr={result['srn_settings']['lr']}, hidden_size={result['srn_settings']['hidden_size']}) were regenerated "
        "via `run_midi_phase15_evaluation.train_models()` (reused, not modified) and run on the Phase 1.5A saved "
        f"chord-id sequence ({result['n_chord_predictions']} predictions). Disagreement at a timestep = the "
        "chord-id model's argmax key differs from the pitch-class baseline's (Phase 3B) key at the same real time."
    )
    lines.append("")

    lines.append("## Alignment handling")
    lines.append("")
    lines.append(
        f"Chord-id predictions start `chord_offset_windows={result['chord_offset_windows']}` windows into the "
        "piece (leading silent chroma windows dropped, same convention as the pitch-class path -- see Phase 2D/3B), "
        "so `chord_prediction_times_sec = (index + chord_offset_windows) * window_sec`. Each chord-id timestep is "
        "matched to the pitch-class path's prediction at the **same real time** (`round(time / window_sec)` into "
        "Phase 3B's full, undropped array), not by assuming shared array indices. "
        f"**Alignment verification: max time error = {_fmt(result['alignment_max_time_error_sec'], 6)} seconds** "
        "(0 confirms the two paths' window grids coincide exactly, as expected since both derive from the same "
        "raw chroma extraction at the same `window_sec`)."
    )
    lines.append("")

    lines.append("## Difficulty signals used")
    lines.append("")
    lines.append(
        "Per this task's instruction and Phase 3B's finding: `low_margin` is **not** used as a standalone "
        "criterion (Phase 3B found it structurally saturated at ~100% for both pieces, due to `SCALE_TEMPLATES` "
        "ties between relative major/minor pairs -- it does not discriminate). Criteria used: `pc_key_switch`, "
        "`pc_large_jump`, `pc_anchor_mismatch`, and `pc_high_tie_count` "
        f"(tie_count >= {result['high_tie_count_threshold']}, a fixed, documented, un-tuned threshold above both "
        "pieces' mean tie_count)."
    )
    lines.append("")

    lines.append("## Overall disagreement")
    lines.append("")
    lines.append(f"- EMA vs. pitch-class baseline: {_fmt(result['overall_disagreement']['ema'])} of timesteps disagree")
    lines.append(f"- SRN vs. pitch-class baseline: {_fmt(result['overall_disagreement']['srn'])} of timesteps disagree")
    lines.append("")

    lines.append("## Chord-id model's own instability (for context)")
    lines.append("")
    for model in ["ema", "srn"]:
        cm = result["chord_model_own_instability"][model]
        lines.append(f"- {model.upper()}: {cm['n_key_switches']} key switches, {cm['n_large_jumps']} large jumps, across {cm['n_predictions']} predictions")
    lines.append("")

    lines.append("## Overlap between disagreement and pitch-class difficulty criteria")
    lines.append("")
    lines.append("| model | criterion | criterion_fraction | disagreement_rate_within | disagreement_rate_outside | concentration_ratio |")
    lines.append("|---|---|---|---|---|---|")
    for model in ["ema", "srn"]:
        for crit_name, stats in result["overlap_with_pitch_class_difficulty_criteria"][model].items():
            lines.append(
                f"| {model.upper()} | {crit_name} | {_fmt(stats['criterion_fraction'])} | "
                f"{_fmt(stats['disagreement_rate_within_criterion'])} | {_fmt(stats['disagreement_rate_outside_criterion'])} | "
                f"{_fmt(stats['concentration_ratio'], 2)} |"
            )
    lines.append("")
    lines.append(
        "`concentration_ratio` = disagreement rate inside the criterion / disagreement rate outside it. "
        "A ratio near 1.0 means disagreement is roughly **uniform** regardless of this criterion (a global "
        "pattern); a ratio well above 1.0 means disagreement **concentrates** inside these flagged windows "
        "(a local, targetable pattern)."
    )
    lines.append("")

    lines.append("## Interpretation for staged architecture")
    lines.append("")
    lines.append(build_interpretation(result))
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This is Phase 3C: comparison/diagnostics only. No neural refinement, Chroma SRN, or Transformer was "
        "implemented. No dense per-timestep MIDI accuracy is claimed anywhere in this report -- all comparisons "
        "are between two model outputs (pitch-class vs. chord-id), not against ground truth."
    )
    lines.append("")

    return "\n".join(lines)


def build_interpretation(result):
    parts = []
    ema_overall = result["overall_disagreement"]["ema"]
    srn_overall = result["overall_disagreement"]["srn"]

    ratios = []
    for model in ["ema", "srn"]:
        for crit, stats in result["overlap_with_pitch_class_difficulty_criteria"][model].items():
            if stats["concentration_ratio"] is not None:
                ratios.append(stats["concentration_ratio"])

    mean_ratio = (sum(ratios) / len(ratios)) if ratios else None

    if mean_ratio is not None and mean_ratio > 2.0:
        pattern = (
            f"Disagreement concentrates meaningfully inside the flagged difficulty windows (mean concentration "
            f"ratio across criteria/models ≈ {mean_ratio:.2f}) rather than being spread uniformly -- this "
            "supports a **local, targetable pattern**: a Stage 4 neural refinement applied specifically to these "
            "flagged regions could plausibly address a real fraction of the chord-id models' disagreement with "
            "the (Phase 2D-validated) pitch-class baseline."
        )
    elif mean_ratio is not None:
        pattern = (
            f"Disagreement is only weakly concentrated inside the flagged difficulty windows (mean concentration "
            f"ratio ≈ {mean_ratio:.2f}, close to 1.0) -- overall disagreement rates are "
            f"{_fmt(ema_overall)} (EMA) and {_fmt(srn_overall)} (SRN), comparable to the rates seen inside and "
            "outside the flagged windows. This supports a **global representation-wide bias** interpretation "
            "(consistent with Phase 1.5B's F-major-bias finding) rather than a small number of locally difficult "
            "spots -- a Stage 4 neural refinement targeted only at these specific flagged windows would likely "
            "leave most of the disagreement untouched, since the disagreement is not concentrated there."
        )
    else:
        pattern = "Insufficient data to compute a concentration pattern for this piece."

    parts.append(pattern)
    parts.append(
        "This directly informs Phase 3A's Stage 3/4 design question: if disagreement is global, the staged "
        "architecture's premise (fix a validated fast filter's rare local failures with a small, targeted neural "
        "pass) does not describe the chord-id models' actual failure mode -- the chord-id path's problem is "
        "representation-wide, not a handful of hard windows, and no amount of *targeted* refinement (as opposed "
        "to representation-level change, e.g. a Chroma SRN operating on chroma directly rather than triadic "
        "chord ids) would be expected to close most of the gap. If disagreement is local, the staged premise "
        "holds and Stage 4 is worth pursuing on the flagged windows specifically."
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


def run_verification(combined_metrics, all_out_paths, midi_mtimes_before, prior_outputs_mtimes_before,
                      twinkle12_result):
    checks = []

    for p in all_out_paths:
        checks.append((f"{os.path.basename(p)} exists", os.path.exists(p)))
        checks.append((f"{os.path.basename(p)} is non-empty", os.path.exists(p) and os.path.getsize(p) > 0))

    nan_paths = _scan_for_nan(combined_metrics)
    checks.append(("no NaNs in metrics", len(nan_paths) == 0))

    checks.append(("Twinkle 12.mid alignment max_time_error_sec is ~0 (grids coincide)", twinkle12_result["alignment_max_time_error_sec"] < 1e-6))

    for path, mtime_before in midi_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} MIDI file unchanged", os.path.getmtime(path) == mtime_before))
    for path, mtime_before in prior_outputs_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} prior Phase 1/1.5/2/3B output unchanged", os.path.getmtime(path) == mtime_before))

    print()
    print("compare_phase3c_disagreement.py verification")
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
    prior_outputs_sample = [
        os.path.join(_FIGURES_DIR, "PHASE1_5B_MIDI_EMA_vs_SRN_metrics.json"),
        os.path.join(_FIGURES_DIR, "PHASE2D_pitch_class_baseline_metrics.json"),
        os.path.join(_FIGURES_DIR, "PHASE3B_pitch_class_uncertainty_metrics.json"),
        os.path.join(_DERIVED_DIAG_DIR, "Twinkle_12_mid_pitch_class_difficulty.json"),
        os.path.join(_DERIVED_CHORD_DIR, "Twinkle_12_mid_chord_ids.npy"),
    ]
    prior_outputs_mtimes_before = {p: os.path.getmtime(p) for p in prior_outputs_sample if os.path.exists(p)}

    print("Training EMA+MLP and SRN (reusing run_midi_phase15_evaluation.train_models(), deterministic seed=269)...")
    model_mlp, mlp_device, model_srn, srn_device, mlp_final, srn_final = train_models()

    print("\nAnalyzing Twinkle.mid...")
    twinkle_result, twinkle_arrays, twinkle_pc = analyze_piece("Twinkle.mid", "Twinkle_mid", model_mlp, mlp_device, model_srn, srn_device, TWINKLE_ANCHORS)
    print(f"  overall disagreement: EMA={twinkle_result['overall_disagreement']['ema']:.4f} SRN={twinkle_result['overall_disagreement']['srn']:.4f}")

    print("\nAnalyzing Twinkle 12.mid...")
    twinkle12_result, twinkle12_arrays, twinkle12_pc = analyze_piece("Twinkle 12.mid", "Twinkle_12_mid", model_mlp, mlp_device, model_srn, srn_device, TWINKLE12_ANCHORS)
    print(f"  overall disagreement: EMA={twinkle12_result['overall_disagreement']['ema']:.4f} SRN={twinkle12_result['overall_disagreement']['srn']:.4f}")
    print(f"  alignment max_time_error_sec: {twinkle12_result['alignment_max_time_error_sec']}")

    print("\nSaving Phase 3C derived arrays...")
    twinkle_saved_paths = save_phase3c_arrays("Twinkle_mid", twinkle_arrays)
    twinkle12_saved_paths = save_phase3c_arrays("Twinkle_12_mid", twinkle12_arrays)

    print("\nPlotting...")
    plot_disagreement_timeline(
        twinkle12_result, twinkle12_arrays, twinkle12_pc,
        "Phase 3C — Twinkle 12.mid: Pitch-Class vs. Chord-ID Disagreement Timeline",
        OUT["twinkle12_timeline_png"], key_events=TWINKLE12_KEY_EVENTS,
    )
    plot_disagreement_timeline(
        twinkle_result, twinkle_arrays, twinkle_pc,
        "Phase 3C — Twinkle.mid: Pitch-Class vs. Chord-ID Disagreement Timeline",
        OUT["twinkle_timeline_png"], key_events=None,
    )

    print("\nWriting reports...")
    with open(OUT["twinkle_report_md"], "w") as f:
        f.write(build_report_md("Twinkle.mid", twinkle_result))
    with open(OUT["twinkle12_report_md"], "w") as f:
        f.write(build_report_md("Twinkle 12.mid", twinkle12_result))

    combined_metrics = {
        "phase": "phase_3c_pitch_class_vs_chord_id_disagreement",
        "twinkle_mid": twinkle_result,
        "twinkle_12_mid": twinkle12_result,
    }
    with open(OUT["metrics_json"], "w") as f:
        json.dump(combined_metrics, f, indent=2)

    print(f"Wrote {OUT['twinkle_report_md']}")
    print(f"Wrote {OUT['twinkle12_report_md']}")
    print(f"Wrote {OUT['metrics_json']}")

    all_out_paths = [
        OUT["twinkle_report_md"], OUT["twinkle12_report_md"], OUT["metrics_json"],
        OUT["twinkle12_timeline_png"], OUT["twinkle_timeline_png"],
    ] + list(twinkle_saved_paths.values()) + list(twinkle12_saved_paths.values())

    run_verification(combined_metrics, all_out_paths, midi_mtimes_before, prior_outputs_mtimes_before, twinkle12_result)


if __name__ == "__main__":
    main()
