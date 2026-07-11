"""evaluate_phase3g_pitch_class_corpus.py

Phase 3G-A: corpus-aware pitch-class/chroma baseline + uncertainty
evaluation across the full 6-level benchmark ladder (see
../HANDOFF_PHASE3G.md and ../STATUS.md Phase 3F/3F.6/3F.7/3F.8).

This script is new -- it does NOT edit midi_chroma_extraction.py,
pitch_class_baseline.py, evaluate_pitch_class_phase2d.py, or
pitch_class_uncertainty_diagnostics.py in place. Those four modules are
Twinkle-only (hardcoded TWINKLE_MIDI / TWINKLE12_MIDI module constants) and
cannot loop over an arbitrary file list, per HANDOFF_PHASE3G.md's explicit
warning. Instead, this script imports their reusable, already-generic
functions (none of which are hardcoded to any particular file) and loops
them over the six ladder pieces:

  L1 Twinkle.mid                       (C major, full piece)
  L2 Bach Minuet in G Major             (G major, full piece)
  L3 Fur Elise opening excerpt          (A minor, excerpt [0, 54.0]s)
  L4 Chopin Prelude Op. 28 No. 4        (E minor, full piece)
  L5 Clementi Op. 36 No. 1 exposition   (C major -> G major, excerpt [0, 17.3]s)
  L6 Twinkle 12.mid                     (C -> Eb -> C, full piece)

Scope: Stage 1 (pitch-class/chroma fast filter) + Stage 3 (uncertainty
diagnostics) only, per HANDOFF_PHASE3G.md's proposed Phase 3G-A split.
Does NOT run chord-id EMA/SRN, does NOT implement a Chroma SRN, a
Transformer, or any neural refinement (Stage 4 remains deferred per
Phase 3D). Does NOT regenerate any existing Phase 2C/2D/3B/3C plot or
metrics file -- everything here is written under a new PHASE3G_A_ prefix
into 05_Figures_Results/ and a new 03_MIDI_Data/derived_phase3g_corpus/
directory.

Conventions preserved exactly from the frozen Twinkle-only scripts:
  - window_sec = 0.5
  - chroma memory_decay = 0.8 (pitch-class path's constant, not the
    chord-id path's 0.6)
  - threshold_ratio = 0.10
  - SCALE_TEMPLATES from pitch_class_baseline.py (imported, not redefined)
  - top-1 key selection via plain np.argmax, never np.argsort (see
    pitch_class_uncertainty_diagnostics.py's module docstring for the
    ~7% divergence bug this avoids)
  - prediction_times_sec / offset_windows real-time alignment handled
    exactly as established in Phase 2D (leading-silent-window offset) and
    Phase 3B (full, undropped grid with an explicit `active` mask)
  - low_margin is NOT treated as a standalone difficulty signal (it is
    structurally saturated by scale-template ties across relative
    major/minor pairs -- key_switch / large_jump / anchor_mismatch /
    tie_count are used instead, per Phase 3B's finding)
  - Chopin No. 4's genuine compositional silence at t~=95.36-99.64s is
    documented and explicitly NOT treated as corruption or a pipeline bug
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
_CANDIDATE_DIR = os.path.join(_MIDI_DIR, "candidate_intermediate_midi")
_EXCERPT_DIR = os.path.join(_CANDIDATE_DIR, "excerpts")
_DERIVED_CORPUS_DIR = os.path.join(_MIDI_DIR, "derived_phase3g_corpus")
_FIGURES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "05_Figures_Results"))

from shared_music_defs import decode_key, fifth_distance, key_tonic_pc, key_index, FIFTH_POS  # noqa: E402

from pitch_class_baseline import (  # noqa: E402
    midi_to_key_baseline,
    SCALE_TEMPLATES,
    DEFAULT_WINDOW_SEC,
    DEFAULT_MEMORY_DECAY,
)
from midi_chroma_extraction import extract_chroma_sequence, DEFAULT_THRESHOLD_RATIO  # noqa: E402

from pitch_class_uncertainty_diagnostics import (  # noqa: E402
    analyze_piece,
    compute_summary,
    compute_anchor_diagnostics,
    build_difficulty_output,
    _expected_key_id,
    DEFAULT_LOW_MARGIN_THRESHOLD,
    DEFAULT_LARGE_JUMP_THRESHOLD,
)

from evaluate_pitch_class_phase2d import (  # noqa: E402
    compute_descriptive_metrics,
    plot_key_trajectory,
    plot_fifths_walk_single,
    _to_native,
)

assert DEFAULT_WINDOW_SEC == 0.5
assert DEFAULT_MEMORY_DECAY == 0.8
assert DEFAULT_THRESHOLD_RATIO == 0.10


# ---------------------------------------------------------------------------
# Corpus definition
# ---------------------------------------------------------------------------

TWINKLE12_KEY_EVENTS = [
    {"time": 0.0, "key_name": "C Major"},
    {"time": 384.0, "key_name": "Eb Major"},
    {"time": 392.0, "key_name": "Eb Major"},
    {"time": 432.0, "key_name": "C Major"},
    {"time": 440.0, "key_name": "C Major"},
]

CHOPIN_SILENCE_REGION = {
    "start_sec": 95.36,
    "end_sec": 99.64,
    "note": (
        "Genuine, compositionally-intentional silence (Chopin's famous dramatic pause near the "
        "piece's end), confirmed in Phase 3F.8 -- NOT file corruption or a pipeline bug. Expected "
        "to register as an inactive/undefined-key stretch under this workspace's active-window "
        "convention; do not misdiagnose as a baseline failure."
    ),
}


def _static_anchors(anchors):
    return lambda duration_sec: anchors


def _tie_break_bias_diagnostic(analysis, expected_key_name):
    """For anchors whose expected key is minor: checks whether the
    expected key ever wins np.argmax's tie-break against a lower-index
    (i.e. major-key, since SCALE_TEMPLATES/key_index put all 12 major
    keys at indices 0-11 and all 12 minor keys at indices 12-23) template
    it is EXACTLY tied with on score. If it never does, that is a
    structural property of the tie-breaking convention itself (verified
    here, not assumed) -- not just "the model is often wrong under
    ambiguity" but "the model cannot possibly select this minor key on
    any window where it ties a major key," which is a stronger and more
    precise claim. Only meaningful for minor-mode anchors; returns None
    for major-mode anchors."""
    tonic, mode = expected_key_name.split()
    if not mode.lower().startswith("min"):
        return None
    expected_id = _expected_key_id(expected_key_name)
    raw_scores = analysis["raw_scores"]
    active = analysis["active"]
    key_id = analysis["key_id"]
    max_score = raw_scores.max(axis=1)
    expected_score = raw_scores[:, expected_id]
    tied_for_max = active & (max_score > 0) & (np.abs(expected_score - max_score) < 1e-9)
    n_tied_for_max = int(tied_for_max.sum())
    n_selected_when_tied = int((tied_for_max & (key_id == expected_id)).sum())
    return {
        "expected_key": expected_key_name,
        "expected_key_id": int(expected_id),
        "n_active_windows_where_expected_key_ties_for_max_score": n_tied_for_max,
        "n_of_those_where_expected_key_is_actually_selected": n_selected_when_tied,
        "note": (
            "SCALE_TEMPLATES/key_index place all 12 major keys at indices 0-11 and all 12 minor keys at "
            "indices 12-23; np.argmax's leftmost-tie convention therefore always resolves an exact tie in "
            "favor of a tied major key over a tied minor key. If n_of_those_where_expected_key_is_actually_"
            "selected is 0 while n_active_windows_where_expected_key_ties_for_max_score > 0, the baseline is "
            "structurally prevented from ever selecting this minor tonic on those windows -- this is a "
            "verified property of the argmax tie-break convention, not merely noisy/probabilistic ambiguity."
        ),
    }


def _clementi_anchors(duration_sec):
    half = duration_sec / 2.0
    return [
        {
            "name": "approx_first_half",
            "start_sec": None,
            "end_sec": half,
            "expected_key_name": "C Major",
            "confirmed_boundary": False,
        },
        {
            "name": "approx_second_half",
            "start_sec": half,
            "end_sec": None,
            "expected_key_name": "G Major",
            "confirmed_boundary": False,
        },
    ]


PIECES = [
    {
        "level": "L1",
        "stem": "Twinkle",
        "display_name": "Twinkle.mid",
        "midi_path": os.path.join(_MIDI_DIR, "Twinkle.mid"),
        "anchors_fn": _static_anchors(
            [{"name": "full_piece", "start_sec": None, "end_sec": None, "expected_key_name": "C Major", "confirmed_boundary": True}]
        ),
        "key_events": None,
        "silence_region": None,
        "role": "Monophonic sanity check, full piece.",
    },
    {
        "level": "L2",
        "stem": "Bach_Minuet_G",
        "display_name": "Bach — Minuet in G Major, BWV Anh. 114",
        "midi_path": os.path.join(_CANDIDATE_DIR, "J.S. Bach - Minuet in G Major, BWV Anh. 114.mid"),
        "anchors_fn": _static_anchors(
            [{"name": "full_piece", "start_sec": None, "end_sec": None, "expected_key_name": "G Major", "confirmed_boundary": True}]
        ),
        "key_events": None,
        "silence_region": None,
        "role": "Non-C tonic + light accompaniment, full piece.",
    },
    {
        "level": "L3",
        "stem": "FurElise_excerpt",
        "display_name": "Beethoven — Für Elise (opening excerpt, [0.0, 54.0]s)",
        "midi_path": os.path.join(_EXCERPT_DIR, "Fur_Elise_opening_0_54s.mid"),
        "anchors_fn": _static_anchors(
            [{"name": "full_excerpt", "start_sec": None, "end_sec": None, "expected_key_name": "A Minor", "confirmed_boundary": True}]
        ),
        "key_events": None,
        "silence_region": None,
        "role": "Relative-major/minor ambiguity test. Excerpt is the source of truth (per Phase 3F.8), not the full 169.7s piece.",
    },
    {
        "level": "L4",
        "stem": "Chopin_Op28No4",
        "display_name": "Chopin — Prelude in E minor, Op. 28 No. 4",
        "midi_path": os.path.join(_CANDIDATE_DIR, "f-f-chopin-prelude-op-28-no-4.mid"),
        "anchors_fn": _static_anchors(
            [{"name": "full_piece", "start_sec": None, "end_sec": None, "expected_key_name": "E Minor", "confirmed_boundary": True}]
        ),
        "key_events": None,
        "silence_region": CHOPIN_SILENCE_REGION,
        "role": "Slow harmony + chromatic pressure, full piece. Contains a genuine dramatic silence (see silence_region).",
    },
    {
        "level": "L5",
        "stem": "Clementi_excerpt",
        "display_name": "Clementi — Sonatina Op. 36 No. 1, I (exposition, [0.0, 17.3]s)",
        "midi_path": os.path.join(_EXCERPT_DIR, "Clementi_Op36_No1_I_exposition_norepeat_0_17p3s.mid"),
        "anchors_fn": _clementi_anchors,
        "key_events": None,
        "silence_region": None,
        "role": (
            "Short, clear, single modulation (C major -> G major). The exact transition time within the "
            "excerpt is NOT confirmed, so the two anchors below are an approximate first-half/second-half "
            "split, reported descriptively (tonic vs. dominant-region behavior) -- not a scored, confirmed "
            "boundary. Do not overclaim precision from these anchors."
        ),
    },
    {
        "level": "L6",
        "stem": "Twinkle12",
        "display_name": "Twinkle 12.mid (Mozart 12 Variations)",
        "midi_path": os.path.join(_MIDI_DIR, "Twinkle 12.mid"),
        "anchors_fn": _static_anchors(
            [
                {"name": "pre_384s", "start_sec": None, "end_sec": 384.0, "expected_key_name": "C Major", "confirmed_boundary": True},
                {"name": "384_to_432s", "start_sec": 384.0, "end_sec": 432.0, "expected_key_name": "Eb Major", "confirmed_boundary": True},
                {"name": "post_432s", "start_sec": 432.0, "end_sec": None, "expected_key_name": "C Major", "confirmed_boundary": True},
            ]
        ),
        "key_events": TWINKLE12_KEY_EVENTS,
        "silence_region": None,
        "role": "High-stress ornamented variation / modulation stress test, full piece, real embedded key signatures C->Eb->C.",
    },
]

OUT_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3G_A_pitch_class_corpus_metrics.json")
OUT_REPORT_MD = os.path.join(_FIGURES_DIR, "PHASE3G_A_pitch_class_corpus_report.md")


# ---------------------------------------------------------------------------
# Per-piece pipeline
# ---------------------------------------------------------------------------

def _key_run_length_encode(key_id, times):
    """Run-length-encodes the (offset-corrected) baseline key_id sequence
    into contiguous same-key runs, so a piece's actual predicted-key
    trajectory (e.g. C -> G -> C) can be reported precisely instead of
    only via coarse half-split proportions, which can silently hide a
    non-monotonic trajectory."""
    runs = []
    T = len(key_id)
    t = 0
    while t < T:
        start = t
        k = key_id[t]
        while t < T and key_id[t] == k:
            t += 1
        name, mode = decode_key(int(k))
        runs.append({
            "key": f"{name} {mode}", "start_sec": float(times[start]), "end_sec": float(times[t - 1]),
            "n_windows": int(t - start),
        })
    return runs


def save_diagnostic_arrays_corpus(piece_stem, analysis, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    array_fields = [
        "raw_scores", "active", "key_id", "top1_score", "top2_score", "raw_margin",
        "normalized_margin", "entropy", "tie_count", "key_switch", "jump_distance",
        "large_jump", "low_margin", "prediction_indices", "prediction_times_sec",
    ]
    for field in array_fields:
        path = os.path.join(out_dir, f"{piece_stem}_{field}.npy")
        np.save(path, analysis[field])
        paths[field] = path
    return paths


def process_piece(piece):
    midi_path = piece["midi_path"]
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"Missing MIDI file for {piece['level']} ({piece['display_name']}): {midi_path}")

    print(f"\n=== {piece['level']} — {piece['display_name']} ===")

    # --- Chroma extraction (Phase 2C logic, reused function, corpus-local save dir) ---
    raw_chroma, smoothed_chroma, thresholded_chroma, chroma_meta = extract_chroma_sequence(
        midi_path, window_sec=DEFAULT_WINDOW_SEC, memory_decay=DEFAULT_MEMORY_DECAY,
        threshold_ratio=DEFAULT_THRESHOLD_RATIO, return_metadata=True,
    )
    os.makedirs(_DERIVED_CORPUS_DIR, exist_ok=True)
    chroma_paths = {
        "raw": os.path.join(_DERIVED_CORPUS_DIR, f"{piece['stem']}_raw_chroma.npy"),
        "smoothed": os.path.join(_DERIVED_CORPUS_DIR, f"{piece['stem']}_smoothed_chroma_decay08.npy"),
        "thresholded": os.path.join(_DERIVED_CORPUS_DIR, f"{piece['stem']}_thresholded_smoothed_chroma_decay08.npy"),
        "metadata": os.path.join(_DERIVED_CORPUS_DIR, f"{piece['stem']}_chroma_metadata.json"),
    }
    np.save(chroma_paths["raw"], raw_chroma)
    np.save(chroma_paths["smoothed"], smoothed_chroma)
    np.save(chroma_paths["thresholded"], thresholded_chroma)
    with open(chroma_paths["metadata"], "w") as f:
        json.dump(chroma_meta, f, indent=2)
    print(f"  chroma: duration={chroma_meta['duration_sec']:.2f}s windows={raw_chroma.shape[0]}")

    duration_sec = chroma_meta["duration_sec"]

    # --- Pitch-class baseline (Phase 2B logic, reused function): primary
    # predicted key sequence, with the leading-silent-window offset applied
    # exactly as established in Phase 2D. Used for the trajectory/circle-of-
    # fifths plots and dominant-key descriptive stats. ---
    baseline_key_ids, baseline_meta = midi_to_key_baseline(
        midi_path, window_sec=DEFAULT_WINDOW_SEC, memory_decay=DEFAULT_MEMORY_DECAY, return_metadata=True,
    )
    offset_windows = baseline_meta["n_chroma_windows"] - baseline_meta["n_key_predictions"]
    print(f"  baseline: n_predictions={len(baseline_key_ids)} offset_windows={offset_windows}")

    descriptive = compute_descriptive_metrics(baseline_key_ids)

    # --- Uncertainty diagnostics (Phase 3B logic, reused function): full,
    # undropped grid, np.argmax top-1, explicit `active` mask, no offset
    # arithmetic needed (prediction_times_sec = index * window_sec always). ---
    analysis = analyze_piece(
        thresholded_chroma, DEFAULT_WINDOW_SEC,
        low_margin_threshold=DEFAULT_LOW_MARGIN_THRESHOLD, large_jump_threshold=DEFAULT_LARGE_JUMP_THRESHOLD,
    )

    anchors_spec = piece["anchors_fn"](duration_sec)
    anchor_results = compute_anchor_diagnostics(analysis, anchors_spec)

    npy_paths = save_diagnostic_arrays_corpus(piece["stem"], analysis, _DERIVED_CORPUS_DIR)
    difficulty_out = build_difficulty_output(piece["display_name"], analysis, anchor_results, chroma_meta, npy_paths)

    summary = compute_summary(analysis)

    # --- Inactive/silent window handling: report explicitly, and for
    # Chopin, tie the known compositional silence region to its inactive
    # windows so it is never mistaken for a pipeline failure. ---
    inactive_report = {
        "n_inactive_windows": int((~analysis["active"]).sum()),
        "inactive_fraction": float((~analysis["active"]).mean()) if analysis["T"] > 0 else None,
    }
    if piece["silence_region"] is not None:
        times = analysis["prediction_times_sec"]
        sil_mask = (times >= piece["silence_region"]["start_sec"]) & (times <= piece["silence_region"]["end_sec"])
        sil_idxs = np.where(sil_mask)[0]
        inactive_report["known_silence_region"] = dict(piece["silence_region"])
        inactive_report["known_silence_region"]["n_windows_in_region"] = int(len(sil_idxs))
        inactive_report["known_silence_region"]["n_inactive_windows_in_region"] = int((~analysis["active"][sil_idxs]).sum()) if len(sil_idxs) else 0

    # --- Plots (Phase 2D logic, reused functions) ---
    traj_png = os.path.join(_FIGURES_DIR, f"PHASE3G_A_{piece['stem']}_key_trajectory.png")
    plot_key_trajectory(
        baseline_key_ids, DEFAULT_WINDOW_SEC,
        title=f"Phase 3G-A — Pitch-Class Baseline: {piece['display_name']} Key Trajectory",
        out_path=traj_png,
        key_events=piece["key_events"],
        offset_windows=offset_windows,
    )
    print(f"  wrote {traj_png}")

    fifths_png = os.path.join(_FIGURES_DIR, f"PHASE3G_A_{piece['stem']}_circle_of_fifths.png")
    circle_readable = len(baseline_key_ids) > 0
    if circle_readable:
        plot_fifths_walk_single(
            baseline_key_ids,
            f"Phase 3G-A — Pitch-Class Baseline\n{piece['display_name']} Circle of Fifths Walk",
            fifths_png,
        )
        print(f"  wrote {fifths_png}")
    else:
        fifths_png = None

    tie_summary = {
        "mean_tie_count_active_windows": summary["mean_tie_count_active_windows"],
        "max_tie_count_active_windows": summary["max_tie_count_active_windows"],
    }

    anchor_mismatch_summary = {
        name: {
            "expected_key": a["expected_key"],
            "confirmed_boundary": next((sp.get("confirmed_boundary") for sp in anchors_spec if sp["name"] == name), None),
            "n_predictions": a["n_predictions"],
            "proportion_expected_key": a["proportion_expected_key"],
            "mismatch_count": a["mismatch_count"],
            "n_mismatch_intervals": len(a["mismatch_intervals"]),
            "tie_break_bias": _tie_break_bias_diagnostic(analysis, a["expected_key"]),
        }
        for name, a in anchor_results.items()
    }

    result = {
        "level": piece["level"],
        "display_name": piece["display_name"],
        "midi_path": os.path.relpath(midi_path, os.path.join(_THIS_DIR, "..")),
        "role": piece["role"],
        "settings": {
            "window_sec": DEFAULT_WINDOW_SEC,
            "memory_decay": DEFAULT_MEMORY_DECAY,
            "threshold_ratio": DEFAULT_THRESHOLD_RATIO,
        },
        "duration_sec": duration_sec,
        "n_chroma_windows": int(raw_chroma.shape[0]),
        "baseline_offset_windows": int(offset_windows),
        "predicted_key_sequence": {
            "n_predictions": descriptive["n_timesteps"],
            "dominant_predicted_keys": descriptive["top_predicted_keys"],
            "n_unique_predicted_keys": descriptive["n_unique_predicted_keys"],
            "predicted_key_runs": _key_run_length_encode(
                baseline_key_ids, (np.arange(len(baseline_key_ids)) + offset_windows) * DEFAULT_WINDOW_SEC
            ),
        },
        "fifths_jump_stats_baseline_sequence": descriptive["fifths_jump_stats"],
        "uncertainty_summary": summary,
        "tie_count_summary": tie_summary,
        "anchor_mismatch_summary": anchor_mismatch_summary,
        "anchor_diagnostics_full": {k: v for k, v in anchor_results.items()},
        "inactive_window_handling": inactive_report,
        "difficulty_criterion_counts": difficulty_out["difficulty_criterion_counts"],
        "difficulty_criterion_proportions": difficulty_out["difficulty_criterion_proportions"],
        "n_difficult_intervals": difficulty_out["n_difficult_intervals"],
        "outputs": {
            "chroma": {k: os.path.relpath(v, os.path.join(_THIS_DIR, "..")) for k, v in chroma_paths.items()},
            "diagnostic_arrays": {k: os.path.relpath(v, os.path.join(_THIS_DIR, "..")) for k, v in npy_paths.items()},
            "key_trajectory_png": os.path.relpath(traj_png, os.path.join(_THIS_DIR, "..")),
            "circle_of_fifths_png": os.path.relpath(fifths_png, os.path.join(_THIS_DIR, "..")) if fifths_png else None,
        },
    }
    return _to_native(result), analysis


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt(x, digits=4):
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _piece_section_md(r):
    lines = []
    lines.append(f"## {r['level']} — {r['display_name']}")
    lines.append("")
    lines.append(f"- Role: {r['role']}")
    lines.append(f"- MIDI: `{r['midi_path']}`")
    lines.append(f"- Duration: {r['duration_sec']:.2f}s, {r['n_chroma_windows']} chroma windows, baseline_offset_windows={r['baseline_offset_windows']}")
    lines.append("")

    pks = r["predicted_key_sequence"]
    lines.append(f"**Predicted key sequence:** {pks['n_predictions']} predictions, {pks['n_unique_predicted_keys']} unique keys.")
    lines.append("Dominant predicted keys: " + ", ".join(f"{tk['key']} ({tk['fraction']:.1%})" for tk in pks["dominant_predicted_keys"]))
    lines.append("")

    fj = r["fifths_jump_stats_baseline_sequence"]
    lines.append(
        f"**Circle-of-Fifths jumps (baseline sequence):** mean={_fmt(fj.get('mean_jump'), 2)}, "
        f"max={_fmt(fj.get('max_jump'), 2)}, large jumps (>=3): {fj.get('large_jump_count', 'n/a')} "
        f"({_fmt(fj.get('large_jump_fraction'))} of transitions)"
    )
    lines.append("")

    s = r["uncertainty_summary"]
    lines.append(
        f"**Uncertainty summary (full grid):** {s['n_predictions']} predictions, {s['n_active_windows']} active "
        f"({_fmt(s['active_fraction'])}); {s['n_unique_predicted_keys']} unique keys "
        f"({', '.join(s['unique_predicted_keys'])}); key switches {s['n_key_switches']}/{s['n_key_switch_eligible_transitions']} "
        f"({_fmt(s['key_switch_proportion'])}); large jumps {s['n_large_jumps']} ({_fmt(s['large_jump_proportion'])})."
    )
    lines.append("")

    tc = r["tie_count_summary"]
    lines.append(
        f"**tie_count summary:** mean={_fmt(tc['mean_tie_count_active_windows'], 2)} tied keys/active window, "
        f"max={tc['max_tie_count_active_windows']}. (Structurally saturated by relative major/minor scale-template "
        "ties, per Phase 3B -- `low_margin` alone is not treated as a difficulty signal here.)"
    )
    lines.append("")

    lines.append("**Anchor mismatch summary:**")
    lines.append("")
    for name, a in r["anchor_mismatch_summary"].items():
        conf = "confirmed" if a["confirmed_boundary"] else "approximate/unconfirmed"
        lines.append(
            f"- `{name}` (expected {a['expected_key']}, {conf} window): n={a['n_predictions']}, "
            f"proportion_expected_key={_fmt(a['proportion_expected_key'])}, "
            f"mismatch_count={a['mismatch_count']} across {a['n_mismatch_intervals']} interval(s)"
        )
        tb = a.get("tie_break_bias")
        if tb is not None:
            lines.append(
                f"  - tie-break check: expected key exactly tied for the max scale-template score in "
                f"{tb['n_active_windows_where_expected_key_ties_for_max_score']} active window(s); actually "
                f"selected in {tb['n_of_those_where_expected_key_is_actually_selected']} of those. "
                + ("**Structurally impossible for the expected minor key to win these ties** (major-key "
                   "indices 0-11 always beat tied minor-key indices 12-23 under `np.argmax`'s leftmost-tie "
                   "rule) -- verified, not assumed."
                   if tb['n_active_windows_where_expected_key_ties_for_max_score'] > 0 and tb['n_of_those_where_expected_key_is_actually_selected'] == 0
                   else "")
            )
    lines.append("")

    if r["level"] == "L5":
        runs = r["predicted_key_sequence"]["predicted_key_runs"]
        lines.append(
            "**Full predicted-key run sequence** (not just the half-split proportions above -- shows whether the "
            "trajectory is a clean monotonic modulation or something messier):"
        )
        lines.append("")
        for run in runs:
            lines.append(f"- {run['key']}: t={run['start_sec']:.2f}-{run['end_sec']:.2f}s ({run['n_windows']} windows)")
        lines.append("")

    iw = r["inactive_window_handling"]
    lines.append(f"**Inactive/silent window handling:** {iw['n_inactive_windows']} inactive windows ({_fmt(iw['inactive_fraction'])} of grid).")
    if "known_silence_region" in iw:
        ksr = iw["known_silence_region"]
        lines.append(
            f"  Known compositional silence region t={ksr['start_sec']:.2f}-{ksr['end_sec']:.2f}s: "
            f"{ksr['n_inactive_windows_in_region']}/{ksr['n_windows_in_region']} windows inactive there. {ksr['note']}"
        )
    lines.append("")

    lines.append(f"Outputs: `{r['outputs']['key_trajectory_png']}`" + (f", `{r['outputs']['circle_of_fifths_png']}`" if r['outputs']['circle_of_fifths_png'] else " (circle-of-fifths not generated: empty prediction sequence)"))
    lines.append("")
    return lines


def build_answers_md(results_by_level):
    l1, l2, l3, l4, l5, l6 = (results_by_level[k] for k in ["L1", "L2", "L3", "L4", "L5", "L6"])

    lines = []
    lines.append("## Cross-piece findings")
    lines.append("")

    # Q1: generalize beyond C major?
    l2_anchor = l2["anchor_mismatch_summary"]["full_piece"]
    l2_prop = l2_anchor["proportion_expected_key"]
    q1 = (
        f"**1. Does the pitch-class baseline generalize beyond C major?** On L2 (Bach, G major, full piece), "
        f"proportion_expected_key (G Major) = {_fmt(l2_prop)}. "
        + ("The baseline tracks a non-C tonic well, extending Phase 2D's C-major-only evidence to a different tonic."
           if (l2_prop or 0) > 0.5 else
           "The baseline does not reliably recover a non-C tonic even on a clean, lightly-accompanied piece, "
           "suggesting Phase 2D's Twinkle.mid result may have been C-major-specific rather than evidence of general "
           "tonic recovery.")
    )
    lines.append(q1)
    lines.append("")

    # Q2: minor-key ambiguity
    l3_anchor = l3["anchor_mismatch_summary"]["full_excerpt"]
    l3_prop = l3_anchor["proportion_expected_key"]
    l3_tb = l3_anchor.get("tie_break_bias")
    l4_anchor = l4["anchor_mismatch_summary"]["full_piece"]
    l4_prop = l4_anchor["proportion_expected_key"]
    l4_tb = l4_anchor.get("tie_break_bias")

    def _tb_clause(tb):
        if tb is None:
            return ""
        if tb["n_active_windows_where_expected_key_ties_for_max_score"] > 0 and tb["n_of_those_where_expected_key_is_actually_selected"] == 0:
            return (
                f" Verified structurally, not just observed: the expected minor key exactly tied the max "
                f"scale-template score in {tb['n_active_windows_where_expected_key_ties_for_max_score']} active "
                "window(s), and was selected in 0 of them -- because major-key indices (0-11) always beat a "
                "tied minor-key index (12-23) under `np.argmax`'s leftmost-tie rule, the baseline is "
                "structurally incapable of ever choosing this minor tonic on a tied window, independent of "
                "how much real evidence favors it."
            )
        return ""

    q2 = (
        f"**2. Does it fail on minor-key pieces due to relative major/minor ambiguity?** On L3 (Für Elise excerpt, "
        f"A minor), proportion_expected_key = {_fmt(l3_prop)}.{_tb_clause(l3_tb)} On L4 (Chopin No. 4, E minor), "
        f"proportion_expected_key = {_fmt(l4_prop)}.{_tb_clause(l4_tb)} "
        + ("Both minor-key pieces show a proportion_expected_key of exactly 0.0 -- the baseline never once "
           "predicts the true minor tonic for either piece, always resolving to a relative-major or other "
           "major-key candidate instead. This goes beyond Phase 3B's structural observation that scale-template "
           "scores tie across relative major/minor pairs: the tie-break checks above confirm the resolution is "
           "not close-but-wrong noise, it is a deterministic consequence of `np.argmax` + the major-keys-first "
           "index ordering, which makes any tied minor key categorically unselectable."
           if (l3_prop or 0) == 0.0 and (l4_prop or 0) == 0.0 else
           "Both minor-key pieces show low-but-nonzero expected-key proportions -- see the per-piece tie-break "
           "diagnostics above for how much of this is attributable to the argmax tie-break convention itself "
           "vs. genuine evidence favoring the relative major.")
    )
    lines.append(q2)
    lines.append("")

    # Q3: Clementi single-modulation
    l5_runs = l5["predicted_key_sequence"]["predicted_key_runs"]
    l5_run_desc = " -> ".join(f"{run['key']} ({run['start_sec']:.1f}-{run['end_sec']:.1f}s)" for run in l5_runs)
    l5_keys_visited = [run["key"] for run in l5_runs]  # already consecutive-deduplicated by run-length encoding
    is_monotonic_c_to_g = l5_keys_visited == ["C maj", "G maj"]
    q3 = (
        f"**3. Does Clementi show a usable single-modulation challenge?** The exact transition time within the "
        f"excerpt is not confirmed. The actual predicted-key run sequence (not just coarse half-split "
        f"proportions) is: {l5_run_desc}. "
        + ("This is a clean, monotonic C major -> G major shift with no reversion -- a genuinely usable single-"
           "modulation test case once the exact boundary is independently confirmed."
           if is_monotonic_c_to_g else
           "This is **not** a clean monotonic C major -> G major modulation -- the baseline oscillates back to C "
           "major after visiting G major, rather than settling on the dominant key. Real dominant-key "
           "responsiveness is present (G major is reached), but the excerpt (at this window/threshold setting) "
           "does not cleanly isolate a single one-way modulation the way a simple before/after anchor split "
           "would assume. Any future use of Clementi as a single-modulation benchmark should account for this "
           "oscillation rather than treating the coarse half-split proportions reported above at face value.")
    )
    lines.append(q3)
    lines.append("")

    # Q4: local Stage 1 failures for Stage 4
    candidates = []
    for lvl, r in results_by_level.items():
        s = r["uncertainty_summary"]
        if (s.get("large_jump_proportion") or 0) > 0 or (s.get("key_switch_proportion") or 0) > 0:
            candidates.append(f"{lvl} ({r['display_name']}: key_switch_proportion={_fmt(s.get('key_switch_proportion'))}, large_jump_proportion={_fmt(s.get('large_jump_proportion'))})")
    q4 = (
        "**4. Which pieces create local Stage 1 failures suitable for later Stage 4?** "
        + ("Pieces with nonzero key-switch or large-jump activity: " + "; ".join(candidates) + ". "
           "These, together with any anchor_mismatch intervals reported per piece above, are the natural first "
           "candidates for a future targeted local refinement, per Phase 3D's original recommendation to look for "
           "intermediate-difficulty examples rather than the two globally-easy/globally-biased original pieces."
           if candidates else
           "No piece in this corpus shows nonzero key-switch or large-jump proportions on the uncertainty grid; "
           "any anchor mismatches reported per piece above are the only candidate local-failure regions.")
    )
    lines.append(q4)
    lines.append("")

    # Q5: Stage 4 still deferred?
    q5 = (
        "**5. Should Stage 4 remain deferred after Phase 3G-A?** This script performs Stage 1 + Stage 3 only "
        "(pitch-class baseline + uncertainty diagnostics) -- no chord-id EMA/SRN disagreement comparison "
        "(Phase 3C-style) has been run on this new corpus, so the question of whether disagreement with a "
        "recurrent chord-id model is local or global (Phase 3D's actual deciding factor for the original two "
        "pieces) remains unanswered here. Per HANDOFF_PHASE3G.md's explicit scope, this task does not implement, "
        "and does not recommend implementing, any Chroma SRN, Transformer, or neural refinement. Whether Stage 4 "
        "should be reconsidered is a Phase 3G-B/3H question, contingent on running that disagreement comparison "
        "on the newly-identified candidate regions from finding 4 above."
    )
    lines.append(q5)
    lines.append("")

    return lines


def build_report_md(results_by_level):
    lines = []
    lines.append("# Phase 3G-A — Corpus-Aware Pitch-Class Baseline & Uncertainty Evaluation")
    lines.append("")
    lines.append(
        "Extends the Phase 2C (chroma extraction) -> Phase 2B/2D (pitch-class/scale-template baseline) -> "
        "Phase 3B (uncertainty diagnostics) pipeline to the full 6-level benchmark ladder, using a new, "
        "corpus-aware script (`evaluate_phase3g_pitch_class_corpus.py`) that reuses (imports, does not modify) "
        "the exact logic, formulas, and conventions of the frozen Twinkle-only scripts. **Stage 1 + Stage 3 "
        "only** -- no chord-id EMA/SRN comparison, no Chroma SRN, no Transformer, no neural refinement of any kind."
    )
    lines.append("")
    lines.append("## Settings")
    lines.append("")
    lines.append(f"- window_sec: {DEFAULT_WINDOW_SEC}")
    lines.append(f"- chroma memory_decay: {DEFAULT_MEMORY_DECAY}")
    lines.append(f"- threshold_ratio: {DEFAULT_THRESHOLD_RATIO}")
    lines.append("- SCALE_TEMPLATES: imported from `pitch_class_baseline.py` (24x12 full major/natural-minor scales), not redefined")
    lines.append("- top-1 key selection: plain `np.argmax` (never `np.argsort`), matching Phase 2B/3B exactly")
    lines.append(
        "- `low_margin` is not used as a standalone difficulty signal (structurally saturated by relative "
        "major/minor scale-template ties, per Phase 3B) -- `key_switch`, `large_jump`, `anchor_mismatch`, and "
        "`tie_count` are used instead"
    )
    lines.append("")

    for lvl in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        lines.extend(_piece_section_md(results_by_level[lvl]))

    lines.extend(build_answers_md(results_by_level))

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This is Phase 3G-A only: pitch-class/chroma baseline evaluation + uncertainty diagnostics on the full "
        "corpus. No chord-id EMA/SRN comparison, no Chroma SRN, no Transformer, and no neural refinement has been "
        "implemented. No existing Phase 1/1.5/2/3 script was modified, and no existing Phase 1/1.5/2/3 output was "
        "regenerated -- all outputs here are new, under a `PHASE3G_A_` prefix or in a new "
        "`03_MIDI_Data/derived_phase3g_corpus/` directory."
    )
    lines.append("")

    return "\n".join(lines)


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


def run_verification(results_by_level, all_out_paths, midi_mtimes_before, prior_outputs_mtimes_before):
    checks = []

    for p in all_out_paths:
        checks.append((f"{os.path.relpath(p, _MIDI_DIR + '/..')} exists", os.path.exists(p)))
        checks.append((f"{os.path.relpath(p, _MIDI_DIR + '/..')} is non-empty", os.path.exists(p) and os.path.getsize(p) > 0))

    nan_paths = _scan_for_nan(results_by_level)
    checks.append(("no NaNs in corpus metrics", len(nan_paths) == 0))

    for lvl, r in results_by_level.items():
        checks.append((f"{lvl} has >0 chroma windows", r["n_chroma_windows"] > 0))
        checks.append((f"{lvl} has >0 predicted keys", r["predicted_key_sequence"]["n_predictions"] > 0))

    for path, mtime_before in midi_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} MIDI file unchanged", os.path.exists(path) and os.path.getmtime(path) == mtime_before))

    for path, mtime_before in prior_outputs_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} prior Phase output unchanged", os.path.exists(path) and os.path.getmtime(path) == mtime_before))

    print()
    print("evaluate_phase3g_pitch_class_corpus.py verification")
    print("-" * 60)
    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")
    print("-" * 60)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    if nan_paths:
        print("NaN found at:", nan_paths)
    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    os.makedirs(_DERIVED_CORPUS_DIR, exist_ok=True)

    midi_mtimes_before = {p["midi_path"]: os.path.getmtime(p["midi_path"]) for p in PIECES if os.path.exists(p["midi_path"])}

    prior_outputs_sample = [
        os.path.join(_FIGURES_DIR, "PHASE2D_pitch_class_baseline_metrics.json"),
        os.path.join(_FIGURES_DIR, "PHASE3B_pitch_class_uncertainty_metrics.json"),
        os.path.join(_FIGURES_DIR, "PHASE3C_disagreement_metrics.json"),
        os.path.join(_FIGURES_DIR, "PHASE2D_Twinkle_pitch_class_key_trajectory.png"),
    ]
    prior_outputs_mtimes_before = {p: os.path.getmtime(p) for p in prior_outputs_sample if os.path.exists(p)}

    results_by_level = {}
    out_paths = []
    for piece in PIECES:
        result, _analysis = process_piece(piece)
        results_by_level[piece["level"]] = result
        out_paths.append(os.path.join(_FIGURES_DIR, os.path.basename(result["outputs"]["key_trajectory_png"])))
        if result["outputs"]["circle_of_fifths_png"]:
            out_paths.append(os.path.join(_FIGURES_DIR, os.path.basename(result["outputs"]["circle_of_fifths_png"])))
        for v in result["outputs"]["chroma"].values():
            out_paths.append(os.path.join(_THIS_DIR, "..", v))
        for v in result["outputs"]["diagnostic_arrays"].values():
            out_paths.append(os.path.join(_THIS_DIR, "..", v))

    metrics_json_content = {
        "phase": "phase_3g_a_pitch_class_corpus_evaluation",
        "settings": {
            "window_sec": DEFAULT_WINDOW_SEC,
            "memory_decay": DEFAULT_MEMORY_DECAY,
            "threshold_ratio": DEFAULT_THRESHOLD_RATIO,
        },
        "pieces": results_by_level,
        "notes": (
            "Phase 3G-A: corpus-aware pitch-class baseline + uncertainty diagnostics, Stage 1 + Stage 3 only. "
            "No chord-id EMA/SRN comparison, no Chroma SRN, no Transformer, no neural refinement. Reuses "
            "(imports, does not modify) pitch_class_baseline.py, midi_chroma_extraction.py, "
            "evaluate_pitch_class_phase2d.py, and pitch_class_uncertainty_diagnostics.py."
        ),
    }
    with open(OUT_METRICS_JSON, "w") as f:
        json.dump(_to_native(metrics_json_content), f, indent=2)
    print(f"\nWrote {OUT_METRICS_JSON}")

    report_md = build_report_md(results_by_level)
    with open(OUT_REPORT_MD, "w") as f:
        f.write(report_md)
    print(f"Wrote {OUT_REPORT_MD}")

    out_paths.extend([OUT_METRICS_JSON, OUT_REPORT_MD])
    run_verification(results_by_level, out_paths, midi_mtimes_before, prior_outputs_mtimes_before)


if __name__ == "__main__":
    main()
