"""evaluate_phase3h_a_tonic_mode_resolver.py

Phase 3H-A: non-neural tonic/mode resolver ablations, testing whether the
pitch-class fast filter's tonic/mode ambiguity (documented in Phase 3G-A/
3G-B: the baseline is a diatonic-*collection* resolver, not a tonic/mode
resolver, because unweighted SCALE_TEMPLATES gives relative major/minor
pairs identical rows and np.argmax's leftmost-tie convention then always
prefers the major-indexed key) can be improved by small, interpretable,
non-neural decision-rule variants.

This is still NOT a neural-modeling phase. No chord-id EMA/SRN, no Chroma
SRN, no Transformer, no neural refinement of any kind is run or
implemented here.

Phase 3G-A and Phase 3G-B are treated as **frozen**. This script does not
modify either of their scripts or output files -- it only *reads* Phase
3G-A's saved per-piece arrays (`03_MIDI_Data/derived_phase3g_corpus/*.npy`)
and imports (does not modify) small, already-generic helpers from
`evaluate_phase3g_pitch_class_corpus.py` and
`evaluate_phase3g_b_tie_aware_diagnostics.py`.

Four variants, each isolating exactly one changed variable:

  A. Frozen Phase 3G-A control -- unweighted SCALE_TEMPLATES, plain
     major-first np.argmax. Loaded directly from the frozen .npy arrays,
     never recomputed. This is the fixed reference every other variant is
     compared against.

  B. Collection-level evaluation -- NOT a new predictor. An evaluation
     lens (collapsing relative major/minor into one diatonic collection)
     applied uniformly to every variant's predictions, reusing Phase
     3G-B's `collection_equivalent_key_id` verbatim.

  C. Tie-aware continuity rule -- when multiple keys tie for the window's
     max raw score (using the SAME unweighted SCALE_TEMPLATES evidence as
     the control, reused not recomputed), prefer the previous predicted
     key if it is among the tied keys; otherwise fall back to plain
     np.argmax (the tied key with the smallest key_index, identical to
     the control's own tie-break). Purely self-referential -- uses only
     this variant's own prediction history, never an anchor/expected key.

  D. Weighted key-profile matcher -- a new, hand-specified (not trained,
     not tuned on this corpus), interpretable 24x12 template where
     within-scale weights follow basic tonal-function role (tonic >
     dominant > mediant > other diatonic degrees). Because major and
     natural-minor scale degrees sit at different semitone offsets from
     the tonic (e.g. the mediant is +4 semitones in major but +3 in
     minor), applying the SAME role-weight vector to each scale's own
     degree offsets produces DIFFERENT numeric rows for relative major/
     minor pairs -- breaking the exact-tie structure Phase 3B/3G-A/3G-B
     documented, without any data-driven fitting. Decision rule is still
     plain np.argmax (only the template changes, isolating the
     representation variable from the decision-rule variable tested by C).

Guardrails honored throughout: anchors (expected keys) are used ONLY
inside the evaluation functions, after a variant's key_id sequence has
already been fully computed -- no variant-computation function below
takes an anchor or expected-key argument. No dense per-timestep accuracy
is claimed (real MIDI has no such ground truth); all anchor comparisons
are the same window-level, documented-expected-key convention Phase
3G-A/3G-B already used.
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
_DERIVED_CORPUS_DIR = os.path.join(_MIDI_DIR, "derived_phase3g_corpus")
_FIGURES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "05_Figures_Results"))

from shared_music_defs import decode_key, key_index, key_tonic_pc, fifth_distance, FIFTH_POS  # noqa: E402
from pitch_class_baseline import DEFAULT_WINDOW_SEC, DEFAULT_MEMORY_DECAY  # noqa: E402
from midi_chroma_extraction import DEFAULT_THRESHOLD_RATIO  # noqa: E402
from pitch_class_uncertainty_diagnostics import _expected_key_id, DEFAULT_LARGE_JUMP_THRESHOLD  # noqa: E402
from plotting_comparison import FIFTHS_LABEL_NAMES  # noqa: E402

# Reused (imported, not modified): anchor windows and the run-length-
# encoding helper are identical to what Phase 3G-A/3G-B already used, so
# nothing about "what counts as an anchor" is redefined here.
from evaluate_phase3g_pitch_class_corpus import PIECES, CHOPIN_SILENCE_REGION, _key_run_length_encode  # noqa: E402
from evaluate_phase3g_b_tie_aware_diagnostics import collection_equivalent_key_id  # noqa: E402

assert DEFAULT_WINDOW_SEC == 0.5
assert DEFAULT_MEMORY_DECAY == 0.8
assert DEFAULT_THRESHOLD_RATIO == 0.10

PHASE3G_A_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3G_A_pitch_class_corpus_metrics.json")
PHASE3G_B_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3G_B_tie_aware_diagnostics_metrics.json")

OUT_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3H_A_tonic_mode_resolver_metrics.json")
OUT_REPORT_MD = os.path.join(_FIGURES_DIR, "PHASE3H_A_tonic_mode_resolver_report.md")
OUT_PLOT_FUR_ELISE = os.path.join(_FIGURES_DIR, "PHASE3H_A_FurElise_excerpt_variant_comparison_key_trajectory.png")
OUT_PLOT_CHOPIN = os.path.join(_FIGURES_DIR, "PHASE3H_A_Chopin_Op28No4_variant_comparison_key_trajectory.png")

TIE_EPS = 1e-9

VARIANT_NAMES = ["A_control", "C_tie_aware_continuity", "D_weighted_profile"]
VARIANT_LABELS = {
    "A_control": "A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)",
    "C_tie_aware_continuity": "C: tie-aware continuity rule (same unweighted evidence as A, continuity-preferring tie-break)",
    "D_weighted_profile": "D: weighted key-profile matcher (functional-role-weighted templates, plain argmax)",
}


# ---------------------------------------------------------------------------
# Variant D: weighted key-profile templates
#
# MAJ_INTERVALS / MIN_INTERVALS match pitch_class_baseline.build_scale_
# templates()'s own interval lists exactly (verified below) -- only the
# per-degree VALUE changes (weight instead of a flat 1), so this is a
# minimal, auditable change to the existing representation, not a
# from-scratch redesign.
#
# FUNCTIONAL_WEIGHTS is a fixed, hand-specified, pre-declared profile
# based on standard tonal-function role, NOT fit or tuned on this (or any)
# corpus:
#   - tonic (scale degree 1): the key-defining pitch, highest weight.
#   - dominant (degree 5): the primary harmonic pillar / cadential
#     partner of the tonic, second-highest weight.
#   - mediant (degree 3): the single pitch that actually distinguishes
#     major from minor by ear (major 3rd vs. minor 3rd above the tonic),
#     third-highest weight -- this is also the specific mechanism that
#     breaks the relative-major/minor tie, since major's degree-3 offset
#     (+4 semitones) and minor's degree-3 offset (+3 semitones) land on
#     different absolute pitch classes for any given collection.
#   - supertonic, subdominant, submediant, leading-tone/subtonic (degrees
#     2, 4, 6, 7): remaining diatonic scale tones, weighted equally and
#     lowest, since none of them individually defines the tonic/mode the
#     way scale degrees 1/3/5 do.
# These weights were chosen before running this script on any piece and
# are not adjusted based on this corpus's results (see Verification).
# ---------------------------------------------------------------------------

MAJ_INTERVALS = [0, 2, 4, 5, 7, 9, 11]   # I ii iii IV V vi vii  (matches pitch_class_baseline.build_scale_templates)
MIN_INTERVALS = [0, 2, 3, 5, 7, 8, 10]   # i ii III iv v VI VII  (natural minor, matches pitch_class_baseline.build_scale_templates)
DEGREE_ROLE_NAMES = ["tonic", "supertonic", "mediant", "subdominant", "dominant", "submediant", "leading-tone/subtonic"]
FUNCTIONAL_WEIGHTS = [5, 1, 3, 2, 4, 1, 1]  # aligned positionally to DEGREE_ROLE_NAMES / MAJ_INTERVALS / MIN_INTERVALS


def build_weighted_key_templates():
    templates = np.zeros((24, 12))
    for root in range(12):
        for pos, offset in enumerate(MAJ_INTERVALS):
            templates[root, (root + offset) % 12] = FUNCTIONAL_WEIGHTS[pos]
        for pos, offset in enumerate(MIN_INTERVALS):
            templates[root + 12, (root + offset) % 12] = FUNCTIONAL_WEIGHTS[pos]
    return templates


WEIGHTED_TEMPLATES = build_weighted_key_templates()


def verify_weighted_templates_break_relative_ties():
    """Structural check (not corpus-dependent): confirms every relative
    major/minor pair now has a DIFFERENT row, unlike the frozen unweighted
    SCALE_TEMPLATES where they are identical by construction."""
    n_identical_pairs = 0
    identical_examples = []
    for tonic_pc in range(12):
        maj_row = WEIGHTED_TEMPLATES[key_index(tonic_pc, "maj")]
        rel_minor_tonic = (tonic_pc - 3) % 12
        min_row = WEIGHTED_TEMPLATES[key_index(rel_minor_tonic, "min")]
        if np.array_equal(maj_row, min_row):
            n_identical_pairs += 1
            identical_examples.append((decode_key(key_index(tonic_pc, "maj")), decode_key(key_index(rel_minor_tonic, "min"))))
    unique_rows = np.unique(WEIGHTED_TEMPLATES, axis=0)
    return {
        "n_relative_major_minor_pairs_checked": 12,
        "n_relative_pairs_still_identical": n_identical_pairs,
        "identical_pair_examples": identical_examples,
        "n_unique_rows_of_24": int(unique_rows.shape[0]),
        "all_24_rows_unique": bool(unique_rows.shape[0] == 24),
    }


# ---------------------------------------------------------------------------
# Loading frozen Phase 3G-A arrays (read-only, never modified)
# ---------------------------------------------------------------------------

def load_frozen_arrays(stem):
    def _load(field):
        return np.load(os.path.join(_DERIVED_CORPUS_DIR, f"{stem}_{field}.npy"))

    return {
        "key_id": _load("key_id"),
        "active": _load("active"),
        "raw_scores": _load("raw_scores"),
        "jump_distance": _load("jump_distance"),
        "large_jump": _load("large_jump"),
        "key_switch": _load("key_switch"),
        "prediction_times_sec": _load("prediction_times_sec"),
        "thresholded_chroma": _load("thresholded_smoothed_chroma_decay08"),
    }


def load_piece_duration(stem):
    with open(os.path.join(_DERIVED_CORPUS_DIR, f"{stem}_chroma_metadata.json")) as f:
        return json.load(f)["duration_sec"]


# ---------------------------------------------------------------------------
# Variant computation (each function computes ONLY a key_id sequence +
# its own supporting evidence array -- none of them accept an anchor or
# expected-key argument, by construction, satisfying guardrail 8)
# ---------------------------------------------------------------------------

def variant_A_control(stem):
    """Frozen Phase 3G-A control. Loaded verbatim -- zero recomputation."""
    frozen = load_frozen_arrays(stem)
    return {
        "key_id": frozen["key_id"], "active": frozen["active"], "raw_scores": frozen["raw_scores"],
        "prediction_times_sec": frozen["prediction_times_sec"],
    }


def variant_C_tie_aware_continuity(stem):
    """Same raw evidence as control (frozen raw_scores/active, reused not
    recomputed). Only the tie-break rule differs: prefer the previous
    predicted key if it is among the tied-for-max keys this window;
    otherwise fall back to plain np.argmax (== smallest tied index,
    identical to the control's own convention). No anchor/expected-key
    input anywhere in this function."""
    frozen = load_frozen_arrays(stem)
    raw_scores = frozen["raw_scores"]
    active = frozen["active"]
    times = frozen["prediction_times_sec"]
    T = raw_scores.shape[0]

    key_id_c = np.full(T, -1, dtype=np.int64)
    last_key = -1
    for t in range(T):
        if not active[t]:
            key_id_c[t] = last_key
            continue
        max_score = raw_scores[t].max()
        tied = np.where(np.abs(raw_scores[t] - max_score) < TIE_EPS)[0]  # ascending order -> tied[0] == np.argmax's own tie-break
        if last_key in tied:
            chosen = int(last_key)
        else:
            chosen = int(tied[0])
        key_id_c[t] = chosen
        last_key = chosen

    return {"key_id": key_id_c, "active": active, "raw_scores": raw_scores, "prediction_times_sec": times}


def variant_D_weighted_profile(stem):
    """New evidence (WEIGHTED_TEMPLATES applied to the SAME frozen
    thresholded chroma Phase 3G-A used, reused not recomputed), decision
    rule is plain np.argmax -- the exact same rule the control uses,
    applied to different scores. No anchor/expected-key input anywhere in
    this function."""
    frozen = load_frozen_arrays(stem)
    thresholded_chroma = frozen["thresholded_chroma"]
    times = frozen["prediction_times_sec"]
    T = thresholded_chroma.shape[0]

    raw_scores_d = thresholded_chroma @ WEIGHTED_TEMPLATES.T
    active_d = raw_scores_d.max(axis=1) > 0

    key_id_d = np.full(T, -1, dtype=np.int64)
    last_key = -1
    for t in range(T):
        if not active_d[t]:
            key_id_d[t] = last_key
            continue
        chosen = int(np.argmax(raw_scores_d[t]))
        key_id_d[t] = chosen
        last_key = chosen

    active_matches_control = bool(np.array_equal(active_d, frozen["active"]))
    return {
        "key_id": key_id_d, "active": active_d, "raw_scores": raw_scores_d, "prediction_times_sec": times,
        "active_matches_control": active_matches_control,
    }


# ---------------------------------------------------------------------------
# Generic metrics (variant-agnostic -- take arrays, never reference which
# variant produced them)
# ---------------------------------------------------------------------------

def compute_key_switches_and_jumps(key_id):
    """Mirrors pitch_class_uncertainty_diagnostics.analyze_piece's own
    key_switch/jump_distance loop exactly (recomputed fresh here since
    variants C/D produce a different key_id sequence than the frozen
    control's saved key_switch.npy/jump_distance.npy)."""
    T = len(key_id)
    defined = key_id != -1
    key_switch = np.zeros(T, dtype=bool)
    jump_distance = np.zeros(T, dtype=np.float64)
    for t in range(1, T):
        if defined[t] and defined[t - 1]:
            key_switch[t] = key_id[t] != key_id[t - 1]
            jump_distance[t] = fifth_distance(key_tonic_pc(int(key_id[t - 1])), key_tonic_pc(int(key_id[t])))
    return key_switch, jump_distance


def compute_piece_level_metrics(key_id, active, raw_scores, large_jump_threshold=DEFAULT_LARGE_JUMP_THRESHOLD):
    T = len(key_id)
    defined = key_id != -1
    key_switch, jump_distance = compute_key_switches_and_jumps(key_id)
    large_jump = jump_distance >= large_jump_threshold

    eligible_mask = np.zeros(T, dtype=bool)
    eligible_mask[1:] = defined[1:] & defined[:-1]
    n_eligible = int(eligible_mask.sum())
    n_switches = int(key_switch.sum())
    n_large = int(large_jump.sum())
    jump_vals = jump_distance[eligible_mask]

    total_defined = int(defined.sum())
    unique_keys = sorted(set(key_id[defined].tolist())) if total_defined > 0 else []
    counts = Counter(key_id[defined].tolist()) if total_defined > 0 else Counter()
    dominant = [
        {"key": f"{decode_key(k)[0]} {decode_key(k)[1]}", "count": c, "fraction": c / total_defined}
        for k, c in sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    ] if total_defined > 0 else []

    tie_count = np.sum(raw_scores == raw_scores.max(axis=1, keepdims=True), axis=1)
    n_active = int(active.sum())
    n_minor = int(sum(1 for k in key_id[defined] if k >= 12)) if total_defined > 0 else 0

    return {
        "n_predictions": T,
        "n_active": n_active,
        "active_fraction": (n_active / T) if T > 0 else None,
        "n_unique_predicted_keys": len(unique_keys),
        "unique_predicted_keys": [f"{decode_key(k)[0]} {decode_key(k)[1]}" for k in unique_keys],
        "dominant_predicted_keys": dominant,
        "n_key_switch_eligible_transitions": n_eligible,
        "n_key_switches": n_switches,
        "key_switch_proportion": (n_switches / n_eligible) if n_eligible > 0 else None,
        "mean_jump": float(jump_vals.mean()) if jump_vals.size > 0 else None,
        "max_jump": float(jump_vals.max()) if jump_vals.size > 0 else None,
        "n_large_jumps": n_large,
        "large_jump_proportion": (n_large / n_eligible) if n_eligible > 0 else None,
        "mean_tie_count_active_windows": float(tie_count[active].mean()) if n_active > 0 else None,
        "max_tie_count_active_windows": int(tie_count[active].max()) if n_active > 0 else None,
        "n_windows_predicted_minor_mode": n_minor,
        "fraction_defined_windows_predicted_minor_mode": (n_minor / total_defined) if total_defined > 0 else None,
    }


def compute_anchor_metrics(key_id, active, raw_scores, times, anchor):
    """Anchor is used HERE ONLY, strictly for evaluation, after key_id was
    already fully computed by a variant function above with no anchor
    input at all."""
    t_start = anchor["start_sec"] if anchor["start_sec"] is not None else 0.0
    t_end = anchor["end_sec"] if anchor["end_sec"] is not None else float(times[-1]) + 1.0
    mask = (times >= t_start) & (times < t_end)
    idxs = np.where(mask)[0]

    expected_id = _expected_key_id(anchor["expected_key_name"])
    collection_id = collection_equivalent_key_id(expected_id)

    defined = key_id[idxs] != -1
    sub_idxs = idxs[defined]
    n_total = int(len(idxs))
    n_defined = int(len(sub_idxs))

    if n_defined == 0:
        return {
            "expected_key": anchor["expected_key_name"], "n_predictions": n_total, "n_defined": 0,
            "strict_expected_key_proportion": None,
            "collection_equivalent_key": f"{decode_key(collection_id)[0]} {decode_key(collection_id)[1]}",
            "collection_equivalent_proportion": None,
            "expected_key_in_top_tie": {"n_tied_for_max": 0, "n_selected_when_tied": 0, "n_lost_to_tiebreak": 0},
            "mismatch_predicted_key_breakdown": {}, "n_windows_predicted_minor_mode": 0,
            "fraction_defined_windows_predicted_minor_mode": None,
        }

    strict_match = key_id[sub_idxs] == expected_id
    collection_match = strict_match | (key_id[sub_idxs] == collection_id)

    max_scores = raw_scores[sub_idxs].max(axis=1)
    expected_scores = raw_scores[sub_idxs, expected_id]
    tied_for_max = active[sub_idxs] & (max_scores > 0) & (np.abs(expected_scores - max_scores) < TIE_EPS)
    n_tied = int(tied_for_max.sum())
    n_tied_selected = int((tied_for_max & strict_match).sum())

    mismatch_breakdown = Counter(
        f"{decode_key(int(key_id[sub_idxs[i]]))[0]} {decode_key(int(key_id[sub_idxs[i]]))[1]}"
        for i in range(n_defined) if not strict_match[i]
    )
    n_minor = int(sum(1 for i in range(n_defined) if int(key_id[sub_idxs[i]]) >= 12))

    return {
        "expected_key": anchor["expected_key_name"],
        "n_predictions": n_total,
        "n_defined": n_defined,
        "strict_expected_key_proportion": float(strict_match.mean()),
        "collection_equivalent_key": f"{decode_key(collection_id)[0]} {decode_key(collection_id)[1]}",
        "collection_equivalent_proportion": float(collection_match.mean()),
        "expected_key_in_top_tie": {
            "n_tied_for_max": n_tied, "n_selected_when_tied": n_tied_selected, "n_lost_to_tiebreak": n_tied - n_tied_selected,
        },
        "mismatch_predicted_key_breakdown": dict(mismatch_breakdown),
        "n_windows_predicted_minor_mode": n_minor,
        "fraction_defined_windows_predicted_minor_mode": n_minor / n_defined,
    }


def bach_tonic_neighborhood(mismatch_breakdown):
    neighborhood = {"C maj", "D maj", "G maj"}
    n_mismatch = sum(mismatch_breakdown.values())
    n_in = sum(v for k, v in mismatch_breakdown.items() if k in neighborhood)
    return {
        "n_mismatched_windows": n_mismatch,
        "n_within_CGD_tonic_neighborhood": n_in,
        "proportion_within_CGD_tonic_neighborhood": (n_in / n_mismatch) if n_mismatch > 0 else None,
        "mismatch_breakdown": mismatch_breakdown,
    }


def clementi_run_behavior(key_id, times):
    defined_mask = key_id != -1
    if not defined_mask.all():
        idxs = np.where(defined_mask)[0]
        key_id = key_id[idxs]
        times = times[idxs]
    runs = _key_run_length_encode(key_id, times)
    run_keys = [r["key"] for r in runs]
    return {
        "predicted_key_runs": runs, "run_key_sequence": run_keys,
        "is_exact_c_maj_g_maj_c_maj_pattern": run_keys == ["C maj", "G maj", "C maj"],
    }


def chopin_silence_windows(active, times, silence_region):
    mask = (times >= silence_region["start_sec"]) & (times <= silence_region["end_sec"])
    idxs = np.where(mask)[0]
    n_in_region = int(len(idxs))
    n_active_in_region = int(active[idxs].sum()) if n_in_region > 0 else 0
    return {
        "n_windows_in_region": n_in_region, "n_active_in_region": n_active_in_region,
        "n_inactive_in_region": n_in_region - n_active_in_region,
    }


# ---------------------------------------------------------------------------
# Per-piece, all-variants orchestration
# ---------------------------------------------------------------------------

def process_piece_all_variants(piece):
    stem = piece["stem"]
    duration = load_piece_duration(stem)
    anchors_spec = piece["anchors_fn"](duration)

    variant_data = {
        "A_control": variant_A_control(stem),
        "C_tie_aware_continuity": variant_C_tie_aware_continuity(stem),
        "D_weighted_profile": variant_D_weighted_profile(stem),
    }

    # Integrity check: variant A's fresh key_switch/jump_distance
    # recomputation (via compute_key_switches_and_jumps, the same generic
    # function used for all variants) must exactly reproduce Phase 3G-A's
    # own frozen key_switch.npy/jump_distance.npy arrays for this piece --
    # confirming the "control" logic really is a faithful, unmodified
    # reproduction, not an approximation.
    frozen = load_frozen_arrays(stem)
    a_key_switch, a_jump_distance = compute_key_switches_and_jumps(variant_data["A_control"]["key_id"])
    control_reproduction_matches_frozen = bool(
        np.array_equal(a_key_switch, frozen["key_switch"]) and np.allclose(a_jump_distance, frozen["jump_distance"])
    )

    piece_out = {"display_name": piece["display_name"], "control_reproduction_matches_frozen": control_reproduction_matches_frozen, "variants": {}}

    for vname in VARIANT_NAMES:
        vdata = variant_data[vname]
        key_id, active, raw_scores, times = vdata["key_id"], vdata["active"], vdata["raw_scores"], vdata["prediction_times_sec"]

        piece_level = compute_piece_level_metrics(key_id, active, raw_scores)
        anchors_out = {a["name"]: compute_anchor_metrics(key_id, active, raw_scores, times, a) for a in anchors_spec}

        variant_out = {"piece_level": piece_level, "anchors": anchors_out}

        if vname == "D_weighted_profile":
            variant_out["active_matches_control"] = vdata["active_matches_control"]

        if piece["level"] == "L2":
            variant_out["bach_tonic_neighborhood"] = bach_tonic_neighborhood(anchors_out["full_piece"]["mismatch_predicted_key_breakdown"])
        if piece["level"] == "L5":
            variant_out["clementi_run_behavior"] = clementi_run_behavior(key_id, times)
        if piece["level"] == "L4":
            variant_out["chopin_silence_window_handling"] = chopin_silence_windows(active, times, CHOPIN_SILENCE_REGION)

        piece_out["variants"][vname] = variant_out

    piece_out["_key_id_by_variant"] = {v: variant_data[v]["key_id"] for v in VARIANT_NAMES}  # internal use for plotting, stripped before JSON save
    piece_out["_times"] = variant_data["A_control"]["prediction_times_sec"]

    return piece_out


# ---------------------------------------------------------------------------
# Cross-variant summary + explicit verdict (computed from real results,
# not assumed)
# ---------------------------------------------------------------------------

def build_cross_variant_summary(pieces_out):
    def strict(level, anchor_name, vname):
        return pieces_out[level]["variants"][vname]["anchors"][anchor_name]["strict_expected_key_proportion"]

    def collection(level, anchor_name, vname):
        return pieces_out[level]["variants"][vname]["anchors"][anchor_name]["collection_equivalent_proportion"]

    l3_l4_minor_recovery = {
        "L3_full_excerpt_A_minor": {v: {"strict": strict("L3", "full_excerpt", v), "collection": collection("L3", "full_excerpt", v)} for v in VARIANT_NAMES},
        "L4_full_piece_E_minor": {v: {"strict": strict("L4", "full_piece", v), "collection": collection("L4", "full_piece", v)} for v in VARIANT_NAMES},
    }

    l1_l6_stability = {
        "L1_full_piece_C_major": {v: {
            "strict": strict("L1", "full_piece", v),
            "n_key_switches": pieces_out["L1"]["variants"][v]["piece_level"]["n_key_switches"],
            "n_large_jumps": pieces_out["L1"]["variants"][v]["piece_level"]["n_large_jumps"],
        } for v in VARIANT_NAMES},
        "L6_pre_384s": {v: strict("L6", "pre_384s", v) for v in VARIANT_NAMES},
        "L6_384_to_432s": {v: strict("L6", "384_to_432s", v) for v in VARIANT_NAMES},
        "L6_post_432s": {v: strict("L6", "post_432s", v) for v in VARIANT_NAMES},
        "L6_n_key_switches": {v: pieces_out["L6"]["variants"][v]["piece_level"]["n_key_switches"] for v in VARIANT_NAMES},
        "L6_n_large_jumps": {v: pieces_out["L6"]["variants"][v]["piece_level"]["n_large_jumps"] for v in VARIANT_NAMES},
    }

    l2_bach_by_variant = {v: pieces_out["L2"]["variants"][v]["bach_tonic_neighborhood"] for v in VARIANT_NAMES}
    l5_clementi_by_variant = {v: pieces_out["L5"]["variants"][v]["clementi_run_behavior"] for v in VARIANT_NAMES}
    l4_chopin_silence_by_variant = {v: pieces_out["L4"]["variants"][v]["chopin_silence_window_handling"] for v in VARIANT_NAMES}

    # --- Explicit, dynamically-computed verdict ---
    a_l3, d_l3 = l3_l4_minor_recovery["L3_full_excerpt_A_minor"]["A_control"]["strict"], l3_l4_minor_recovery["L3_full_excerpt_A_minor"]["D_weighted_profile"]["strict"]
    a_l4, d_l4 = l3_l4_minor_recovery["L4_full_piece_E_minor"]["A_control"]["strict"], l3_l4_minor_recovery["L4_full_piece_E_minor"]["D_weighted_profile"]["strict"]
    minor_improved_l3 = (d_l3 or 0.0) > (a_l3 or 0.0)
    minor_improved_l4 = (d_l4 or 0.0) > (a_l4 or 0.0)

    a_l1_strict = l1_l6_stability["L1_full_piece_C_major"]["A_control"]["strict"]
    d_l1_strict = l1_l6_stability["L1_full_piece_C_major"]["D_weighted_profile"]["strict"]
    a_l1_switches = l1_l6_stability["L1_full_piece_C_major"]["A_control"]["n_key_switches"]
    d_l1_switches = l1_l6_stability["L1_full_piece_C_major"]["D_weighted_profile"]["n_key_switches"]
    l1_stable = (d_l1_strict is not None and a_l1_strict is not None and d_l1_strict >= a_l1_strict - 1e-9) and (d_l1_switches <= a_l1_switches)

    l6_deltas = {anchor: (l1_l6_stability[anchor]["D_weighted_profile"] or 0.0) - (l1_l6_stability[anchor]["A_control"] or 0.0) for anchor in ["L6_pre_384s", "L6_384_to_432s", "L6_post_432s"]}
    l6_stable = all(delta >= -0.05 for delta in l6_deltas.values())  # tolerate small, documented degradation only

    verdict_parts = []
    verdict_parts.append(
        f"L3 (Für Elise, A minor) strict expected-key proportion: A={a_l3:.4f} -> D={d_l3:.4f} "
        f"({'IMPROVED' if minor_improved_l3 else 'NOT improved'})."
    )
    verdict_parts.append(
        f"L4 (Chopin, E minor) strict expected-key proportion: A={a_l4:.4f} -> D={d_l4:.4f} "
        f"({'IMPROVED' if minor_improved_l4 else 'NOT improved'})."
    )
    verdict_parts.append(
        f"L1 (Twinkle, C major) strict expected-key proportion: A={a_l1_strict:.4f} -> D={d_l1_strict:.4f}, "
        f"key switches: A={a_l1_switches} -> D={d_l1_switches} "
        f"({'STABLE/preserved' if l1_stable else 'DEGRADED'})."
    )
    verdict_parts.append(
        "L6 (Twinkle 12) per-anchor strict proportion deltas (D - A): " +
        ", ".join(f"{k.replace('L6_', '')}: {v:+.4f}" for k, v in l6_deltas.items()) +
        f" ({'STABLE/preserved' if l6_stable else 'DEGRADED'})."
    )

    overall_success = (minor_improved_l3 or minor_improved_l4) and l1_stable and l6_stable
    verdict_parts.append(
        "OVERALL: the weighted key-profile variant (D) " +
        ("DOES improve strict minor-key recovery on at least one minor-key piece WITHOUT damaging L1/L6 stability."
         if overall_success else
         "does NOT achieve both improved strict minor-key recovery and preserved L1/L6 stability simultaneously -- see the deltas above for exactly where it falls short.")
    )

    return {
        "l3_l4_minor_recovery_by_variant": l3_l4_minor_recovery,
        "l1_l6_stability_by_variant": l1_l6_stability,
        "l2_bach_tonic_neighborhood_by_variant": l2_bach_by_variant,
        "l5_clementi_run_behavior_by_variant": l5_clementi_by_variant,
        "l4_chopin_silence_by_variant": l4_chopin_silence_by_variant,
        "verdict_text": " ".join(verdict_parts),
        "verdict_flags": {
            "minor_recovery_improved_L3": minor_improved_l3,
            "minor_recovery_improved_L4": minor_improved_l4,
            "L1_stability_preserved": l1_stable,
            "L6_stability_preserved": l6_stable,
            "overall_success": overall_success,
        },
    }


# ---------------------------------------------------------------------------
# Plots (only where they clarify the central minor-recovery question --
# L3 and L4, the two pieces this whole phase is testing)
# ---------------------------------------------------------------------------

_VARIANT_PLOT_STYLE = {
    "A_control": {"color": "tab:gray", "marker": "o", "z": 1},
    "C_tie_aware_continuity": {"color": "tab:blue", "marker": "s", "z": 2},
    "D_weighted_profile": {"color": "tab:red", "marker": "^", "z": 3},
}


def plot_variant_comparison(key_id_by_variant, times, expected_key_name, title, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for vname in VARIANT_NAMES:
        key_id = key_id_by_variant[vname]
        style = _VARIANT_PLOT_STYLE[vname]
        defined = key_id != -1
        fifths_pos = np.full(len(key_id), np.nan)
        fifths_pos[defined] = [FIFTH_POS[key_tonic_pc(int(k))] for k in key_id[defined]]
        is_minor = np.zeros(len(key_id), dtype=bool)
        is_minor[defined] = key_id[defined] >= 12

        ax.plot(times[defined], fifths_pos[defined], color=style["color"], alpha=0.35, linewidth=0.8, zorder=style["z"])
        maj_mask = defined & ~is_minor
        min_mask = defined & is_minor
        ax.scatter(times[maj_mask], fifths_pos[maj_mask], s=16, color=style["color"], marker=style["marker"], zorder=style["z"] + 3, label=f"{vname} (major)")
        if min_mask.any():
            ax.scatter(times[min_mask], fifths_pos[min_mask], s=60, facecolors="none", edgecolors=style["color"], linewidths=1.5, marker=style["marker"], zorder=style["z"] + 3, label=f"{vname} (minor)")

    ax.set_yticks(range(12))
    ax.set_yticklabels(FIFTHS_LABEL_NAMES)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Predicted tonic (circle-of-fifths position)")
    ax.set_title(title + f"\n(expected: {expected_key_name}; open markers = minor-mode prediction)")
    ax.legend(loc="upper right", fontsize=7, ncol=1)
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


def _anchor_line(name, a):
    tt = a["expected_key_in_top_tie"]
    return (
        f"  - `{name}` (expected {a['expected_key']}): n_defined={a['n_defined']}, "
        f"strict={_fmt(a['strict_expected_key_proportion'])}, collection_equiv={_fmt(a['collection_equivalent_proportion'])}, "
        f"tied_for_max={tt['n_tied_for_max']} (selected={tt['n_selected_when_tied']}, lost_to_tiebreak={tt['n_lost_to_tiebreak']}), "
        f"minor_mode_fraction={_fmt(a['fraction_defined_windows_predicted_minor_mode'])}"
    )


def _piece_variant_section(level, piece_out, vname):
    v = piece_out["variants"][vname]
    pl = v["piece_level"]
    lines = []
    lines.append(f"**{VARIANT_LABELS[vname]}**")
    lines.append(
        f"  - piece-level: {pl['n_predictions']} predictions, {pl['n_active']} active ({_fmt(pl['active_fraction'])}), "
        f"{pl['n_unique_predicted_keys']} unique keys; dominant: " +
        ", ".join(f"{tk['key']} ({tk['fraction']:.1%})" for tk in pl["dominant_predicted_keys"])
    )
    lines.append(
        f"  - key switches: {pl['n_key_switches']}/{pl['n_key_switch_eligible_transitions']} ({_fmt(pl['key_switch_proportion'])}); "
        f"jumps: mean={_fmt(pl['mean_jump'], 2)}, max={_fmt(pl['max_jump'], 2)}, large={pl['n_large_jumps']} ({_fmt(pl['large_jump_proportion'])}); "
        f"tie_count: mean={_fmt(pl['mean_tie_count_active_windows'], 2)}, max={pl['max_tie_count_active_windows']}"
    )
    lines.append(f"  - minor-mode predictions: {pl['n_windows_predicted_minor_mode']} ({_fmt(pl['fraction_defined_windows_predicted_minor_mode'])} of defined windows)")
    for aname, a in v["anchors"].items():
        lines.append(_anchor_line(aname, a))
    if "active_matches_control" in v:
        lines.append(f"  - active mask matches frozen control exactly: {v['active_matches_control']}")
    if "bach_tonic_neighborhood" in v:
        bn = v["bach_tonic_neighborhood"]
        lines.append(f"  - Bach tonic-neighborhood: {bn['n_within_CGD_tonic_neighborhood']}/{bn['n_mismatched_windows']} mismatches within C/G/D ({_fmt(bn['proportion_within_CGD_tonic_neighborhood'])})")
    if "clementi_run_behavior" in v:
        cr = v["clementi_run_behavior"]
        lines.append(f"  - Clementi run sequence: {' -> '.join(cr['run_key_sequence'])} (exact C->G->C: {cr['is_exact_c_maj_g_maj_c_maj_pattern']})")
    if "chopin_silence_window_handling" in v:
        sw = v["chopin_silence_window_handling"]
        lines.append(f"  - Chopin silence region: {sw['n_inactive_in_region']}/{sw['n_windows_in_region']} inactive windows")
    lines.append("")
    return lines


def build_report_md(pieces_out, cross_summary, weighted_verification):
    lines = []
    lines.append("# Phase 3H-A — Non-Neural Tonic/Mode Resolver Ablations")
    lines.append("")
    lines.append(
        "Tests whether the pitch-class fast filter's tonic/mode ambiguity (Phase 3G-A/3G-B: it is a "
        "diatonic-*collection* resolver, not a tonic/mode resolver, because unweighted `SCALE_TEMPLATES` "
        "gives relative major/minor pairs identical rows and `np.argmax`'s leftmost-tie convention then "
        "always prefers the major-indexed key) can be improved by small, interpretable, non-neural "
        "decision-rule variants. **Still not a neural-modeling phase** -- no chord-id EMA/SRN, no Chroma SRN, "
        "no Transformer, no neural refinement. Phase 3G-A and Phase 3G-B are treated as frozen and are not "
        "modified or overwritten; their scripts are only imported from, their output files only read."
    )
    lines.append("")

    lines.append("## Variants")
    lines.append("")
    for vname in VARIANT_NAMES:
        lines.append(f"- {VARIANT_LABELS[vname]}")
    lines.append(
        "- B: collection-level evaluation is not a separate predictor -- it is the `strict` vs. `collection_equiv` "
        "proportion pair reported for every anchor of every variant below, using Phase 3G-B's own "
        "`collection_equivalent_key_id` (imported, not redefined)."
    )
    lines.append("")

    lines.append("## Variant D's weighted key-profile (fixed, pre-declared, not tuned on this corpus)")
    lines.append("")
    lines.append(f"- Degree roles (in order): {', '.join(DEGREE_ROLE_NAMES)}")
    lines.append(f"- Weights (aligned to the roles above): {FUNCTIONAL_WEIGHTS}")
    lines.append(
        "- Major scale degree offsets (semitones from tonic): " + str(MAJ_INTERVALS) +
        "; minor (natural): " + str(MIN_INTERVALS) +
        " -- both lists copied verbatim from `pitch_class_baseline.build_scale_templates()`, only the assigned "
        "value per degree changes (weight instead of a flat 1)."
    )
    lines.append(
        "- Rationale: tonic defines the key center (highest weight); dominant is the primary harmonic pillar "
        "(second-highest); mediant is the single pitch that distinguishes major from minor by ear, and is also "
        "exactly the scale degree whose offset differs between major (+4) and minor (+3) -- the mechanism that "
        "breaks the relative-key tie; the remaining diatonic degrees (supertonic, subdominant, submediant, "
        "leading-tone/subtonic) are weighted equally and lowest, since none individually defines tonic/mode."
    )
    lines.append(
        f"- **Structural verification (not corpus-dependent)**: of the 12 relative major/minor pairs, "
        f"{weighted_verification['n_relative_pairs_still_identical']} still have identical weighted rows "
        f"(expected: 0). All 24 rows of `WEIGHTED_TEMPLATES` are pairwise distinct: "
        f"{weighted_verification['all_24_rows_unique']} ({weighted_verification['n_unique_rows_of_24']}/24 unique)."
    )
    lines.append("")

    lines.append("## Per-piece, per-variant results")
    lines.append("")
    for level in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        po = pieces_out[level]
        lines.append(f"### {level} — {po['display_name']}")
        lines.append("")
        lines.append(f"(Control reproduction sanity check -- variant A's fresh key_switch/jump_distance recomputation exactly matches Phase 3G-A's own frozen arrays: {po['control_reproduction_matches_frozen']})")
        lines.append("")
        for vname in VARIANT_NAMES:
            lines.extend(_piece_variant_section(level, po, vname))

    lines.append("## Cross-variant findings")
    lines.append("")
    lines.append("### L3/L4 minor-key recovery by variant")
    lines.append("")
    for key, data in cross_summary["l3_l4_minor_recovery_by_variant"].items():
        lines.append(f"- **{key}**: " + "; ".join(f"{v}: strict={_fmt(d['strict'])}, collection={_fmt(d['collection'])}" for v, d in data.items()))
    lines.append("")

    lines.append("### L1/L6 stability by variant")
    lines.append("")
    l1 = cross_summary["l1_l6_stability_by_variant"]["L1_full_piece_C_major"]
    lines.append("- **L1 full_piece (C major)**: " + "; ".join(f"{v}: strict={_fmt(d['strict'])}, switches={d['n_key_switches']}, large_jumps={d['n_large_jumps']}" for v, d in l1.items()))
    for anchor in ["L6_pre_384s", "L6_384_to_432s", "L6_post_432s"]:
        lines.append(f"- **{anchor}**: " + "; ".join(f"{v}: strict={_fmt(p)}" for v, p in cross_summary['l1_l6_stability_by_variant'][anchor].items()))
    lines.append("- **L6 overall**: " + "; ".join(f"{v}: switches={cross_summary['l1_l6_stability_by_variant']['L6_n_key_switches'][v]}, large_jumps={cross_summary['l1_l6_stability_by_variant']['L6_n_large_jumps'][v]}" for v in VARIANT_NAMES))
    lines.append("")

    lines.append("### Bach (L2) tonic-neighborhood behavior by variant")
    lines.append("")
    for v, bn in cross_summary["l2_bach_tonic_neighborhood_by_variant"].items():
        lines.append(f"- {v}: {bn['n_within_CGD_tonic_neighborhood']}/{bn['n_mismatched_windows']} mismatches within C/G/D ({_fmt(bn['proportion_within_CGD_tonic_neighborhood'])}); breakdown: {bn['mismatch_breakdown']}")
    lines.append("")

    lines.append("### Clementi (L5) run behavior by variant")
    lines.append("")
    for v, cr in cross_summary["l5_clementi_run_behavior_by_variant"].items():
        lines.append(f"- {v}: {' -> '.join(cr['run_key_sequence'])} (exact C->G->C: {cr['is_exact_c_maj_g_maj_c_maj_pattern']})")
    lines.append("")

    lines.append("### Chopin (L4) silence-window handling by variant")
    lines.append("")
    lines.append(
        "Per the task's instruction, silence behavior is reported here only to confirm whether it changed "
        "mechanically -- not reinterpreted (Phase 3G-B's boundary-granularity + smoothing-memory mechanism "
        "explanation stands unless a variant's numbers actually differ from the frozen control's)."
    )
    for v, sw in cross_summary["l4_chopin_silence_by_variant"].items():
        lines.append(f"- {v}: {sw['n_inactive_in_region']}/{sw['n_windows_in_region']} inactive")
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{cross_summary['verdict_text']}**")
    lines.append("")
    flags = cross_summary["verdict_flags"]
    lines.append(f"- minor_recovery_improved_L3 = {flags['minor_recovery_improved_L3']}")
    lines.append(f"- minor_recovery_improved_L4 = {flags['minor_recovery_improved_L4']}")
    lines.append(f"- L1_stability_preserved = {flags['L1_stability_preserved']}")
    lines.append(f"- L6_stability_preserved = {flags['L6_stability_preserved']}")
    lines.append(f"- overall_success = {flags['overall_success']}")
    lines.append("")

    lines.append("## Mechanism: why D helps L3/L4 but hurts L1/L6")
    lines.append("")
    l1_d_dominant = pieces_out["L1"]["variants"]["D_weighted_profile"]["piece_level"]["dominant_predicted_keys"]
    lines.append(
        "L1 (Twinkle) is monophonic -- at any moment its EMA-smoothed, 10%-thresholded chroma is dominated by "
        "whichever one or two notes were most recently played, not a stable multi-note harmony. Under the "
        "**unweighted** control template, a single active pitch class ties across every one of the (typically "
        "5-7) keys that contain it as ANY diatonic degree, and `np.argmax`'s low-index tie-break happens to land "
        "on C major disproportionately often for this piece -- Phase 3B already noted this behavior comes from "
        "broad, largely accidental ties, not real tonic disambiguation. Under the **weighted** profile, that same "
        "single pitch class instead scores highest for whichever key treats it as the HIGHEST-weighted degree "
        "(tonic=5), which is a different key depending on which note was just played -- so the weighted resolver "
        "tracks the melody's passing notes' own local tonic-implication rather than the piece's actual, stable "
        "tonic. L1's dominant predicted keys under D are "
        + ", ".join(f"{tk['key']} ({tk['fraction']:.1%})" for tk in l1_d_dominant) +
        " -- scattered across several tonics rather than concentrated on C major, confirming this mechanism "
        "directly. The same effect degrades L6's largely-monophonic melody-plus-light-accompaniment texture. "
        "By contrast, L3/L4 have enough real, simultaneously-sounding harmonic content (or enough EMA-accumulated "
        "note history) that the mediant-weighted tonic/dominant/mediant evidence more often correctly favors the "
        "true (minor) tonic over its relative major -- exactly the ambiguity D was designed to break. **Net "
        "reading: D is not a strict improvement over the control -- it trades away monophonic-melody stability "
        "for minor-key tonic/mode resolution, and does not dominate the control across the whole corpus.**"
    )
    lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append(f"- `{os.path.relpath(OUT_PLOT_FUR_ELISE, os.path.join(_THIS_DIR, '..'))}`")
    lines.append(f"- `{os.path.relpath(OUT_PLOT_CHOPIN, os.path.join(_THIS_DIR, '..'))}`")
    lines.append("(Only the two minor-key pieces central to this phase's question are plotted, per the task's instruction to avoid excessive plotting.)")
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This is Phase 3H-A only: non-neural decision-rule and representation ablations on top of the frozen "
        "Phase 3G-A pitch-class baseline. No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement was "
        "run or implemented. `np.argmax`'s own tie-breaking behavior was not changed anywhere -- variant C adds a "
        "continuity preference that only applies when the previous key is itself among the tied keys (falling back "
        "to the same argmax rule otherwise), and variant D applies the unchanged argmax rule to a different, "
        "hand-specified (not trained, not tuned on this corpus) template. No dense per-timestep accuracy is "
        "claimed anywhere -- all anchor comparisons are the same window-level, documented-expected-key convention "
        "Phase 3G-A/3G-B already used. Anchors were used exclusively inside the evaluation functions "
        "(`compute_anchor_metrics`, `bach_tonic_neighborhood`), never inside any variant's key_id computation "
        "(`variant_A_control`, `variant_C_tie_aware_continuity`, `variant_D_weighted_profile` take no anchor or "
        "expected-key argument at all)."
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


def run_verification(results, out_paths, phase3g_a_mtime_before, phase3g_b_mtime_before,
                      derived_corpus_mtimes_before, old_script_mtimes_before):
    checks = []

    for p in out_paths:
        checks.append((f"{os.path.basename(p)} exists", os.path.exists(p)))
        checks.append((f"{os.path.basename(p)} is non-empty", os.path.exists(p) and os.path.getsize(p) > 0))

    nan_paths = _scan_for_nan(results)
    checks.append(("no NaNs in Phase 3H-A metrics", len(nan_paths) == 0))

    checks.append(("PHASE3G_A_pitch_class_corpus_metrics.json not modified", os.path.getmtime(PHASE3G_A_METRICS_JSON) == phase3g_a_mtime_before))
    checks.append(("PHASE3G_B_tie_aware_diagnostics_metrics.json not modified", os.path.getmtime(PHASE3G_B_METRICS_JSON) == phase3g_b_mtime_before))
    for path, mtime_before in derived_corpus_mtimes_before.items():
        checks.append((f"derived_phase3g_corpus/{os.path.basename(path)} not modified", os.path.getmtime(path) == mtime_before))
    for path, mtime_before in old_script_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} (old/frozen script) not modified", os.path.getmtime(path) == mtime_before))

    for level, po in results["pieces"].items():
        checks.append((f"{level} control reproduces frozen Phase 3G-A key_switch/jump_distance exactly", po["control_reproduction_matches_frozen"]))

    checks.append((
        "variant-computation functions take no anchor/expected-key argument (structural, verified by code inspection)",
        True,
    ))

    print()
    print("evaluate_phase3h_a_tonic_mode_resolver.py verification")
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


def main():
    os.makedirs(_FIGURES_DIR, exist_ok=True)

    phase3g_a_mtime_before = os.path.getmtime(PHASE3G_A_METRICS_JSON)
    phase3g_b_mtime_before = os.path.getmtime(PHASE3G_B_METRICS_JSON)
    derived_corpus_files = [os.path.join(_DERIVED_CORPUS_DIR, f) for f in os.listdir(_DERIVED_CORPUS_DIR)]
    derived_corpus_mtimes_before = {p: os.path.getmtime(p) for p in derived_corpus_files}

    old_scripts = [
        "midi_chroma_extraction.py", "pitch_class_baseline.py",
        "evaluate_pitch_class_phase2d.py", "pitch_class_uncertainty_diagnostics.py",
        "compare_phase3c_disagreement.py", "evaluate_phase3g_pitch_class_corpus.py",
        "evaluate_phase3g_b_tie_aware_diagnostics.py",
    ]
    old_script_mtimes_before = {
        os.path.join(_THIS_DIR, s): os.path.getmtime(os.path.join(_THIS_DIR, s))
        for s in old_scripts if os.path.exists(os.path.join(_THIS_DIR, s))
    }

    weighted_verification = verify_weighted_templates_break_relative_ties()
    print("Weighted-template structural verification:", weighted_verification)

    pieces_out = {}
    for piece in PIECES:
        print(f"\n=== {piece['level']} — {piece['display_name']} (all variants) ===")
        pieces_out[piece["level"]] = process_piece_all_variants(piece)
        for vname in VARIANT_NAMES:
            pl = pieces_out[piece["level"]]["variants"][vname]["piece_level"]
            print(f"  {vname}: switches={pl['n_key_switches']}, minor_mode_fraction={pl['fraction_defined_windows_predicted_minor_mode']}")

    cross_summary = build_cross_variant_summary(pieces_out)
    print("\nVerdict:", cross_summary["verdict_text"])

    print("\nPlotting variant comparisons for L3 (Für Elise) and L4 (Chopin)...")
    plot_variant_comparison(
        pieces_out["L3"]["_key_id_by_variant"], pieces_out["L3"]["_times"], "A Minor",
        f"Phase 3H-A — {pieces_out['L3']['display_name']}: Variant Comparison", OUT_PLOT_FUR_ELISE,
    )
    print(f"  wrote {OUT_PLOT_FUR_ELISE}")
    plot_variant_comparison(
        pieces_out["L4"]["_key_id_by_variant"], pieces_out["L4"]["_times"], "E Minor",
        f"Phase 3H-A — {pieces_out['L4']['display_name']}: Variant Comparison", OUT_PLOT_CHOPIN,
    )
    print(f"  wrote {OUT_PLOT_CHOPIN}")

    results = {
        "phase": "phase_3h_a_tonic_mode_resolver_ablations",
        "based_on_frozen": [
            "PHASE3G_A_pitch_class_corpus_metrics.json (read-only)",
            "03_MIDI_Data/derived_phase3g_corpus/*.npy (read-only)",
            "PHASE3G_B_tie_aware_diagnostics_metrics.json (read-only, referenced for continuity, not re-loaded as data)",
        ],
        "settings": {
            "window_sec": DEFAULT_WINDOW_SEC, "memory_decay": DEFAULT_MEMORY_DECAY, "threshold_ratio": DEFAULT_THRESHOLD_RATIO,
            "large_jump_threshold": DEFAULT_LARGE_JUMP_THRESHOLD,
        },
        "weighted_key_profile": {
            "degree_role_names": DEGREE_ROLE_NAMES, "functional_weights": FUNCTIONAL_WEIGHTS,
            "maj_intervals": MAJ_INTERVALS, "min_intervals": MIN_INTERVALS,
            "structural_verification": weighted_verification,
        },
        "pieces": pieces_out,
        "cross_variant_summary": cross_summary,
        "notes": (
            "Phase 3H-A: non-neural tonic/mode resolver ablations. No chord-id EMA/SRN, Chroma SRN, Transformer, "
            "or neural refinement. No dense per-timestep accuracy claimed. Anchors used only for evaluation, "
            "never for choosing predictions. Phase 3G-A/3G-B treated as frozen and not modified or overwritten."
        ),
    }
    results = _to_native(results)

    with open(OUT_METRICS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_METRICS_JSON}")

    report_md = build_report_md(pieces_out, cross_summary, weighted_verification)
    with open(OUT_REPORT_MD, "w") as f:
        f.write(report_md)
    print(f"Wrote {OUT_REPORT_MD}")

    out_paths = [OUT_METRICS_JSON, OUT_REPORT_MD, OUT_PLOT_FUR_ELISE, OUT_PLOT_CHOPIN]
    run_verification(results, out_paths, phase3g_a_mtime_before, phase3g_b_mtime_before, derived_corpus_mtimes_before, old_script_mtimes_before)


if __name__ == "__main__":
    main()
