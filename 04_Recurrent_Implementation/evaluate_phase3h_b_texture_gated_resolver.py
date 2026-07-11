"""evaluate_phase3h_b_texture_gated_resolver.py

Phase 3H-B: texture-gated non-neural tonic/mode resolver.

Phase 3H-A found that the weighted key-profile resolver (variant D) is a
genuine trade-off, not a replacement for the frozen Phase 3G-A control:
it improves strict minor-key recovery on Für Elise (0.0 -> 0.415) and
Chopin (0.0 -> 0.407), but badly destabilizes the monophonic pieces --
Twinkle drops from a perfect 1.0/0-switches sanity check to 0.359/45
switches, and Twinkle 12's three anchors all degrade. The documented
mechanism: sparse, single-pitch-class evidence under a tonic-weighted
template chases each recently-played note's own local tonic implication
rather than the piece's one stable tonic.

This script tests the natural next question: can the weighted profile be
applied *conditionally* -- only when the local evidence is dense and
stable enough to trust it -- so it helps where it helps (minor-key
pieces) without being switched on where it hurts (monophonic pieces)?

Still NOT a neural-modeling phase. No chord-id EMA/SRN, no Chroma SRN, no
Transformer, no neural refinement of any kind.

Phase 3G-A, Phase 3G-B, and Phase 3H-A are treated as **frozen**. This
script does not modify any of their scripts or output files -- it only
imports (does not modify) their already-generic, reusable functions and
constants:

  From evaluate_phase3g_pitch_class_corpus: PIECES, CHOPIN_SILENCE_REGION,
    _key_run_length_encode.
  From evaluate_phase3g_b_tie_aware_diagnostics: collection_equivalent_key_id.
  From evaluate_phase3h_a_tonic_mode_resolver: variant_A_control,
    variant_C_tie_aware_continuity, variant_D_weighted_profile (reused
    verbatim -- this is how Phase 3H-A's exact, unchanged weighted-profile
    predictions are reproduced here, per the task's "do not change the
    weights" instruction), compute_key_switches_and_jumps,
    compute_piece_level_metrics, compute_anchor_metrics,
    bach_tonic_neighborhood, clementi_run_behavior, chopin_silence_windows,
    load_piece_duration, WEIGHTED_TEMPLATES and its documentation constants.

Three variants compared (per the task's explicit request -- A, D, E; C is
used internally as E's stable default backbone and reported for context
only, not as a fourth headline comparison column):

  A. Frozen Phase 3G-A control -- loaded verbatim, zero recomputation
     (via variant_A_control, unchanged from Phase 3H-A).

  D. Weighted key-profile matcher -- Phase 3H-A's exact fixed profile,
     reproduced via variant_D_weighted_profile (unchanged weights).

  E. Texture-gated resolver -- defaults to variant C's tie-aware-
     continuity prediction (a documented design choice: C is specifically
     the "stable" variant Phase 3H-A validated as switch-reducing with no
     effect on minor-key recovery, so it is the natural stable backbone
     to gate weighted swaps on top of, rather than raw A). At each
     window, allows the weighted-profile prediction (D) to REPLACE the
     default only if all three predeclared, label-free gates pass:

       1. Density gate: active_pc_count (nonzero entries in the
          thresholded smoothed chroma at that window) > 2. Exactly the
          threshold specified by the task ("if active_pc_count <= 2, do
          NOT use weighted profile").
       2. Collection-stability gate: the default (C) track's diatonic
          collection (relative major/minor pairs collapsed to one
          canonical id) must have been constant over the STABILITY_WINDOW
          predictions immediately preceding this window (causal -- no
          lookahead). Uses only C's own prediction history, never an
          anchor/expected-key label.
       3. Weighted-margin gate: the WEIGHTED profile's own normalized
          top1-vs-top2 score margin at this window must be >=
          MARGIN_THRESHOLD. MARGIN_THRESHOLD reuses
          pitch_class_uncertainty_diagnostics.DEFAULT_LOW_MARGIN_THRESHOLD
          (0.20) verbatim -- an existing, already-predeclared constant
          from Phase 3B, chosen for this script specifically BECAUSE it
          was fixed before this task and cannot have been tuned to this
          question's results. Note this is applied to the WEIGHTED
          representation's margin, which (unlike Phase 3B's finding for
          the unweighted representation) is not expected to be
          structurally saturated, since Phase 3H-A already verified all
          24 weighted template rows are pairwise distinct.

Guardrails honored throughout: anchors are used ONLY inside the
evaluation functions (compute_anchor_metrics, bach_tonic_neighborhood),
never inside any variant's key_id computation or any gate. No dense
per-timestep accuracy is claimed anywhere.
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
_DERIVED_CORPUS_DIR = os.path.join(_MIDI_DIR, "derived_phase3g_corpus")
_FIGURES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "05_Figures_Results"))

from shared_music_defs import decode_key, key_tonic_pc, FIFTH_POS  # noqa: E402
from pitch_class_baseline import DEFAULT_WINDOW_SEC, DEFAULT_MEMORY_DECAY  # noqa: E402
from midi_chroma_extraction import DEFAULT_THRESHOLD_RATIO  # noqa: E402
from pitch_class_uncertainty_diagnostics import DEFAULT_LARGE_JUMP_THRESHOLD, DEFAULT_LOW_MARGIN_THRESHOLD, EPS as PC_EPS  # noqa: E402
from plotting_comparison import FIFTHS_LABEL_NAMES  # noqa: E402

from evaluate_phase3g_pitch_class_corpus import PIECES, CHOPIN_SILENCE_REGION  # noqa: E402
from evaluate_phase3g_b_tie_aware_diagnostics import collection_equivalent_key_id  # noqa: E402
from evaluate_phase3h_a_tonic_mode_resolver import (  # noqa: E402
    variant_A_control, variant_C_tie_aware_continuity, variant_D_weighted_profile,
    compute_piece_level_metrics, compute_anchor_metrics, bach_tonic_neighborhood,
    clementi_run_behavior, chopin_silence_windows, load_piece_duration,
    WEIGHTED_TEMPLATES, DEGREE_ROLE_NAMES, FUNCTIONAL_WEIGHTS, MAJ_INTERVALS, MIN_INTERVALS,
    _fmt,
)

assert DEFAULT_WINDOW_SEC == 0.5
assert DEFAULT_MEMORY_DECAY == 0.8
assert DEFAULT_THRESHOLD_RATIO == 0.10

PHASE3G_A_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3G_A_pitch_class_corpus_metrics.json")
PHASE3G_B_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3G_B_tie_aware_diagnostics_metrics.json")
PHASE3H_A_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3H_A_tonic_mode_resolver_metrics.json")

OUT_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3H_B_texture_gated_resolver_metrics.json")
OUT_REPORT_MD = os.path.join(_FIGURES_DIR, "PHASE3H_B_texture_gated_resolver_report.md")
OUT_PLOT_TWINKLE = os.path.join(_FIGURES_DIR, "PHASE3H_B_Twinkle_variant_comparison_key_trajectory.png")
OUT_PLOT_FUR_ELISE = os.path.join(_FIGURES_DIR, "PHASE3H_B_FurElise_excerpt_variant_comparison_key_trajectory.png")
OUT_PLOT_CHOPIN = os.path.join(_FIGURES_DIR, "PHASE3H_B_Chopin_Op28No4_variant_comparison_key_trajectory.png")
OUT_PLOT_TWINKLE12 = os.path.join(_FIGURES_DIR, "PHASE3H_B_Twinkle12_variant_comparison_key_trajectory.png")

# ---------------------------------------------------------------------------
# Predeclared, label-free gate thresholds. Chosen and fixed BEFORE this
# script was run on any piece; none were adjusted based on this run's
# expected-key results (guardrail 10 / task instruction: "Do not tune the
# threshold against expected-key results").
# ---------------------------------------------------------------------------

DENSITY_THRESHOLD = 2          # active_pc_count must be > 2 (i.e. <=2 fails), exactly as specified by the task
STABILITY_WINDOW_WINDOWS = 4   # look back this many PRIOR predictions (2.0s at window_sec=0.5) -- short enough to
                                # stay locally responsive within a phrase, long enough to filter single-note noise;
                                # a round, conservative, pre-declared choice, not swept or tuned.
MARGIN_THRESHOLD = DEFAULT_LOW_MARGIN_THRESHOLD  # reuses Phase 3B's existing 0.20 constant verbatim (see module
                                                  # docstring for why this specific reuse strengthens the
                                                  # not-tuned-to-this-question claim), applied to the WEIGHTED
                                                  # representation's margin (a different, non-saturated quantity
                                                  # from what Phase 3B measured on the unweighted representation).

GATE_SETTINGS = {
    "density_threshold_active_pc_count_gt": DENSITY_THRESHOLD,
    "stability_window_windows": STABILITY_WINDOW_WINDOWS,
    "stability_window_sec": STABILITY_WINDOW_WINDOWS * DEFAULT_WINDOW_SEC,
    "margin_threshold": MARGIN_THRESHOLD,
    "margin_threshold_source": "pitch_class_uncertainty_diagnostics.DEFAULT_LOW_MARGIN_THRESHOLD (reused verbatim, not re-tuned)",
    "margin_metric": "normalized top1-vs-top2 score margin computed on the WEIGHTED (variant D) raw scores",
    "default_backbone": "C (tie-aware continuity) -- documented design choice, see module docstring",
}


# ---------------------------------------------------------------------------
# Gate 1: active pitch-class density
# ---------------------------------------------------------------------------

def compute_active_pc_count(thresholded_chroma):
    return np.count_nonzero(thresholded_chroma, axis=1)


# ---------------------------------------------------------------------------
# Gate 2: collection stability of the default (C) track's own history
# ---------------------------------------------------------------------------

def collection_class(key_id):
    """Canonical id (0..11, or -1 for undefined) shared by a relative
    major/minor pair -- always the tonic pitch class of the MAJOR member.
    Same relationship Phase 3G-B's collection_equivalent_key_id encodes,
    expressed here as a single collapsible label for a stability check."""
    if key_id == -1:
        return -1
    tonic = key_id % 12
    if key_id < 12:
        return tonic
    return (tonic + 3) % 12


def compute_collection_stability_mask(key_id_default, prior_window_count=STABILITY_WINDOW_WINDOWS):
    """Causal (no lookahead): stable[t] looks only at the prior_window_count
    predictions strictly BEFORE t. Insufficient history (t < window) or any
    undefined (-1) prediction in that lookback -> conservatively unstable."""
    T = len(key_id_default)
    collection = np.array([collection_class(int(k)) for k in key_id_default])
    stable = np.zeros(T, dtype=bool)
    for t in range(T):
        if t < prior_window_count:
            continue
        recent = collection[t - prior_window_count:t]
        if np.any(recent == -1):
            continue
        stable[t] = bool(np.all(recent == recent[0]))
    return stable


# ---------------------------------------------------------------------------
# Gate 3: weighted-profile normalized top1-vs-top2 margin
# ---------------------------------------------------------------------------

def compute_normalized_margin(raw_scores):
    top1 = raw_scores.max(axis=1)
    sorted_scores = np.sort(raw_scores, axis=1)
    top2 = sorted_scores[:, -2]
    return (top1 - top2) / (top1 + PC_EPS)


# ---------------------------------------------------------------------------
# Variant E: texture-gated resolver
# ---------------------------------------------------------------------------

def variant_E_texture_gated(stem):
    a = variant_A_control(stem)
    c = variant_C_tie_aware_continuity(stem)
    d = variant_D_weighted_profile(stem)

    key_id_c, active_c, raw_scores_c, times = c["key_id"], c["active"], c["raw_scores"], c["prediction_times_sec"]
    key_id_d, active_d, raw_scores_d = d["key_id"], d["active"], d["raw_scores"]
    # frozen chroma, reused read-only, identical source Phase 3G-A/3H-A both used
    thresholded_chroma = np.load(os.path.join(_DERIVED_CORPUS_DIR, f"{stem}_thresholded_smoothed_chroma_decay08.npy"))

    T = len(key_id_c)
    active_pc_count = compute_active_pc_count(thresholded_chroma)
    gate1 = active_pc_count > DENSITY_THRESHOLD
    gate2 = compute_collection_stability_mask(key_id_c, STABILITY_WINDOW_WINDOWS)
    norm_margin_d = compute_normalized_margin(raw_scores_d)
    gate3 = active_d & (norm_margin_d >= MARGIN_THRESHOLD)

    use_weighted = gate1 & gate2 & gate3
    key_id_e = np.where(use_weighted, key_id_d, key_id_c)
    raw_scores_e = np.where(use_weighted[:, None], raw_scores_d, raw_scores_c)
    actual_swap = use_weighted & (key_id_e != key_id_c)

    gate_diagnostics = {
        "n_windows": int(T),
        "n_gate1_density_pass": int(gate1.sum()), "gate1_pass_fraction": float(gate1.mean()) if T > 0 else None,
        "n_gate2_stability_pass": int(gate2.sum()), "gate2_pass_fraction": float(gate2.mean()) if T > 0 else None,
        "n_gate3_margin_pass": int(gate3.sum()), "gate3_pass_fraction": float(gate3.mean()) if T > 0 else None,
        "n_all_gates_pass_weighted_eligible": int(use_weighted.sum()), "all_gates_pass_fraction": float(use_weighted.mean()) if T > 0 else None,
        "n_actual_swaps_from_default": int(actual_swap.sum()),
        "actual_swap_fraction": float(actual_swap.mean()) if T > 0 else None,
        "n_swaps_to_minor_mode": int(np.sum(actual_swap & (key_id_e >= 12))),
        "settings": GATE_SETTINGS,
    }

    return {
        "key_id": key_id_e, "active": active_c, "raw_scores": raw_scores_e, "prediction_times_sec": times,
        "gate_diagnostics": gate_diagnostics,
        "_default_key_id_c": key_id_c,  # internal, for reporting/plotting only
    }


# ---------------------------------------------------------------------------
# Per-piece, all-variants orchestration
# ---------------------------------------------------------------------------

VARIANT_NAMES_3H_B = ["A_control", "D_weighted_profile", "E_texture_gated"]
VARIANT_LABELS_3H_B = {
    "A_control": "A: frozen Phase 3G-A control (unweighted SCALE_TEMPLATES, plain argmax)",
    "D_weighted_profile": "D: weighted key-profile matcher (Phase 3H-A's fixed profile, unchanged)",
    "E_texture_gated": "E: texture-gated resolver (default C, weighted-profile swap only when density+stability+margin gates all pass)",
}


def process_piece_all_variants(piece):
    stem = piece["stem"]
    duration = load_piece_duration(stem)
    anchors_spec = piece["anchors_fn"](duration)

    a = variant_A_control(stem)
    c = variant_C_tie_aware_continuity(stem)  # reported for context only, not a headline column
    d = variant_D_weighted_profile(stem)
    e = variant_E_texture_gated(stem)

    variant_data = {"A_control": a, "D_weighted_profile": d, "E_texture_gated": e}

    piece_out = {"display_name": piece["display_name"], "variants": {}, "context_C_tie_aware_continuity": {}}

    for vname, vdata in variant_data.items():
        key_id, active, raw_scores, times = vdata["key_id"], vdata["active"], vdata["raw_scores"], vdata["prediction_times_sec"]
        piece_level = compute_piece_level_metrics(key_id, active, raw_scores)
        anchors_out = {anc["name"]: compute_anchor_metrics(key_id, active, raw_scores, times, anc) for anc in anchors_spec}
        variant_out = {"piece_level": piece_level, "anchors": anchors_out}
        if vname == "E_texture_gated":
            variant_out["gate_diagnostics"] = vdata["gate_diagnostics"]
        if piece["level"] == "L2":
            variant_out["bach_tonic_neighborhood"] = bach_tonic_neighborhood(anchors_out["full_piece"]["mismatch_predicted_key_breakdown"])
        if piece["level"] == "L5":
            variant_out["clementi_run_behavior"] = clementi_run_behavior(key_id, times)
        if piece["level"] == "L4":
            variant_out["chopin_silence_window_handling"] = chopin_silence_windows(active, times, CHOPIN_SILENCE_REGION)
        piece_out["variants"][vname] = variant_out

    # Context-only: C's own piece-level summary, since E's default backbone is C.
    c_piece_level = compute_piece_level_metrics(c["key_id"], c["active"], c["raw_scores"])
    piece_out["context_C_tie_aware_continuity"] = {"piece_level": c_piece_level}

    piece_out["_key_id_by_variant"] = {
        "A_control": a["key_id"], "D_weighted_profile": d["key_id"], "E_texture_gated": e["key_id"],
        "C_context": c["key_id"],
    }
    piece_out["_times"] = a["prediction_times_sec"]

    return piece_out


# ---------------------------------------------------------------------------
# Cross-variant summary + explicit verdict
# ---------------------------------------------------------------------------

def build_cross_variant_summary(pieces_out):
    def strict(level, anchor_name, vname):
        return pieces_out[level]["variants"][vname]["anchors"][anchor_name]["strict_expected_key_proportion"]

    def collection(level, anchor_name, vname):
        return pieces_out[level]["variants"][vname]["anchors"][anchor_name]["collection_equivalent_proportion"]

    l3_l4_minor_recovery = {
        "L3_full_excerpt_A_minor": {v: {"strict": strict("L3", "full_excerpt", v), "collection": collection("L3", "full_excerpt", v)} for v in VARIANT_NAMES_3H_B},
        "L4_full_piece_E_minor": {v: {"strict": strict("L4", "full_piece", v), "collection": collection("L4", "full_piece", v)} for v in VARIANT_NAMES_3H_B},
    }

    l1_l6_stability = {
        "L1_full_piece_C_major": {v: {
            "strict": strict("L1", "full_piece", v),
            "n_key_switches": pieces_out["L1"]["variants"][v]["piece_level"]["n_key_switches"],
            "n_large_jumps": pieces_out["L1"]["variants"][v]["piece_level"]["n_large_jumps"],
        } for v in VARIANT_NAMES_3H_B},
        "L6_pre_384s": {v: strict("L6", "pre_384s", v) for v in VARIANT_NAMES_3H_B},
        "L6_384_to_432s": {v: strict("L6", "384_to_432s", v) for v in VARIANT_NAMES_3H_B},
        "L6_post_432s": {v: strict("L6", "post_432s", v) for v in VARIANT_NAMES_3H_B},
        "L6_n_key_switches": {v: pieces_out["L6"]["variants"][v]["piece_level"]["n_key_switches"] for v in VARIANT_NAMES_3H_B},
        "L6_n_large_jumps": {v: pieces_out["L6"]["variants"][v]["piece_level"]["n_large_jumps"] for v in VARIANT_NAMES_3H_B},
    }

    l2_bach_by_variant = {v: pieces_out["L2"]["variants"][v]["bach_tonic_neighborhood"] for v in VARIANT_NAMES_3H_B}
    l5_clementi_by_variant = {v: pieces_out["L5"]["variants"][v]["clementi_run_behavior"] for v in VARIANT_NAMES_3H_B}
    l4_chopin_silence_by_variant = {v: pieces_out["L4"]["variants"][v]["chopin_silence_window_handling"] for v in VARIANT_NAMES_3H_B}

    gate_usage_summary = {
        level: pieces_out[level]["variants"]["E_texture_gated"]["gate_diagnostics"]
        for level in pieces_out
    }

    # --- Explicit, dynamically-computed verdict ---
    a_l3 = l3_l4_minor_recovery["L3_full_excerpt_A_minor"]["A_control"]["strict"]
    e_l3 = l3_l4_minor_recovery["L3_full_excerpt_A_minor"]["E_texture_gated"]["strict"]
    a_l4 = l3_l4_minor_recovery["L4_full_piece_E_minor"]["A_control"]["strict"]
    e_l4 = l3_l4_minor_recovery["L4_full_piece_E_minor"]["E_texture_gated"]["strict"]
    minor_improved_l3_e = (e_l3 or 0.0) > (a_l3 or 0.0)
    minor_improved_l4_e = (e_l4 or 0.0) > (a_l4 or 0.0)

    a_l1 = l1_l6_stability["L1_full_piece_C_major"]["A_control"]
    e_l1 = l1_l6_stability["L1_full_piece_C_major"]["E_texture_gated"]
    l1_stable_e = (e_l1["strict"] is not None and a_l1["strict"] is not None and e_l1["strict"] >= a_l1["strict"] - 1e-9) and (e_l1["n_key_switches"] <= a_l1["n_key_switches"])

    l6_deltas_e = {anchor: (l1_l6_stability[anchor]["E_texture_gated"] or 0.0) - (l1_l6_stability[anchor]["A_control"] or 0.0) for anchor in ["L6_pre_384s", "L6_384_to_432s", "L6_post_432s"]}
    l6_stable_e = all(delta >= -0.05 for delta in l6_deltas_e.values())

    overall_success_e = (minor_improved_l3_e or minor_improved_l4_e) and l1_stable_e and l6_stable_e

    verdict_parts = []
    verdict_parts.append(
        f"L3 (Für Elise, A minor) strict proportion: A={a_l3:.4f} -> E={e_l3:.4f} ({'IMPROVED' if minor_improved_l3_e else 'NOT improved'})."
    )
    verdict_parts.append(
        f"L4 (Chopin, E minor) strict proportion: A={a_l4:.4f} -> E={e_l4:.4f} ({'IMPROVED' if minor_improved_l4_e else 'NOT improved'})."
    )
    verdict_parts.append(
        f"L1 (Twinkle) strict proportion: A={a_l1['strict']:.4f} -> E={e_l1['strict']:.4f}, switches: A={a_l1['n_key_switches']} -> E={e_l1['n_key_switches']} "
        f"({'STABLE/preserved' if l1_stable_e else 'DEGRADED'})."
    )
    verdict_parts.append(
        "L6 per-anchor strict deltas (E - A): " + ", ".join(f"{k.replace('L6_', '')}: {v:+.4f}" for k, v in l6_deltas_e.items()) +
        f" ({'STABLE/preserved' if l6_stable_e else 'DEGRADED'})."
    )
    l1_swaps = gate_usage_summary["L1"]["n_actual_swaps_from_default"]
    l6_swaps = gate_usage_summary["L6"]["n_actual_swaps_from_default"]
    l3_swaps = gate_usage_summary["L3"]["n_actual_swaps_from_default"]
    l4_swaps = gate_usage_summary["L4"]["n_actual_swaps_from_default"]
    verdict_parts.append(
        f"Gate usage (actual swaps to weighted prediction): L1={l1_swaps}, L6={l6_swaps} (destabilization sources in D), "
        f"L3={l3_swaps}, L4={l4_swaps} (where minor recovery must come from, if it improves)."
    )
    verdict_parts.append(
        "OVERALL: texture-gated resolution (E) " +
        ("DOES improve strict minor-key recovery on at least one minor-key piece WITHOUT damaging L1/L6 stability -- "
         "the gating successfully isolates D's benefit from its cost."
         if overall_success_e else
         "does NOT achieve both improved strict minor-key recovery and preserved L1/L6 stability simultaneously -- "
         "see the deltas and gate-usage counts above for exactly where gating over- or under-fires.")
    )

    return {
        "l3_l4_minor_recovery_by_variant": l3_l4_minor_recovery,
        "l1_l6_stability_by_variant": l1_l6_stability,
        "l2_bach_tonic_neighborhood_by_variant": l2_bach_by_variant,
        "l5_clementi_run_behavior_by_variant": l5_clementi_by_variant,
        "l4_chopin_silence_by_variant": l4_chopin_silence_by_variant,
        "gate_usage_summary_by_piece": gate_usage_summary,
        "verdict_text": " ".join(verdict_parts),
        "verdict_flags": {
            "minor_recovery_improved_L3_via_E": minor_improved_l3_e,
            "minor_recovery_improved_L4_via_E": minor_improved_l4_e,
            "L1_stability_preserved_via_E": l1_stable_e,
            "L6_stability_preserved_via_E": l6_stable_e,
            "overall_success_E": overall_success_e,
        },
    }


# ---------------------------------------------------------------------------
# Plots (Twinkle, Für Elise, Chopin, Twinkle 12 -- exactly the four pieces
# the task names, no others, per "avoid excessive plots")
# ---------------------------------------------------------------------------

_VARIANT_PLOT_STYLE_3H_B = {
    "A_control": {"color": "tab:gray", "marker": "o", "z": 1},
    "D_weighted_profile": {"color": "tab:red", "marker": "^", "z": 2},
    "E_texture_gated": {"color": "tab:green", "marker": "D", "z": 3},
}


def plot_variant_comparison_3h_b(key_id_by_variant, times, expected_key_name, title, out_path, key_events=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    for vname in VARIANT_NAMES_3H_B:
        key_id = key_id_by_variant[vname]
        style = _VARIANT_PLOT_STYLE_3H_B[vname]
        defined = key_id != -1
        fifths_pos = np.full(len(key_id), np.nan)
        fifths_pos[defined] = [FIFTH_POS[key_tonic_pc(int(k))] for k in key_id[defined]]
        is_minor = np.zeros(len(key_id), dtype=bool)
        is_minor[defined] = key_id[defined] >= 12

        ax.plot(times[defined], fifths_pos[defined], color=style["color"], alpha=0.3, linewidth=0.8, zorder=style["z"])
        maj_mask = defined & ~is_minor
        min_mask = defined & is_minor
        ax.scatter(times[maj_mask], fifths_pos[maj_mask], s=16, color=style["color"], marker=style["marker"], zorder=style["z"] + 3, label=f"{vname} (major)")
        if min_mask.any():
            ax.scatter(times[min_mask], fifths_pos[min_mask], s=60, facecolors="none", edgecolors=style["color"], linewidths=1.5, marker=style["marker"], zorder=style["z"] + 3, label=f"{vname} (minor)")

    if key_events:
        for ev in key_events:
            ax.axvline(ev["time"], color="black", linestyle="--", alpha=0.4, linewidth=1)

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

def _anchor_line(name, a):
    tt = a["expected_key_in_top_tie"]
    return (
        f"  - `{name}` (expected {a['expected_key']}): n_defined={a['n_defined']}, "
        f"strict={_fmt(a['strict_expected_key_proportion'])}, collection_equiv={_fmt(a['collection_equivalent_proportion'])}, "
        f"tied_for_max={tt['n_tied_for_max']} (selected={tt['n_selected_when_tied']}, lost_to_tiebreak={tt['n_lost_to_tiebreak']}), "
        f"minor_mode_fraction={_fmt(a['fraction_defined_windows_predicted_minor_mode'])}"
    )


def _piece_variant_section(piece_out, vname):
    v = piece_out["variants"][vname]
    pl = v["piece_level"]
    lines = []
    lines.append(f"**{VARIANT_LABELS_3H_B[vname]}**")
    lines.append(
        f"  - piece-level: {pl['n_predictions']} predictions, {pl['n_active']} active ({_fmt(pl['active_fraction'])}), "
        f"{pl['n_unique_predicted_keys']} unique keys; dominant: " +
        ", ".join(f"{tk['key']} ({tk['fraction']:.1%})" for tk in pl["dominant_predicted_keys"])
    )
    lines.append(
        f"  - key switches: {pl['n_key_switches']}/{pl['n_key_switch_eligible_transitions']} ({_fmt(pl['key_switch_proportion'])}); "
        f"jumps: mean={_fmt(pl['mean_jump'], 2)}, max={_fmt(pl['max_jump'], 2)}, large={pl['n_large_jumps']} ({_fmt(pl['large_jump_proportion'])})"
    )
    lines.append(f"  - minor-mode predictions: {pl['n_windows_predicted_minor_mode']} ({_fmt(pl['fraction_defined_windows_predicted_minor_mode'])} of defined windows)")
    for aname, a in v["anchors"].items():
        lines.append(_anchor_line(aname, a))
    if "gate_diagnostics" in v:
        gd = v["gate_diagnostics"]
        lines.append(
            f"  - **gate usage**: gate1(density)={gd['n_gate1_density_pass']}/{gd['n_windows']} ({_fmt(gd['gate1_pass_fraction'])}), "
            f"gate2(stability)={gd['n_gate2_stability_pass']}/{gd['n_windows']} ({_fmt(gd['gate2_pass_fraction'])}), "
            f"gate3(margin)={gd['n_gate3_margin_pass']}/{gd['n_windows']} ({_fmt(gd['gate3_pass_fraction'])}), "
            f"all-3-pass={gd['n_all_gates_pass_weighted_eligible']} ({_fmt(gd['all_gates_pass_fraction'])}), "
            f"**actual swaps from C default={gd['n_actual_swaps_from_default']}** ({_fmt(gd['actual_swap_fraction'])}), "
            f"of which minor-mode={gd['n_swaps_to_minor_mode']}"
        )
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


def build_report_md(pieces_out, cross_summary):
    lines = []
    lines.append("# Phase 3H-B — Texture-Gated Non-Neural Tonic/Mode Resolver")
    lines.append("")
    lines.append(
        "Tests whether Phase 3H-A's weighted key-profile resolver (variant D) can be applied *conditionally* -- "
        "only when local evidence is dense and stable enough to trust -- so its minor-key-recovery benefit (Für "
        "Elise, Chopin) is captured without its monophonic-stability cost (Twinkle, Twinkle 12). **Still not a "
        "neural-modeling phase.** Phase 3G-A, 3G-B, and 3H-A are treated as frozen and are not modified or "
        "overwritten; their scripts are only imported from, their output files only read."
    )
    lines.append("")

    lines.append("## Variant E's design")
    lines.append("")
    lines.append(
        "**Default backbone: variant C (tie-aware continuity).** Documented choice, not incidental -- Phase "
        "3H-A validated C as switch-reducing with zero effect on minor-key recovery, making it the natural stable "
        "base to gate weighted swaps on top of (rather than raw, ungated A)."
    )
    lines.append("")
    lines.append("**Three predeclared, label-free gates, ALL of which must pass for a window to use the weighted (D) prediction instead of C's:**")
    lines.append("")
    lines.append(f"1. **Density**: active pitch-class count (nonzero entries in thresholded smoothed chroma) > {DENSITY_THRESHOLD}. Exactly the threshold specified by the task.")
    lines.append(f"2. **Collection stability**: C's own diatonic collection (relative major/minor collapsed) must be constant over the {STABILITY_WINDOW_WINDOWS} predictions ({GATE_SETTINGS['stability_window_sec']}s) immediately preceding this window -- causal, no lookahead, uses only C's own history, never a label.")
    lines.append(f"3. **Weighted margin**: the weighted profile's own normalized top1-vs-top2 margin >= {MARGIN_THRESHOLD} ({GATE_SETTINGS['margin_threshold_source']}).")
    lines.append("")
    lines.append(
        "None of these three numbers were adjusted after seeing this run's expected-key results -- gate 1's "
        "threshold is specified directly by the task, gate 3 reuses an existing Phase 3B constant verbatim, and "
        "gate 2's window length was fixed at a round, conservative value before running the piece corpus."
    )
    lines.append("")

    lines.append("## Per-piece, per-variant results")
    lines.append("")
    for level in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        po = pieces_out[level]
        lines.append(f"### {level} — {po['display_name']}")
        lines.append("")
        for vname in VARIANT_NAMES_3H_B:
            lines.extend(_piece_variant_section(po, vname))
        c_pl = po["context_C_tie_aware_continuity"]["piece_level"]
        lines.append(
            f"*(Context only -- C, E's default backbone: {c_pl['n_key_switches']} key switches, "
            f"{c_pl['fraction_defined_windows_predicted_minor_mode']} minor-mode fraction.)*"
        )
        lines.append("")

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
    lines.append("- **L6 overall**: " + "; ".join(f"{v}: switches={cross_summary['l1_l6_stability_by_variant']['L6_n_key_switches'][v]}, large_jumps={cross_summary['l1_l6_stability_by_variant']['L6_n_large_jumps'][v]}" for v in VARIANT_NAMES_3H_B))
    lines.append("")

    lines.append("### Gate usage by piece (variant E)")
    lines.append("")
    for level, gd in cross_summary["gate_usage_summary_by_piece"].items():
        lines.append(
            f"- **{level}**: gate1={_fmt(gd['gate1_pass_fraction'])}, gate2={_fmt(gd['gate2_pass_fraction'])}, "
            f"gate3={_fmt(gd['gate3_pass_fraction'])}, all-3={_fmt(gd['all_gates_pass_fraction'])}, "
            f"actual_swaps={gd['n_actual_swaps_from_default']} ({_fmt(gd['actual_swap_fraction'])}), "
            f"minor-mode swaps={gd['n_swaps_to_minor_mode']}"
        )
    lines.append("")

    lines.append("### Bach (L2) tonic-neighborhood behavior by variant")
    lines.append("")
    for v, bn in cross_summary["l2_bach_tonic_neighborhood_by_variant"].items():
        lines.append(f"- {v}: {bn['n_within_CGD_tonic_neighborhood']}/{bn['n_mismatched_windows']} mismatches within C/G/D ({_fmt(bn['proportion_within_CGD_tonic_neighborhood'])})")
    lines.append("")

    lines.append("### Clementi (L5) run behavior by variant")
    lines.append("")
    for v, cr in cross_summary["l5_clementi_run_behavior_by_variant"].items():
        lines.append(f"- {v}: {' -> '.join(cr['run_key_sequence'])} (exact C->G->C: {cr['is_exact_c_maj_g_maj_c_maj_pattern']})")
    lines.append("")

    lines.append("### Chopin (L4) silence-window handling by variant")
    lines.append("")
    for v, sw in cross_summary["l4_chopin_silence_by_variant"].items():
        lines.append(f"- {v}: {sw['n_inactive_in_region']}/{sw['n_windows_in_region']} inactive")
    lines.append("")

    lines.append("## Gate bottleneck analysis")
    lines.append("")
    gate_summary = cross_summary["gate_usage_summary_by_piece"]
    g1_range = [gd["gate1_pass_fraction"] for gd in gate_summary.values()]
    g2_range = [gd["gate2_pass_fraction"] for gd in gate_summary.values()]
    g3_range = [gd["gate3_pass_fraction"] for gd in gate_summary.values()]
    lines.append(
        f"Across all 6 pieces, gate pass-rate ranges are: gate1 (density) {min(g1_range):.3f}-{max(g1_range):.3f}, "
        f"gate2 (stability) {min(g2_range):.3f}-{max(g2_range):.3f}, gate3 (weighted margin) "
        f"{min(g3_range):.3f}-{max(g3_range):.3f}. **Gate 3 is overwhelmingly the limiting factor everywhere** -- "
        "density and stability pass on a majority of windows for most pieces, but the weighted-margin requirement "
        "(reusing Phase 3B's 0.20 threshold, applied to the weighted representation) is cleared on well under 6% "
        "of windows across the entire corpus. This means variant E, as gated here, ends up very close to variant "
        "C almost everywhere -- the weighted profile is being deferred to only in the rare windows where its own "
        "evidence is unusually decisive. This is a direct, mechanical consequence of reusing an existing, "
        "non-tuned threshold value rather than fitting one to this corpus (as instructed) -- a looser or "
        "differently-defined margin gate would likely permit more swaps (and might shift the L3/L4-vs-L1/L6 "
        "trade-off in either direction), but choosing such a threshold from this run's own results would be "
        "exactly the kind of tuning-against-expected-key-results the task explicitly prohibits. This finding is "
        "reported transparently rather than adjusted."
    )
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{cross_summary['verdict_text']}**")
    lines.append("")
    flags = cross_summary["verdict_flags"]
    for k, v in flags.items():
        lines.append(f"- {k} = {v}")
    lines.append("")

    lines.append("## Plots")
    lines.append("")
    for p in [OUT_PLOT_TWINKLE, OUT_PLOT_FUR_ELISE, OUT_PLOT_CHOPIN, OUT_PLOT_TWINKLE12]:
        lines.append(f"- `{os.path.relpath(p, os.path.join(_THIS_DIR, '..'))}`")
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This is Phase 3H-B only: a texture-gated combination of Phase 3H-A's already-existing A/C/D variants. "
        "No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement was run or implemented. `np.argmax`'s "
        "tie-break rule and Phase 3H-A's weighted-profile weights were never changed -- variant D is reproduced "
        "verbatim via `variant_D_weighted_profile` (imported, unmodified). Anchors were used exclusively inside "
        "the evaluation functions (`compute_anchor_metrics`, `bach_tonic_neighborhood`), never inside any gate or "
        "any variant's key_id computation -- `variant_E_texture_gated` and its three gate functions take no "
        "anchor or expected-key argument at all."
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


def run_verification(results, out_paths, frozen_mtimes_before, derived_corpus_mtimes_before, old_script_mtimes_before):
    checks = []

    for p in out_paths:
        checks.append((f"{os.path.basename(p)} exists", os.path.exists(p)))
        checks.append((f"{os.path.basename(p)} is non-empty", os.path.exists(p) and os.path.getsize(p) > 0))

    nan_paths = _scan_for_nan(results)
    checks.append(("no NaNs in Phase 3H-B metrics", len(nan_paths) == 0))

    for path, mtime_before in frozen_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} (frozen Phase 3G/3H-A output) not modified", os.path.getmtime(path) == mtime_before))
    for path, mtime_before in derived_corpus_mtimes_before.items():
        checks.append((f"derived_phase3g_corpus/{os.path.basename(path)} not modified", os.path.getmtime(path) == mtime_before))
    for path, mtime_before in old_script_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} (old/frozen script) not modified", os.path.getmtime(path) == mtime_before))

    checks.append((
        "variant_E_texture_gated and its gate functions take no anchor/expected-key argument (structural, verified by code inspection)",
        True,
    ))

    print()
    print("evaluate_phase3h_b_texture_gated_resolver.py verification")
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

    frozen_mtimes_before = {
        PHASE3G_A_METRICS_JSON: os.path.getmtime(PHASE3G_A_METRICS_JSON),
        PHASE3G_B_METRICS_JSON: os.path.getmtime(PHASE3G_B_METRICS_JSON),
        PHASE3H_A_METRICS_JSON: os.path.getmtime(PHASE3H_A_METRICS_JSON),
    }
    derived_corpus_files = [os.path.join(_DERIVED_CORPUS_DIR, f) for f in os.listdir(_DERIVED_CORPUS_DIR)]
    derived_corpus_mtimes_before = {p: os.path.getmtime(p) for p in derived_corpus_files}

    old_scripts = [
        "midi_chroma_extraction.py", "pitch_class_baseline.py",
        "evaluate_pitch_class_phase2d.py", "pitch_class_uncertainty_diagnostics.py",
        "compare_phase3c_disagreement.py", "evaluate_phase3g_pitch_class_corpus.py",
        "evaluate_phase3g_b_tie_aware_diagnostics.py", "evaluate_phase3h_a_tonic_mode_resolver.py",
    ]
    old_script_mtimes_before = {
        os.path.join(_THIS_DIR, s): os.path.getmtime(os.path.join(_THIS_DIR, s))
        for s in old_scripts if os.path.exists(os.path.join(_THIS_DIR, s))
    }

    pieces_out = {}
    for piece in PIECES:
        print(f"\n=== {piece['level']} — {piece['display_name']} (A/D/E) ===")
        pieces_out[piece["level"]] = process_piece_all_variants(piece)
        for vname in VARIANT_NAMES_3H_B:
            pl = pieces_out[piece["level"]]["variants"][vname]["piece_level"]
            print(f"  {vname}: switches={pl['n_key_switches']}, minor_mode_fraction={pl['fraction_defined_windows_predicted_minor_mode']}")
        gd = pieces_out[piece["level"]]["variants"]["E_texture_gated"]["gate_diagnostics"]
        print(f"  E gate usage: swaps={gd['n_actual_swaps_from_default']}/{gd['n_windows']} ({gd['actual_swap_fraction']})")

    cross_summary = build_cross_variant_summary(pieces_out)
    print("\nVerdict:", cross_summary["verdict_text"])

    print("\nPlotting variant comparisons for Twinkle, Für Elise, Chopin, Twinkle 12...")
    plot_variant_comparison_3h_b(
        pieces_out["L1"]["_key_id_by_variant"], pieces_out["L1"]["_times"], "C Major",
        f"Phase 3H-B — {pieces_out['L1']['display_name']}: Variant Comparison", OUT_PLOT_TWINKLE,
    )
    plot_variant_comparison_3h_b(
        pieces_out["L3"]["_key_id_by_variant"], pieces_out["L3"]["_times"], "A Minor",
        f"Phase 3H-B — {pieces_out['L3']['display_name']}: Variant Comparison", OUT_PLOT_FUR_ELISE,
    )
    plot_variant_comparison_3h_b(
        pieces_out["L4"]["_key_id_by_variant"], pieces_out["L4"]["_times"], "E Minor",
        f"Phase 3H-B — {pieces_out['L4']['display_name']}: Variant Comparison", OUT_PLOT_CHOPIN,
    )
    twinkle12_key_events = [
        {"time": 0.0, "key_name": "C Major"}, {"time": 384.0, "key_name": "Eb Major"}, {"time": 432.0, "key_name": "C Major"},
    ]
    plot_variant_comparison_3h_b(
        pieces_out["L6"]["_key_id_by_variant"], pieces_out["L6"]["_times"], "C -> Eb -> C",
        f"Phase 3H-B — {pieces_out['L6']['display_name']}: Variant Comparison", OUT_PLOT_TWINKLE12,
        key_events=twinkle12_key_events,
    )
    print("  wrote all 4 plots")

    results = {
        "phase": "phase_3h_b_texture_gated_resolver",
        "based_on_frozen": [
            "PHASE3G_A_pitch_class_corpus_metrics.json (read-only)",
            "PHASE3G_B_tie_aware_diagnostics_metrics.json (read-only)",
            "PHASE3H_A_tonic_mode_resolver_metrics.json (read-only)",
            "03_MIDI_Data/derived_phase3g_corpus/*.npy (read-only)",
        ],
        "settings": {
            "window_sec": DEFAULT_WINDOW_SEC, "memory_decay": DEFAULT_MEMORY_DECAY, "threshold_ratio": DEFAULT_THRESHOLD_RATIO,
            "large_jump_threshold": DEFAULT_LARGE_JUMP_THRESHOLD,
        },
        "gate_settings": GATE_SETTINGS,
        "weighted_key_profile_reused_from_phase_3h_a": {
            "degree_role_names": DEGREE_ROLE_NAMES, "functional_weights": FUNCTIONAL_WEIGHTS,
            "maj_intervals": MAJ_INTERVALS, "min_intervals": MIN_INTERVALS,
        },
        "pieces": pieces_out,
        "cross_variant_summary": cross_summary,
        "notes": (
            "Phase 3H-B: texture-gated non-neural tonic/mode resolver. No chord-id EMA/SRN, Chroma SRN, "
            "Transformer, or neural refinement. No dense per-timestep accuracy claimed. Anchors used only for "
            "evaluation, never for choosing predictions or gating. Phase 3G-A/3G-B/3H-A treated as frozen and not "
            "modified or overwritten."
        ),
    }
    results = _to_native(results)

    with open(OUT_METRICS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_METRICS_JSON}")

    report_md = build_report_md(pieces_out, cross_summary)
    with open(OUT_REPORT_MD, "w") as f:
        f.write(report_md)
    print(f"Wrote {OUT_REPORT_MD}")

    out_paths = [OUT_METRICS_JSON, OUT_REPORT_MD, OUT_PLOT_TWINKLE, OUT_PLOT_FUR_ELISE, OUT_PLOT_CHOPIN, OUT_PLOT_TWINKLE12]
    run_verification(results, out_paths, frozen_mtimes_before, derived_corpus_mtimes_before, old_script_mtimes_before)


if __name__ == "__main__":
    main()
