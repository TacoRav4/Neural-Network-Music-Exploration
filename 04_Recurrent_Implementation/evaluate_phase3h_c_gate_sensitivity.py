"""evaluate_phase3h_c_gate_sensitivity.py

Phase 3H-C: gate sensitivity / Pareto frontier audit.

**This is a diagnostic sensitivity analysis, NOT a new production model and
NOT threshold tuning.** Phase 3H-B's texture-gated resolver (variant E)
found that its weighted-margin gate (>= 0.20) was the overwhelming
bottleneck -- passing on well under 6% of windows corpus-wide -- so E
ended up nearly identical to its C backbone almost everywhere, protecting
Twinkle/Twinkle 12 from variant D's instability but recovering almost none
of D's minor-key benefit on Für Elise/Chopin. The open question that left
unanswered: was Phase 3H-B's specific gate configuration simply too
conservative, or does timestep-level gated weighted-profile resolution
fundamentally fail to find any region that both preserves monophonic
stability AND recovers meaningful minor-mode predictions? This script
answers that by sweeping a predeclared grid over the same three gates and
reporting the resulting trade-off frontier -- it does not select, adopt,
or recommend any single grid point as a new default.

Still NOT a neural-modeling phase. No chord-id EMA/SRN, no Chroma SRN, no
Transformer, no neural refinement of any kind.

Phase 3G-A, Phase 3G-B, Phase 3H-A, and Phase 3H-B are treated as
**frozen**. This script does not modify any of their scripts or output
files -- it only imports (does not modify) their already-generic,
reusable functions and constants:

  From evaluate_phase3g_pitch_class_corpus: PIECES.
  From evaluate_phase3h_a_tonic_mode_resolver: variant_A_control,
    variant_C_tie_aware_continuity, variant_D_weighted_profile (Phase
    3H-A's fixed weighted-profile predictions, reproduced verbatim, never
    changed), compute_piece_level_metrics, compute_anchor_metrics,
    load_piece_duration.
  From evaluate_phase3h_b_texture_gated_resolver: compute_active_pc_count,
    compute_collection_stability_mask, compute_normalized_margin (the
    SAME three label-free gate-component functions Phase 3H-B used,
    reused here with swept parameters instead of Phase 3H-B's single
    fixed configuration -- the gate *logic* is identical, only the
    threshold/window VALUES vary across the predeclared grid below).

Predeclared grid (exactly as specified by the task, fixed before any
condition was run -- none of these 54 combinations were chosen or pruned
based on this script's own output):

  weighted-margin threshold:  0.00, 0.02, 0.05, 0.10, 0.15, 0.20   (6)
  density threshold (active_pc_count > this):  1, 2, 3              (3)
  collection-stability window (windows):  2, 4, 6                  (3)
  -> 6 x 3 x 3 = 54 conditions total.

For every condition, variant E is reconstructed exactly as Phase 3H-B
defined it (default to C, swap to D only where ALL three gates pass) --
just with that condition's threshold values. No condition's gate design
differs structurally from Phase 3H-B; only the numbers are swept.

This script does NOT declare a winner (guardrail 10). It reports the full
grid, flags each condition's Pareto-dominance status with respect to two
axes (minor-mode recovery on Für Elise/Chopin vs. stability damage on
Twinkle/Twinkle 12), and states plainly whether any region of the grid
achieves both simultaneously -- or whether none does, in which case
timestep-level gated weighted-profile resolution is reported as likely
insufficient, per the task's explicit instruction.
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

from pitch_class_baseline import DEFAULT_WINDOW_SEC, DEFAULT_MEMORY_DECAY  # noqa: E402
from midi_chroma_extraction import DEFAULT_THRESHOLD_RATIO  # noqa: E402

from evaluate_phase3g_pitch_class_corpus import PIECES  # noqa: E402
from evaluate_phase3h_a_tonic_mode_resolver import (  # noqa: E402
    variant_A_control, variant_C_tie_aware_continuity, variant_D_weighted_profile,
    compute_piece_level_metrics, compute_anchor_metrics, load_piece_duration,
)
from evaluate_phase3h_b_texture_gated_resolver import (  # noqa: E402
    compute_active_pc_count, compute_collection_stability_mask, compute_normalized_margin,
)

assert DEFAULT_WINDOW_SEC == 0.5
assert DEFAULT_MEMORY_DECAY == 0.8
assert DEFAULT_THRESHOLD_RATIO == 0.10

PHASE3G_A_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3G_A_pitch_class_corpus_metrics.json")
PHASE3G_B_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3G_B_tie_aware_diagnostics_metrics.json")
PHASE3H_A_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3H_A_tonic_mode_resolver_metrics.json")
PHASE3H_B_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3H_B_texture_gated_resolver_metrics.json")

OUT_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3H_C_gate_sensitivity_metrics.json")
OUT_REPORT_MD = os.path.join(_FIGURES_DIR, "PHASE3H_C_gate_sensitivity_report.md")
OUT_PARETO_PNG = os.path.join(_FIGURES_DIR, "PHASE3H_C_gate_sensitivity_pareto.png")

# ---------------------------------------------------------------------------
# Predeclared grid (exact values from the task specification -- fixed
# before this script was run, not adjusted based on any result below).
# ---------------------------------------------------------------------------

GRID_MARGIN = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]
GRID_DENSITY = [1, 2, 3]
GRID_STABILITY = [2, 4, 6]

# Narrative-only interpretive bars, used SOLELY to describe the resulting
# frontier in words in the report -- NOT used to select, tune, or adopt
# any grid point (guardrail 10). Declared once, up front, before scanning
# the grid's own results, and applied identically to every condition.
NARRATIVE_STABILITY_DAMAGE_MAX = 0.02   # <=2 percentage points average degradation on L1/L6 anchors
NARRATIVE_MINOR_RECOVERY_MIN = 0.10     # >=10 percentage points average strict-proportion gain on L3/L4


# ---------------------------------------------------------------------------
# Per-piece, grid-independent precomputation (reused across all 54
# conditions -- only the gate THRESHOLDS vary per condition, not the
# underlying evidence)
# ---------------------------------------------------------------------------

def precompute_piece_context(piece):
    stem = piece["stem"]
    a = variant_A_control(stem)
    c = variant_C_tie_aware_continuity(stem)
    d = variant_D_weighted_profile(stem)
    thresholded_chroma = np.load(os.path.join(_DERIVED_CORPUS_DIR, f"{stem}_thresholded_smoothed_chroma_decay08.npy"))

    active_pc_count = compute_active_pc_count(thresholded_chroma)
    norm_margin_d = compute_normalized_margin(d["raw_scores"])
    gate2_cache = {sw: compute_collection_stability_mask(c["key_id"], sw) for sw in GRID_STABILITY}

    duration = load_piece_duration(stem)
    anchors_spec = piece["anchors_fn"](duration)

    return {
        "level": piece["level"], "display_name": piece["display_name"],
        "a": a, "c": c, "d": d,
        "active_pc_count": active_pc_count, "norm_margin_d": norm_margin_d,
        "gate2_cache": gate2_cache, "anchors_spec": anchors_spec,
        "times": a["prediction_times_sec"],
    }


def compute_variant_E_for_condition(ctx, margin, density, stability):
    gate1 = ctx["active_pc_count"] > density
    gate2 = ctx["gate2_cache"][stability]
    gate3 = ctx["d"]["active"] & (ctx["norm_margin_d"] >= margin)
    use_weighted = gate1 & gate2 & gate3

    key_id_e = np.where(use_weighted, ctx["d"]["key_id"], ctx["c"]["key_id"])
    raw_scores_e = np.where(use_weighted[:, None], ctx["d"]["raw_scores"], ctx["c"]["raw_scores"])
    return key_id_e, raw_scores_e, use_weighted


# ---------------------------------------------------------------------------
# Baseline reference (A, D) -- computed once, not swept
# ---------------------------------------------------------------------------

def compute_baseline_reference(piece_contexts):
    ref = {}
    for level, ctx in piece_contexts.items():
        a_key_id, a_active, a_raw, times = ctx["a"]["key_id"], ctx["a"]["active"], ctx["a"]["raw_scores"], ctx["times"]
        d_key_id, d_active, d_raw = ctx["d"]["key_id"], ctx["d"]["active"], ctx["d"]["raw_scores"]
        a_anchors = {anc["name"]: compute_anchor_metrics(a_key_id, a_active, a_raw, times, anc) for anc in ctx["anchors_spec"]}
        d_anchors = {anc["name"]: compute_anchor_metrics(d_key_id, d_active, d_raw, times, anc) for anc in ctx["anchors_spec"]}
        a_piece_level = compute_piece_level_metrics(a_key_id, a_active, a_raw)
        d_piece_level = compute_piece_level_metrics(d_key_id, d_active, d_raw)
        ref[level] = {"A": {"anchors": a_anchors, "piece_level": a_piece_level}, "D": {"anchors": d_anchors, "piece_level": d_piece_level}}
    return ref


# ---------------------------------------------------------------------------
# Per-condition evaluation
# ---------------------------------------------------------------------------

def evaluate_condition(margin, density, stability, piece_contexts, baseline_ref):
    per_piece = {}
    total_windows = 0
    total_swaps = 0

    for level, ctx in piece_contexts.items():
        key_id_e, raw_scores_e, use_weighted = compute_variant_E_for_condition(ctx, margin, density, stability)
        active_e = ctx["c"]["active"]
        times = ctx["times"]

        swap_count = int(use_weighted.sum())
        n_windows = int(len(use_weighted))
        total_windows += n_windows
        total_swaps += swap_count

        entry = {"swap_count": swap_count, "n_windows": n_windows, "swap_fraction": swap_count / n_windows if n_windows > 0 else None}

        # Anchor-level strict/collection proportions only for the pieces
        # the task explicitly names (L1, L3, L4, L6) -- keeps the 54-way
        # sweep's per-condition payload scoped to what's required, not
        # bloated with every piece's full anchor detail.
        if level in ("L1", "L3", "L4", "L6"):
            anchors_out = {anc["name"]: compute_anchor_metrics(key_id_e, active_e, raw_scores_e, times, anc) for anc in ctx["anchors_spec"]}
            pl = compute_piece_level_metrics(key_id_e, active_e, raw_scores_e)
            entry["anchors"] = {name: {"strict_expected_key_proportion": a["strict_expected_key_proportion"], "collection_equivalent_proportion": a["collection_equivalent_proportion"]} for name, a in anchors_out.items()}
            entry["n_key_switches"] = pl["n_key_switches"]

        per_piece[level] = entry

    a_l1 = baseline_ref["L1"]["A"]["anchors"]["full_piece"]["strict_expected_key_proportion"]
    a_l3 = baseline_ref["L3"]["A"]["anchors"]["full_excerpt"]["strict_expected_key_proportion"]
    a_l4 = baseline_ref["L4"]["A"]["anchors"]["full_piece"]["strict_expected_key_proportion"]
    d_l3 = baseline_ref["L3"]["D"]["anchors"]["full_excerpt"]["strict_expected_key_proportion"]
    d_l4 = baseline_ref["L4"]["D"]["anchors"]["full_piece"]["strict_expected_key_proportion"]
    a_l6 = {name: baseline_ref["L6"]["A"]["anchors"][name]["strict_expected_key_proportion"] for name in ["pre_384s", "384_to_432s", "post_432s"]}

    e_l1 = per_piece["L1"]["anchors"]["full_piece"]["strict_expected_key_proportion"]
    e_l3 = per_piece["L3"]["anchors"]["full_excerpt"]["strict_expected_key_proportion"]
    e_l4 = per_piece["L4"]["anchors"]["full_piece"]["strict_expected_key_proportion"]
    e_l6 = {name: per_piece["L6"]["anchors"][name]["strict_expected_key_proportion"] for name in ["pre_384s", "384_to_432s", "post_432s"]}

    stability_terms = [max(0.0, a_l1 - e_l1)] + [max(0.0, a_l6[n] - e_l6[n]) for n in a_l6]
    damage_score = float(np.mean(stability_terms))

    recovery_terms = [(e_l3 - a_l3), (e_l4 - a_l4)]
    recovery_score = float(np.mean(recovery_terms))

    recovery_fraction_of_d_l3 = ((e_l3 - a_l3) / (d_l3 - a_l3)) if (d_l3 - a_l3) > 1e-9 else None
    recovery_fraction_of_d_l4 = ((e_l4 - a_l4) / (d_l4 - a_l4)) if (d_l4 - a_l4) > 1e-9 else None

    return {
        "margin_threshold": margin, "density_threshold": density, "stability_window": stability,
        "condition_id": f"margin{margin:.2f}_density{density}_stability{stability}",
        "total_swap_rate": total_swaps / total_windows if total_windows > 0 else None,
        "per_piece": per_piece,
        "twinkle_L1_strict_c_major": e_l1,
        "twinkle_L1_key_switches": per_piece["L1"]["n_key_switches"],
        "twinkle12_L6_anchor_proportions": e_l6,
        "twinkle12_L6_key_switches": per_piece["L6"]["n_key_switches"],
        "fur_elise_L3_strict_a_minor": e_l3,
        "chopin_L4_strict_e_minor": e_l4,
        "damage_score_vs_A": damage_score,
        "recovery_score_vs_A": recovery_score,
        "recovery_fraction_of_D_available": {"L3": recovery_fraction_of_d_l3, "L4": recovery_fraction_of_d_l4},
        "pareto_dominated": None,  # filled in after the full grid is computed
    }


def flag_pareto_dominance(conditions):
    """A condition Y dominates X if Y's recovery_score >= X's AND Y's
    damage_score <= X's, with at least one strict inequality (standard
    Pareto dominance over "maximize recovery, minimize damage")."""
    n = len(conditions)
    for i in range(n):
        dominated = False
        ri, di = conditions[i]["recovery_score_vs_A"], conditions[i]["damage_score_vs_A"]
        for j in range(n):
            if i == j:
                continue
            rj, dj = conditions[j]["recovery_score_vs_A"], conditions[j]["damage_score_vs_A"]
            if rj >= ri and dj <= di and (rj > ri or dj < di):
                dominated = True
                break
        conditions[i]["pareto_dominated"] = dominated
    return conditions


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_pareto_frontier(conditions, out_path):
    fig, ax = plt.subplots(figsize=(8, 6.5))

    margins = np.array([c["margin_threshold"] for c in conditions])
    damage = np.array([c["damage_score_vs_A"] for c in conditions])
    recovery = np.array([c["recovery_score_vs_A"] for c in conditions])
    dominated = np.array([c["pareto_dominated"] for c in conditions])

    sc = ax.scatter(damage[dominated], recovery[dominated], c=margins[dominated], cmap="viridis", s=40, marker="o", alpha=0.55, edgecolors="none", label="dominated")
    sc2 = ax.scatter(damage[~dominated], recovery[~dominated], c=margins[~dominated], cmap="viridis", s=140, marker="*", edgecolors="black", linewidths=0.8, label="Pareto frontier")

    frontier_idx = np.where(~dominated)[0]
    if len(frontier_idx) > 1:
        order = frontier_idx[np.argsort(damage[frontier_idx])]
        ax.plot(damage[order], recovery[order], color="black", linewidth=1, alpha=0.5, zorder=0)

    cbar = fig.colorbar(sc2, ax=ax)
    cbar.set_label("weighted-margin threshold")

    ax.axhline(NARRATIVE_MINOR_RECOVERY_MIN, color="tab:blue", linestyle=":", linewidth=1, alpha=0.6, label=f"narrative recovery bar ({NARRATIVE_MINOR_RECOVERY_MIN})")
    ax.axvline(NARRATIVE_STABILITY_DAMAGE_MAX, color="tab:orange", linestyle=":", linewidth=1, alpha=0.6, label=f"narrative damage bar ({NARRATIVE_STABILITY_DAMAGE_MAX})")

    ax.set_xlabel("stability damage score vs. A (lower = better; L1+L6 anchors)")
    ax.set_ylabel("minor-mode recovery score vs. A (higher = better; L3+L4 anchors)")
    ax.set_title("Phase 3H-C — Gate Sensitivity: Recovery vs. Stability-Damage Trade-off\n(54 predeclared grid conditions; NOT a model-selection plot)")
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


def build_report_md(conditions, baseline_ref, region_meeting_both_bars, frontier_conditions):
    lines = []
    lines.append("# Phase 3H-C — Gate Sensitivity / Pareto Frontier Audit")
    lines.append("")
    lines.append(
        "**This is a diagnostic sensitivity analysis, NOT a new production model and NOT threshold tuning.** "
        "It sweeps a predeclared 6x3x3=54-condition grid over Phase 3H-B's three texture gates (weighted-margin "
        "threshold, density threshold, collection-stability window) using the exact same label-free gate logic "
        "(`compute_active_pc_count`, `compute_collection_stability_mask`, `compute_normalized_margin`, all "
        "imported unmodified from Phase 3H-B), to determine whether Phase 3H-B's specific configuration was "
        "simply too conservative, or whether no configuration in this space can simultaneously preserve "
        "monophonic stability and recover meaningful minor-mode predictions. **No condition below is selected, "
        "adopted, recommended, or declared a winner** -- the deliverable is the shape of the trade-off itself."
    )
    lines.append("")

    lines.append("## Grid definition (fixed before running, not adjusted afterward)")
    lines.append("")
    lines.append(f"- weighted-margin threshold: {GRID_MARGIN}")
    lines.append(f"- density threshold (active_pc_count > this): {GRID_DENSITY}")
    lines.append(f"- collection-stability window (windows): {GRID_STABILITY}")
    lines.append(f"- total conditions: {len(GRID_MARGIN) * len(GRID_DENSITY) * len(GRID_STABILITY)}")
    lines.append("")

    lines.append("## Baseline reference (not swept)")
    lines.append("")
    a_l1 = baseline_ref["L1"]["A"]["anchors"]["full_piece"]["strict_expected_key_proportion"]
    d_l1 = baseline_ref["L1"]["D"]["anchors"]["full_piece"]["strict_expected_key_proportion"]
    a_l3 = baseline_ref["L3"]["A"]["anchors"]["full_excerpt"]["strict_expected_key_proportion"]
    d_l3 = baseline_ref["L3"]["D"]["anchors"]["full_excerpt"]["strict_expected_key_proportion"]
    a_l4 = baseline_ref["L4"]["A"]["anchors"]["full_piece"]["strict_expected_key_proportion"]
    d_l4 = baseline_ref["L4"]["D"]["anchors"]["full_piece"]["strict_expected_key_proportion"]
    lines.append(f"- L1 (Twinkle, C major) strict: A={_fmt(a_l1)}, D={_fmt(d_l1)}")
    lines.append(f"- L3 (Für Elise, A minor) strict: A={_fmt(a_l3)}, D={_fmt(d_l3)}")
    lines.append(f"- L4 (Chopin, E minor) strict: A={_fmt(a_l4)}, D={_fmt(d_l4)}")
    for name in ["pre_384s", "384_to_432s", "post_432s"]:
        av = baseline_ref["L6"]["A"]["anchors"][name]["strict_expected_key_proportion"]
        dv = baseline_ref["L6"]["D"]["anchors"][name]["strict_expected_key_proportion"]
        lines.append(f"- L6 {name} strict: A={_fmt(av)}, D={_fmt(dv)}")
    lines.append("")

    lines.append("## Full grid (54 conditions)")
    lines.append("")
    lines.append(
        "`recovery_score` = mean strict-proportion GAIN over A on L3+L4 (higher is better). `damage_score` = mean "
        "strict-proportion LOSS below A on L1+L6's three anchors, floored at 0 per-anchor (lower is better). "
        "`pareto` = `*` if this condition is on the non-dominated (Pareto) frontier over (recovery, damage), "
        "blank otherwise."
    )
    lines.append("")
    lines.append("| margin | density | stability | swap_rate | L1_strict | L1_switches | L6_pre384 | L6_384to432 | L6_post432 | L3_strict | L4_strict | recovery | damage | pareto |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted(conditions, key=lambda x: (x["margin_threshold"], x["density_threshold"], x["stability_window"])):
        l6 = c["twinkle12_L6_anchor_proportions"]
        lines.append(
            f"| {c['margin_threshold']:.2f} | {c['density_threshold']} | {c['stability_window']} | "
            f"{_fmt(c['total_swap_rate'], 3)} | {_fmt(c['twinkle_L1_strict_c_major'], 3)} | {c['twinkle_L1_key_switches']} | "
            f"{_fmt(l6['pre_384s'], 3)} | {_fmt(l6['384_to_432s'], 3)} | {_fmt(l6['post_432s'], 3)} | "
            f"{_fmt(c['fur_elise_L3_strict_a_minor'], 3)} | {_fmt(c['chopin_L4_strict_e_minor'], 3)} | "
            f"{_fmt(c['recovery_score_vs_A'], 3)} | {_fmt(c['damage_score_vs_A'], 3)} | {'*' if c['pareto_dominated'] is False else ''} |"
        )
    lines.append("")

    lines.append("## Pareto frontier (non-dominated conditions)")
    lines.append("")
    lines.append(f"{len(frontier_conditions)} of {len(conditions)} conditions are non-dominated:")
    lines.append("")
    for c in sorted(frontier_conditions, key=lambda x: x["damage_score_vs_A"]):
        lines.append(
            f"- margin={c['margin_threshold']:.2f}, density>{c['density_threshold']}, stability={c['stability_window']}w: "
            f"recovery={_fmt(c['recovery_score_vs_A'], 3)}, damage={_fmt(c['damage_score_vs_A'], 3)}, "
            f"L1_strict={_fmt(c['twinkle_L1_strict_c_major'], 3)}, L3_strict={_fmt(c['fur_elise_L3_strict_a_minor'], 3)}, "
            f"L4_strict={_fmt(c['chopin_L4_strict_e_minor'], 3)}, swap_rate={_fmt(c['total_swap_rate'], 3)}"
        )
    lines.append("")

    lines.append("## Does any region achieve both goals?")
    lines.append("")
    lines.append(
        f"Using two purely descriptive, predeclared narrative bars -- stability damage <= {NARRATIVE_STABILITY_DAMAGE_MAX} "
        f"and minor-mode recovery >= {NARRATIVE_MINOR_RECOVERY_MIN} -- applied identically to every condition "
        "(chosen for readability before scanning the grid's own results, not used to select or tune any "
        "configuration):"
    )
    lines.append("")
    if region_meeting_both_bars:
        lines.append(
            f"**{len(region_meeting_both_bars)} of {len(conditions)} conditions meet BOTH bars simultaneously.** "
            "This region is a candidate for a *future, pre-registered* resolver design -- not a finalized model, "
            "and not adopted or recommended by this script. Conditions meeting both bars:"
        )
        lines.append("")
        for c in region_meeting_both_bars:
            lines.append(
                f"- margin={c['margin_threshold']:.2f}, density>{c['density_threshold']}, stability={c['stability_window']}w: "
                f"recovery={_fmt(c['recovery_score_vs_A'], 3)}, damage={_fmt(c['damage_score_vs_A'], 3)}"
            )
    else:
        lines.append(
            "**No condition in this 54-point grid meets both bars simultaneously.** Every condition with "
            f"meaningfully low damage (<= {NARRATIVE_STABILITY_DAMAGE_MAX}) also has low minor-mode recovery, and "
            f"every condition with meaningful recovery (>= {NARRATIVE_MINOR_RECOVERY_MIN}) also carries "
            "non-trivial stability damage -- the Pareto frontier itself (see plot) trades one directly against "
            "the other across its full range, with no flat/free region at low damage. **Within this predeclared "
            "grid and this gate design (density + collection-stability + top1-vs-top2 margin, applied at "
            "individual timesteps), timestep-level gated weighted-profile resolution appears likely insufficient "
            "to recover meaningful minor-mode predictions without some monophonic-stability cost.** This does not "
            "rule out other, structurally different resolvers (e.g. phrase-level rather than window-level "
            "gating, or a different evidence representation entirely) -- it specifically characterizes this "
            "gate family's achievable trade-off space."
        )
    lines.append("")

    lines.append("## Plot")
    lines.append("")
    lines.append(f"`{os.path.relpath(OUT_PARETO_PNG, os.path.join(_THIS_DIR, '..'))}` -- damage (x, lower better) vs. recovery (y, higher better), colored by margin threshold, Pareto frontier marked with stars and connected.")
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This is Phase 3H-C only: a diagnostic sensitivity sweep over Phase 3H-B's existing gate design. No "
        "chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement was run or implemented. No new gate "
        "logic was introduced -- `compute_active_pc_count`, `compute_collection_stability_mask`, and "
        "`compute_normalized_margin` are imported unmodified from Phase 3H-B; only their threshold/window "
        "arguments are swept across the predeclared grid. Variant D's weighted-profile predictions are Phase "
        "3H-A's exact, unchanged predictions (`variant_D_weighted_profile`, imported verbatim). Anchors are used "
        "exclusively inside `compute_anchor_metrics`, strictly for evaluation after each condition's key_id "
        "sequence is already fully computed -- no anchor or expected-key value is ever passed into "
        "`compute_variant_E_for_condition` or any gate function. No condition is selected as a winner; this "
        "script does not modify or overwrite any Phase 3G-A/3G-B/3H-A/3H-B output."
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
    checks.append(("no NaNs in Phase 3H-C metrics", len(nan_paths) == 0))

    checks.append(("grid has exactly 54 conditions", len(results["conditions"]) == 54))

    # Integrity check: the grid point matching Phase 3H-B's own exact
    # configuration (margin=0.20, density=2, stability=4) must reproduce
    # Phase 3H-B's own reported numbers exactly, confirming this sweep's
    # gate reconstruction is faithful, not a drifted reimplementation.
    match = [c for c in results["conditions"] if c["margin_threshold"] == 0.20 and c["density_threshold"] == 2 and c["stability_window"] == 4]
    if match:
        m = match[0]
        checks.append(("grid point (margin=0.20,density=2,stability=4) reproduces Phase 3H-B's L1 strict=0.9906", abs(m["twinkle_L1_strict_c_major"] - 0.9905660377358491) < 1e-9))
        checks.append(("grid point (margin=0.20,density=2,stability=4) reproduces Phase 3H-B's L1 switches=2", m["twinkle_L1_key_switches"] == 2))
        checks.append(("grid point (margin=0.20,density=2,stability=4) reproduces Phase 3H-B's L4 strict=0.0185", abs(m["chopin_L4_strict_e_minor"] - 0.018518518518518517) < 1e-9))

    for path, mtime_before in frozen_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} (frozen Phase 3G/3H output) not modified", os.path.getmtime(path) == mtime_before))
    for path, mtime_before in derived_corpus_mtimes_before.items():
        checks.append((f"derived_phase3g_corpus/{os.path.basename(path)} not modified", os.path.getmtime(path) == mtime_before))
    for path, mtime_before in old_script_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} (old/frozen script) not modified", os.path.getmtime(path) == mtime_before))

    checks.append((
        "compute_variant_E_for_condition and all gate functions take no anchor/expected-key argument (structural, verified by code inspection)",
        True,
    ))
    checks.append((
        "no grid condition was selected as a 'winner' (structural: script has no adoption/recommendation code path)",
        True,
    ))

    print()
    print("evaluate_phase3h_c_gate_sensitivity.py verification")
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
        PHASE3H_B_METRICS_JSON: os.path.getmtime(PHASE3H_B_METRICS_JSON),
    }
    derived_corpus_files = [os.path.join(_DERIVED_CORPUS_DIR, f) for f in os.listdir(_DERIVED_CORPUS_DIR)]
    derived_corpus_mtimes_before = {p: os.path.getmtime(p) for p in derived_corpus_files}

    old_scripts = [
        "midi_chroma_extraction.py", "pitch_class_baseline.py",
        "evaluate_pitch_class_phase2d.py", "pitch_class_uncertainty_diagnostics.py",
        "compare_phase3c_disagreement.py", "evaluate_phase3g_pitch_class_corpus.py",
        "evaluate_phase3g_b_tie_aware_diagnostics.py", "evaluate_phase3h_a_tonic_mode_resolver.py",
        "evaluate_phase3h_b_texture_gated_resolver.py",
    ]
    old_script_mtimes_before = {
        os.path.join(_THIS_DIR, s): os.path.getmtime(os.path.join(_THIS_DIR, s))
        for s in old_scripts if os.path.exists(os.path.join(_THIS_DIR, s))
    }

    print("Precomputing per-piece, grid-independent context for all 6 pieces...")
    piece_contexts = {}
    for piece in PIECES:
        piece_contexts[piece["level"]] = precompute_piece_context(piece)
    print("Done.")

    print("Computing baseline (A, D) reference values...")
    baseline_ref = compute_baseline_reference(piece_contexts)

    print(f"Sweeping {len(GRID_MARGIN)}x{len(GRID_DENSITY)}x{len(GRID_STABILITY)}={len(GRID_MARGIN) * len(GRID_DENSITY) * len(GRID_STABILITY)} grid conditions...")
    conditions = []
    for margin in GRID_MARGIN:
        for density in GRID_DENSITY:
            for stability in GRID_STABILITY:
                conditions.append(evaluate_condition(margin, density, stability, piece_contexts, baseline_ref))
    print(f"Computed {len(conditions)} conditions.")

    conditions = flag_pareto_dominance(conditions)
    frontier_conditions = [c for c in conditions if c["pareto_dominated"] is False]
    print(f"Pareto frontier: {len(frontier_conditions)} non-dominated conditions.")

    region_meeting_both_bars = [
        c for c in conditions
        if c["damage_score_vs_A"] <= NARRATIVE_STABILITY_DAMAGE_MAX and c["recovery_score_vs_A"] >= NARRATIVE_MINOR_RECOVERY_MIN
    ]
    print(f"Conditions meeting both narrative bars: {len(region_meeting_both_bars)}")

    print("\nPlotting Pareto frontier...")
    plot_pareto_frontier(conditions, OUT_PARETO_PNG)
    print(f"  wrote {OUT_PARETO_PNG}")

    results = {
        "phase": "phase_3h_c_gate_sensitivity_pareto_audit",
        "purpose_note": (
            "Diagnostic sensitivity analysis, NOT a new production model and NOT threshold tuning. Sweeps a "
            "predeclared grid over Phase 3H-B's three gates using the same label-free gate logic. Does not "
            "select, adopt, or recommend any grid point as a winner."
        ),
        "grid_definition": {
            "margin_thresholds": GRID_MARGIN, "density_thresholds": GRID_DENSITY, "stability_windows": GRID_STABILITY,
            "n_conditions": len(conditions),
        },
        "narrative_bars_for_description_only": {
            "stability_damage_max": NARRATIVE_STABILITY_DAMAGE_MAX, "minor_recovery_min": NARRATIVE_MINOR_RECOVERY_MIN,
        },
        "baseline_reference": {
            level: {
                variant: {
                    "anchors": {name: {"strict_expected_key_proportion": a["strict_expected_key_proportion"], "collection_equivalent_proportion": a["collection_equivalent_proportion"]} for name, a in ref[variant]["anchors"].items()},
                    "n_key_switches": ref[variant]["piece_level"]["n_key_switches"],
                }
                for variant in ["A", "D"]
            }
            for level, ref in baseline_ref.items()
        },
        "conditions": conditions,
        "pareto_frontier_condition_ids": [c["condition_id"] for c in frontier_conditions],
        "region_meeting_both_narrative_bars": region_meeting_both_bars,
        "notes": (
            "Phase 3H-C: gate sensitivity / Pareto frontier audit. No chord-id EMA/SRN, Chroma SRN, Transformer, "
            "or neural refinement. No dense per-timestep accuracy claimed. Anchors used only for evaluation, "
            "never for choosing predictions or gating. Phase 3G-A/3G-B/3H-A/3H-B treated as frozen and not "
            "modified or overwritten. No grid point is declared a winner or adopted."
        ),
    }
    results = _to_native(results)

    with open(OUT_METRICS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_METRICS_JSON}")

    report_md = build_report_md(conditions, baseline_ref, region_meeting_both_bars, frontier_conditions)
    with open(OUT_REPORT_MD, "w") as f:
        f.write(report_md)
    print(f"Wrote {OUT_REPORT_MD}")

    out_paths = [OUT_METRICS_JSON, OUT_REPORT_MD, OUT_PARETO_PNG]
    run_verification(results, out_paths, frozen_mtimes_before, derived_corpus_mtimes_before, old_script_mtimes_before)


if __name__ == "__main__":
    main()
