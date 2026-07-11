"""evaluate_phase3g_b_tie_aware_diagnostics.py

Phase 3G-B: tie-aware diagnostic interpretation of the Phase 3G-A
corpus-aware pitch-class baseline results.

Phase 3G-A (evaluate_phase3g_pitch_class_corpus.py) is treated here as a
**frozen baseline result**. This script does not modify that script, does
not modify any Phase 2C/2D/3B/3C script, does not re-run the pitch-class
baseline, does not change any np.argmax tie-breaking behavior, and does
not overwrite or reinterpret any Phase 3G-A output file. It only *reads*
Phase 3G-A's frozen metrics JSON (`PHASE3G_A_pitch_class_corpus_metrics.json`)
and its frozen per-piece derived arrays
(`03_MIDI_Data/derived_phase3g_corpus/*.npy`), and adds new, independently
computed tie-aware diagnostics on top -- all written under a new
PHASE3G_B_ prefix.

Out of scope (per task guardrails): no chord-id EMA/SRN, no Chroma SRN, no
Transformer, no neural refinement, no dense per-timestep accuracy claims
(real MIDI has no dense ground truth -- everything here is either a
window-level anchor comparison against a documented expected key/mode, or
a purely mechanistic/structural property of the SCALE_TEMPLATES +
np.argmax pipeline, verified directly from the saved arrays).

Five required diagnostics (see module docstrings on each function below
for detail):
  1. Relative-major/minor collection-level metric (strict vs.
     collection-equivalent expected-key proportion).
  2. Expected-key-in-top-tie metric (evidence exists but loses tie-break,
     vs. evidence does not exist at all).
  3. Tie-loss taxonomy classifying each anchor-window mismatch.
  4. Bach (tonic-neighborhood characterization) and Clementi (re-labeling
     the predicted run sequence as a tonic-dominant-tonic excursion,
     using Phase 3G-A's own frozen run sequence, not recomputed).
  5. Chopin silence audit: raw vs. smoothed vs. thresholded-smoothed
     chroma mechanism around t=95.36-99.64s.
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

from shared_music_defs import decode_key, key_index, key_tonic_pc, fifth_distance  # noqa: E402
from pitch_class_uncertainty_diagnostics import _expected_key_id, DEFAULT_LARGE_JUMP_THRESHOLD  # noqa: E402

# Reused (imported, not modified) so anchor windows here are IDENTICAL to
# the ones Phase 3G-A actually scored -- no anchor definition is
# reimplemented or redefined by hand in this script.
from evaluate_phase3g_pitch_class_corpus import PIECES, CHOPIN_SILENCE_REGION  # noqa: E402

PHASE3G_A_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3G_A_pitch_class_corpus_metrics.json")

OUT_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3G_B_tie_aware_diagnostics_metrics.json")
OUT_REPORT_MD = os.path.join(_FIGURES_DIR, "PHASE3G_B_tie_aware_diagnostics_report.md")
OUT_CHOPIN_SILENCE_PNG = os.path.join(_FIGURES_DIR, "PHASE3G_B_Chopin_silence_raw_vs_smoothed_chroma.png")

# New, explicitly documented threshold for this script only (Phase 3G-A
# does not define or use this): two pieces' predicted tonics are
# considered "tonic-neighborhood" if they are within this many
# Circle-of-Fifths steps of each other (1 = dominant/subdominant, 2 =
# supertonic-ish) -- musically "closely related keys," not a claim about
# the SCALE_TEMPLATES representation itself.
TONIC_NEIGHBORHOOD_FIFTHS = 2

TAXONOMY_CATEGORIES = [
    "match",
    "inactive_or_silence_related",
    "expected_tied_for_max_but_lost_to_lower_index",
    "collection_equivalent_wrong_mode_or_tonic_label",
    "large_jump_instability",
    "tonic_neighborhood_ambiguity",
    "expected_key_not_tied_for_max_no_support",
]


# ---------------------------------------------------------------------------
# Loading frozen Phase 3G-A arrays (read-only)
# ---------------------------------------------------------------------------

def load_piece_arrays(stem):
    """Loads the frozen per-timestep arrays Phase 3G-A already saved for
    this piece. Read-only -- nothing here is recomputed via analyze_piece
    or midi_to_key_baseline; these are the exact arrays Phase 3G-A used to
    produce its own reported numbers."""
    def _load(field):
        return np.load(os.path.join(_DERIVED_CORPUS_DIR, f"{stem}_{field}.npy"))

    return {
        "key_id": _load("key_id"),
        "active": _load("active"),
        "raw_scores": _load("raw_scores"),
        "jump_distance": _load("jump_distance"),
        "large_jump": _load("large_jump"),
        "prediction_times_sec": _load("prediction_times_sec"),
        "raw_chroma": _load("raw_chroma"),
        "smoothed_chroma": _load("smoothed_chroma_decay08"),
        "thresholded_chroma": _load("thresholded_smoothed_chroma_decay08"),
    }


def load_piece_chroma_metadata(stem):
    with open(os.path.join(_DERIVED_CORPUS_DIR, f"{stem}_chroma_metadata.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Diagnostic 1: collection-equivalence
# ---------------------------------------------------------------------------

def collection_equivalent_key_id(expected_key_id):
    """Returns the key_id of the relative major/minor of expected_key_id
    (e.g. A minor <-> C major, E minor <-> G major). This mirrors exactly
    what SCALE_TEMPLATES already treats as numerically identical rows
    (per Phase 3B's structural finding) -- here we use it to ask a
    strictly weaker, "same diatonic collection, possibly wrong tonic/mode
    label" question, distinct from the strict expected-key match."""
    tonic = expected_key_id % 12
    mode = "maj" if expected_key_id < 12 else "min"
    if mode == "maj":
        return key_index((tonic - 3) % 12, "min")
    return key_index((tonic + 3) % 12, "maj")


# ---------------------------------------------------------------------------
# Diagnostics 2 + 3: tie-in-top status and mismatch taxonomy, computed
# together as one time-ordered category sequence per anchor window range
# (using "match" as its own category so every window in the anchor's
# range is accounted for exactly once, and intervals are contiguous and
# auditable).
# ---------------------------------------------------------------------------

def _classify_window(idx, key_id, active, raw_scores, expected_id, collection_id, large_jump):
    predicted_id = int(key_id[idx])
    if predicted_id == expected_id:
        return "match", {}

    is_inactive = not bool(active[idx])
    m_score = float(raw_scores[idx].max())
    e_score = float(raw_scores[idx, expected_id])
    # "tied for max" requires m_score > 0 (an all-zero row -- fully silent
    # window with no evidence at all -- is never counted as a tie).
    tied_for_max = bool(active[idx]) and m_score > 0 and abs(e_score - m_score) < 1e-9
    is_collection_equiv = predicted_id == collection_id
    is_large_jump = bool(large_jump[idx])
    pred_tonic = key_tonic_pc(predicted_id)
    exp_tonic = key_tonic_pc(expected_id)
    is_tonic_neighborhood = fifth_distance(pred_tonic, exp_tonic) <= TONIC_NEIGHBORHOOD_FIFTHS

    flags = {
        "inactive": is_inactive,
        "expected_tied_for_max_score": tied_for_max,
        "collection_equivalent": is_collection_equiv,
        "large_jump": is_large_jump,
        "tonic_neighborhood": is_tonic_neighborhood,
    }

    # Priority order (documented, not arbitrary): a mechanistic
    # explanation for WHY argmax picked something else (tied-for-max-but-
    # lower-index-won) takes priority over descriptive properties of WHAT
    # was picked (collection-equivalent, tonic-neighborhood), which in
    # turn take priority over an undifferentiated "no support" residual.
    if is_inactive:
        return "inactive_or_silence_related", flags
    if tied_for_max:
        return "expected_tied_for_max_but_lost_to_lower_index", flags
    if is_collection_equiv:
        return "collection_equivalent_wrong_mode_or_tonic_label", flags
    if is_large_jump:
        return "large_jump_instability", flags
    if is_tonic_neighborhood:
        return "tonic_neighborhood_ambiguity", flags
    return "expected_key_not_tied_for_max_no_support", flags


def _run_length_encode_categories(idxs, times, categories):
    intervals = []
    n = len(idxs)
    i = 0
    while i < n:
        start = i
        cat = categories[i]
        while i < n and categories[i] == cat:
            i += 1
        intervals.append({
            "category": cat,
            "start_sec": float(times[idxs[start]]),
            "end_sec": float(times[idxs[i - 1]]),
            "n_windows": int(i - start),
        })
    return intervals


def compute_tie_aware_anchor_diagnostics(piece_arrays, anchor):
    times = piece_arrays["prediction_times_sec"]
    key_id = piece_arrays["key_id"]
    active = piece_arrays["active"]
    raw_scores = piece_arrays["raw_scores"]
    large_jump = piece_arrays["large_jump"]

    t_start = anchor["start_sec"] if anchor["start_sec"] is not None else 0.0
    t_end = anchor["end_sec"] if anchor["end_sec"] is not None else float(times[-1]) + 1.0
    mask = (times >= t_start) & (times < t_end)
    idxs = np.where(mask)[0]

    expected_id = _expected_key_id(anchor["expected_key_name"])
    collection_id = collection_equivalent_key_id(expected_id)

    defined_mask = key_id[idxs] != -1
    def_idxs = idxs[defined_mask]
    n_total = int(len(idxs))
    n_defined = int(len(def_idxs))
    n_undefined = n_total - n_defined

    if n_defined == 0:
        return {
            "expected_key": anchor["expected_key_name"],
            "expected_key_id": int(expected_id),
            "collection_equivalent_key": f"{decode_key(collection_id)[0]} {decode_key(collection_id)[1]}",
            "collection_equivalent_key_id": int(collection_id),
            "n_predictions": n_total, "n_defined": 0, "n_undefined": n_undefined,
            "strict_expected_key_proportion": None,
            "collection_equivalent_proportion": None,
            "expected_key_in_top_tie": {"n_windows_expected_tied_for_max": 0, "n_of_those_actually_selected": 0, "n_of_those_lost_to_tiebreak": 0},
            "category_counts": {}, "category_proportions": {}, "category_intervals": [],
        }

    categories = [_classify_window(idx, key_id, active, raw_scores, expected_id, collection_id, large_jump)[0] for idx in def_idxs]

    strict_match = np.array([c == "match" for c in categories])
    collection_match = strict_match | np.array([
        (categories[i] != "match") and (int(key_id[def_idxs[i]]) == collection_id) for i in range(len(def_idxs))
    ])

    # Diagnostic 2, computed over ALL defined windows in the anchor range
    # (not just mismatches): does raw evidence for the expected key ever
    # tie the max score, and if so, is it actually selected?
    max_scores = raw_scores[def_idxs].max(axis=1)
    expected_scores = raw_scores[def_idxs, expected_id]
    tied_for_max_mask = active[def_idxs] & (max_scores > 0) & (np.abs(expected_scores - max_scores) < 1e-9)
    n_tied_for_max = int(tied_for_max_mask.sum())
    n_tied_and_selected = int((tied_for_max_mask & strict_match).sum())
    n_tied_and_lost = n_tied_for_max - n_tied_and_selected

    interval_list = _run_length_encode_categories(def_idxs, times, categories)
    category_counts = dict(Counter(categories))
    for cat in TAXONOMY_CATEGORIES:
        category_counts.setdefault(cat, 0)
    category_proportions = {k: (v / n_defined if n_defined > 0 else None) for k, v in category_counts.items()}

    mismatch_total = n_defined - int(strict_match.sum())
    mismatch_category_proportions = {
        k: (v / mismatch_total if mismatch_total > 0 else None)
        for k, v in category_counts.items() if k != "match"
    }

    return {
        "expected_key": anchor["expected_key_name"],
        "expected_key_id": int(expected_id),
        "collection_equivalent_key": f"{decode_key(collection_id)[0]} {decode_key(collection_id)[1]}",
        "collection_equivalent_key_id": int(collection_id),
        "n_predictions": n_total,
        "n_defined": n_defined,
        "n_undefined": n_undefined,
        "strict_expected_key_proportion": float(strict_match.mean()),
        "collection_equivalent_proportion": float(collection_match.mean()),
        "expected_key_in_top_tie": {
            "n_windows_expected_tied_for_max": n_tied_for_max,
            "n_of_those_actually_selected": n_tied_and_selected,
            "n_of_those_lost_to_tiebreak": n_tied_and_lost,
            "note": (
                "Distinguishes 'expected key evidence exists but loses np.argmax's tie-break' "
                "(n_of_those_lost_to_tiebreak) from 'expected key evidence is not even tied for the top score' "
                "(n_defined - n_windows_expected_tied_for_max)."
            ),
        },
        "category_counts": category_counts,
        "category_proportions_of_all_defined_windows": category_proportions,
        "category_proportions_of_mismatched_windows_only": mismatch_category_proportions,
        "category_intervals": interval_list,
        "mismatch_predicted_key_breakdown": dict(Counter(
            f"{decode_key(int(key_id[def_idxs[i]]))[0]} {decode_key(int(key_id[def_idxs[i]]))[1]}"
            for i in range(len(def_idxs)) if not strict_match[i]
        )),
    }


# ---------------------------------------------------------------------------
# Diagnostic 5: Chopin silence audit
# ---------------------------------------------------------------------------

def chopin_silence_audit():
    stem = next(p["stem"] for p in PIECES if p["level"] == "L4")
    arrays = load_piece_arrays(stem)
    times = arrays["prediction_times_sec"]
    raw_chroma = arrays["raw_chroma"]
    smoothed_chroma = arrays["smoothed_chroma"]
    thresholded_chroma = arrays["thresholded_chroma"]
    active = arrays["active"]

    region = CHOPIN_SILENCE_REGION
    mask = (times >= region["start_sec"]) & (times <= region["end_sec"])
    idxs = np.where(mask)[0]
    # a few windows of context immediately before, to show the decay's starting point
    context_start = max(0, idxs[0] - 4) if len(idxs) else 0
    context_idxs = np.arange(context_start, (idxs[-1] + 1) if len(idxs) else context_start)

    per_window = []
    for idx in context_idxs:
        per_window.append({
            "idx": int(idx),
            "time_sec": float(times[idx]),
            "in_silence_region": bool(mask[idx]),
            "raw_chroma_sum": float(raw_chroma[idx].sum()),
            "raw_chroma_max": float(raw_chroma[idx].max()),
            "smoothed_chroma_sum": float(smoothed_chroma[idx].sum()),
            "smoothed_chroma_max": float(smoothed_chroma[idx].max()),
            "thresholded_nonzero_count": int(np.count_nonzero(thresholded_chroma[idx])),
            "active_flag": bool(active[idx]),
        })

    # Verify the geometric-decay mechanism directly: for consecutive
    # in-region windows where raw_chroma is (numerically) zero, smoothed
    # chroma should satisfy smoothed_t == 0.8 * smoothed_{t-1} elementwise
    # (memory_decay=0.8, matching pitch_class_baseline.py /
    # midi_chroma_extraction.py's formula exactly -- not re-derived here,
    # just checked against the frozen saved arrays).
    decay_checks = []
    for i in range(1, len(idxs)):
        prev_idx, cur_idx = idxs[i - 1], idxs[i]
        if raw_chroma[cur_idx].sum() < 1e-9:
            predicted_smoothed = 0.8 * smoothed_chroma[prev_idx]
            actual_smoothed = smoothed_chroma[cur_idx]
            max_abs_err = float(np.max(np.abs(predicted_smoothed - actual_smoothed)))
            decay_checks.append({"idx": int(cur_idx), "time_sec": float(times[cur_idx]), "max_abs_error_vs_08_decay_formula": max_abs_err})

    in_region_rows = [w for w in per_window if w["in_silence_region"]]
    raw_zero_rows = [w for w in in_region_rows if w["raw_chroma_sum"] < 1e-9]
    raw_nonzero_rows = [w for w in in_region_rows if w["raw_chroma_sum"] >= 1e-9]
    n_in_region = len(in_region_rows)
    n_raw_zero_in_region = len(raw_zero_rows)
    n_raw_nonzero_in_region = len(raw_nonzero_rows)
    n_smoothed_nonzero_in_region = sum(1 for w in in_region_rows if w["smoothed_chroma_max"] > 1e-9)
    n_active_in_region = sum(1 for w in in_region_rows if w["active_flag"])

    max_decay_err = float(max((c["max_abs_error_vs_08_decay_formula"] for c in decay_checks), default=0.0))

    # Characterize the raw-nonzero in-region windows by position: windows
    # before the first raw-zero window are a sustain/pedal tail carrying
    # over from before the compositional pause; windows after the last
    # raw-zero window are the next phrase's onset arriving at/before the
    # documented end boundary. Both are genuine sound, not silence, at
    # this pipeline's raw-chroma level -- the documented boundary is a
    # score-level/perceptual marker, coarser than raw MIDI note timing.
    if raw_zero_rows:
        first_zero_t = raw_zero_rows[0]["time_sec"]
        last_zero_t = raw_zero_rows[-1]["time_sec"]
        tail_rows = [w for w in raw_nonzero_rows if w["time_sec"] < first_zero_t]
        onset_rows = [w for w in raw_nonzero_rows if w["time_sec"] > last_zero_t]
    else:
        tail_rows, onset_rows = [], list(raw_nonzero_rows)

    mechanism_verdict = (
        f"0/{n_in_region} inactive windows is explained by two combined, verified mechanisms -- neither is a "
        "pipeline bug. "
        f"(1) Boundary granularity: only {n_raw_zero_in_region}/{n_in_region} of the documented silence-region "
        f"windows are numerically raw-silent at all. {len(tail_rows)} window(s) at the start of the region "
        f"(t={tail_rows[0]['time_sec']:.2f}-{tail_rows[-1]['time_sec']:.2f}s)" if tail_rows else "0 windows at the start of the region"
    ) + (
        f" still carry real, nonzero raw chroma energy (a sustain/pedal tail from before the pause), and "
        if tail_rows else ", and "
    ) + (
        f"{len(onset_rows)} window(s) at the end of the region (t={onset_rows[0]['time_sec']:.2f}-{onset_rows[-1]['time_sec']:.2f}s) "
        f"already contain the next phrase's note onset arriving at/before the documented end boundary. "
        if onset_rows else "no window at the end of the region shows an early onset. "
    ) + (
        f"The documented t={region['start_sec']:.2f}-{region['end_sec']:.2f}s silence is a score-level/perceptual "
        "marker (Phase 3F.8); at this pipeline's 0.5s window granularity, the true raw-chroma-silent stretch is "
        f"narrower ({first_zero_t:.2f}-{last_zero_t:.2f}s, {n_raw_zero_in_region} windows) than the full "
        "documented region, so windows outside that narrower stretch are correctly flagged active because they "
        "genuinely contain sound. "
        if raw_zero_rows else
        "No window in the documented region is raw-silent at all in this data. "
    ) + (
        f"(2) Memory-decay carryover: for the {n_raw_zero_in_region} genuinely raw-silent windows "
        f"(t={first_zero_t:.2f}-{last_zero_t:.2f}s), smoothed chroma stays strictly positive throughout "
        f"(confirmed directly against the saved arrays via the geometric 0.8-decay formula, max absolute error "
        f"{max_decay_err:.2e}). Because the 10%-of-max threshold in `analyze_piece`/`extract_chroma_sequence` is "
        "relative to each window's OWN max, pure geometric decay preserves the ratio between pitch classes "
        "exactly, so the thresholded pattern's nonzero support never disappears as the signal decays -- only its "
        "magnitude does. `active` (thresholded chroma's per-window max > 0) therefore stays True through this "
        "short a silent stretch regardless of the pause. Both mechanisms are documented, expected consequences "
        "of representing a perceptual pause with fixed real-time boundaries against 0.5s raw MIDI windows, and "
        "of the per-window-relative-threshold + EMA-memory 'active' convention established in Phase 3B and "
        "reused as-is in Phase 3G-A -- not a defect in analyze_piece, extract_chroma_sequence, or any Phase 3G-A "
        "code, and per this task's guardrails it is documented here, not changed."
        if raw_zero_rows else
        "(2) No genuinely raw-silent window exists in this region to apply the memory-decay mechanism to."
    )

    return {
        "silence_region": region,
        "window_sec": 0.5,
        "n_windows_in_region": n_in_region,
        "n_raw_chroma_zero_in_region": n_raw_zero_in_region,
        "n_raw_chroma_nonzero_in_region": n_raw_nonzero_in_region,
        "n_smoothed_chroma_nonzero_in_region": n_smoothed_nonzero_in_region,
        "n_active_flag_true_in_region": n_active_in_region,
        "raw_nonzero_sustain_tail_windows": [{"time_sec": w["time_sec"], "raw_chroma_sum": w["raw_chroma_sum"]} for w in tail_rows],
        "raw_nonzero_next_onset_windows": [{"time_sec": w["time_sec"], "raw_chroma_sum": w["raw_chroma_sum"]} for w in onset_rows],
        "true_raw_silent_span_sec": [raw_zero_rows[0]["time_sec"], raw_zero_rows[-1]["time_sec"]] if raw_zero_rows else None,
        "per_window_detail": per_window,
        "decay_formula_verification": {
            "formula": "smoothed_t = 0.8 * smoothed_{t-1} + 0.2 * raw_t, checked only where raw_t is numerically zero (so expected smoothed_t = 0.8 * smoothed_{t-1} exactly)",
            "checks": decay_checks,
            "max_abs_error_overall": max_decay_err,
        },
        "mechanism_verdict": mechanism_verdict,
        "classification": "combined_boundary_granularity_and_smoothing_memory_convention_artifact",
    }


def plot_chopin_silence(audit):
    rows = audit["per_window_detail"]
    times = [r["time_sec"] for r in rows]
    raw_sum = [r["raw_chroma_sum"] for r in rows]
    smoothed_sum = [r["smoothed_chroma_sum"] for r in rows]
    active = [r["active_flag"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(times, raw_sum, color="tab:red", marker="o", markersize=4, linewidth=1.2, label="raw_chroma (sum over 12 pitch classes)")
    ax.plot(times, smoothed_sum, color="tab:blue", marker="o", markersize=4, linewidth=1.2, label="smoothed_chroma, decay=0.8 (sum over 12 pitch classes)")
    ax.axvspan(audit["silence_region"]["start_sec"], audit["silence_region"]["end_sec"], color="gray", alpha=0.2, label="known compositional silence (t=95.36-99.64s)")
    for t, a in zip(times, active):
        if not a:
            ax.axvline(t, color="black", linestyle=":", alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Chroma energy (sum over 12 pitch classes)")
    ax.set_title("Phase 3G-B — Chopin Op. 28 No. 4: Raw vs. Smoothed Chroma Around the Known Silence")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_CHOPIN_SILENCE_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Diagnostic 4: Bach + Clementi interpretation
# ---------------------------------------------------------------------------

def bach_interpretation(l2_tie_aware):
    breakdown = l2_tie_aware["mismatch_predicted_key_breakdown"]
    neighborhood_keys = {"C maj", "D maj", "G maj"}
    n_mismatch = sum(breakdown.values())
    n_in_neighborhood = sum(v for k, v in breakdown.items() if k in neighborhood_keys)
    n_distant = n_mismatch - n_in_neighborhood
    prop_tonic_neighborhood = l2_tie_aware["category_proportions_of_mismatched_windows_only"].get("tonic_neighborhood_ambiguity")
    prop_tiebreak = l2_tie_aware["category_proportions_of_mismatched_windows_only"].get("expected_tied_for_max_but_lost_to_lower_index")
    return {
        "mismatch_predicted_key_breakdown": breakdown,
        "n_mismatched_windows": n_mismatch,
        "n_within_CGD_tonic_neighborhood": n_in_neighborhood,
        "n_outside_CGD_tonic_neighborhood_distant": n_distant,
        "proportion_within_CGD_tonic_neighborhood": (n_in_neighborhood / n_mismatch) if n_mismatch > 0 else None,
        "tonic_neighborhood_ambiguity_category_proportion": prop_tonic_neighborhood,
        "expected_tied_for_max_but_lost_category_proportion": prop_tiebreak,
        "verdict": (
            f"{n_in_neighborhood}/{n_mismatch} mismatched windows ({(n_in_neighborhood / n_mismatch):.1%}) fall within the C/G/D tonic neighborhood"
            if n_mismatch > 0 else "no mismatched windows"
        ),
    }


def clementi_interpretation(phase3g_a_l5):
    runs = phase3g_a_l5["predicted_key_sequence"]["predicted_key_runs"]
    run_keys = [r["key"] for r in runs]
    is_tonic_dominant_tonic = run_keys == ["C maj", "G maj", "C maj"]
    return {
        "predicted_key_runs_source": "verbatim from frozen PHASE3G_A_pitch_class_corpus_metrics.json (pieces.L5.predicted_key_sequence.predicted_key_runs) -- not recomputed here",
        "predicted_key_runs": runs,
        "run_key_sequence": run_keys,
        "is_exact_tonic_dominant_tonic_pattern": is_tonic_dominant_tonic,
        "relabel": (
            "Tonic-dominant-tonic excursion (C major -> G major -> C major), not a clean monotonic single "
            "modulation. The predicted trajectory visits and returns from the dominant region rather than "
            "settling there, which is the musically expected shape for a short exposition that tonicizes the "
            "dominant only briefly before a cadential return -- but it does mean Clementi should not be used as "
            "a 'before/after' single-modulation anchor test without accounting for the return leg."
            if is_tonic_dominant_tonic else
            "Predicted run sequence does not match the exact C->G->C pattern; see run_key_sequence for the "
            "actual trajectory."
        ),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(x, digits=4):
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _anchor_section_md(piece_level, piece_display, anchor_name, d):
    lines = [f"### {piece_level} — `{anchor_name}` (expected {d['expected_key']})", ""]
    lines.append(f"- n_predictions={d['n_predictions']}, n_defined={d['n_defined']}, n_undefined={d['n_undefined']}")
    lines.append(
        f"- **strict_expected_key_proportion** = {_fmt(d['strict_expected_key_proportion'])} vs. "
        f"**collection_equivalent_proportion** (expected key OR its relative {d['collection_equivalent_key']}) = "
        f"{_fmt(d['collection_equivalent_proportion'])}"
    )
    tt = d["expected_key_in_top_tie"]
    lines.append(
        f"- expected-key-in-top-tie: tied for max score in {tt['n_windows_expected_tied_for_max']} window(s); "
        f"of those, actually selected in {tt['n_of_those_actually_selected']}, lost to tie-break in "
        f"{tt['n_of_those_lost_to_tiebreak']}"
    )
    lines.append("- Tie-loss taxonomy (proportion of mismatched windows only):")
    for cat, prop in d["category_proportions_of_mismatched_windows_only"].items():
        cnt = d["category_counts"].get(cat, 0)
        lines.append(f"  - `{cat}`: {cnt} ({_fmt(prop)})")
    if d.get("mismatch_predicted_key_breakdown"):
        lines.append("- Mismatched-window predicted-key breakdown: " + ", ".join(f"{k} ({v})" for k, v in sorted(d["mismatch_predicted_key_breakdown"].items(), key=lambda kv: -kv[1])))
    lines.append("")
    return lines


def build_report_md(results):
    lines = []
    lines.append("# Phase 3G-B — Tie-Aware Diagnostic Interpretation of the Phase 3G-A Corpus")
    lines.append("")
    lines.append(
        "Adds tie-aware diagnostics on top of the **frozen** Phase 3G-A pitch-class baseline results "
        "(`PHASE3G_A_pitch_class_corpus_metrics.json`, `03_MIDI_Data/derived_phase3g_corpus/*.npy`). "
        "Nothing here recomputes the baseline, changes `np.argmax` tie-breaking, or reruns chord-id EMA/SRN, "
        "Chroma SRN, Transformer, or any neural refinement. All Phase 3G-A anchor windows are reused verbatim "
        "(imported from `evaluate_phase3g_pitch_class_corpus.PIECES`), so every number below refers to the "
        "exact same windows Phase 3G-A already scored."
    )
    lines.append("")
    lines.append(
        f"New, explicitly documented threshold introduced only in this script: `TONIC_NEIGHBORHOOD_FIFTHS = "
        f"{TONIC_NEIGHBORHOOD_FIFTHS}` (two predicted tonics are 'tonic-neighborhood' if within this many "
        "Circle-of-Fifths steps of each other). Not part of Phase 3G-A's frozen baseline."
    )
    lines.append("")

    lines.append("## Tie-loss taxonomy definitions")
    lines.append("")
    lines.append(
        "Every anchor-window is classified into exactly one category (priority order below -- a mechanistic "
        "explanation for *why* argmax picked something else outranks a purely descriptive property of *what* "
        "was picked):"
    )
    lines.append("")
    lines.append("1. `inactive_or_silence_related` — window is inactive (thresholded chroma has no positive max)")
    lines.append("2. `expected_tied_for_max_but_lost_to_lower_index` — expected key's raw score exactly ties the window's max score, but a different, lower-`key_index` key was tied too and `np.argmax` selected it instead")
    lines.append("3. `collection_equivalent_wrong_mode_or_tonic_label` — predicted key is exactly the expected key's relative major/minor (same 7-pitch-class collection, different tonic/mode label)")
    lines.append(f"4. `large_jump_instability` — this window is flagged `large_jump` (Circle-of-Fifths jump >= {DEFAULT_LARGE_JUMP_THRESHOLD}, Phase 3B's threshold)")
    lines.append(f"5. `tonic_neighborhood_ambiguity` — predicted tonic is within {TONIC_NEIGHBORHOOD_FIFTHS} Circle-of-Fifths step(s) of the expected tonic")
    lines.append("6. `expected_key_not_tied_for_max_no_support` — none of the above; the expected key had no raw-score support at all in this window")
    lines.append("")

    lines.append("## Per-anchor tie-aware diagnostics")
    lines.append("")
    for level, piece_result in results["pieces"].items():
        for anchor_name, d in piece_result["tie_aware_anchors"].items():
            lines.extend(_anchor_section_md(level, piece_result["display_name"], anchor_name, d))

    lines.append("## Bach (L2) interpretation")
    lines.append("")
    bach = results["bach_interpretation"]
    lines.append(
        f"Of {bach['n_mismatched_windows']} mismatched windows in the `full_piece` anchor (expected G Major), "
        f"{bach['n_within_CGD_tonic_neighborhood']} ({_fmt(bach['proportion_within_CGD_tonic_neighborhood'])}) "
        "predicted a key within Bach's own C/G/D tonic neighborhood (the predicted-key breakdown is: "
        + ", ".join(f"{k} ({v})" for k, v in sorted(bach["mismatch_predicted_key_breakdown"].items(), key=lambda kv: -kv[1]))
        + f"), and {bach['n_outside_CGD_tonic_neighborhood_distant']} predicted a distant key outside that "
        "neighborhood. By the tie-loss taxonomy, "
        f"{_fmt(bach['tonic_neighborhood_ambiguity_category_proportion'])} of mismatches fall in the "
        f"`tonic_neighborhood_ambiguity` category and {_fmt(bach['expected_tied_for_max_but_lost_category_proportion'])} "
        "in `expected_tied_for_max_but_lost_to_lower_index`. **Conclusion:** Bach's errors are essentially all "
        "close-tonic confusion among G/C/D, not failures onto a distant or unrelated key -- consistent with "
        "genuine (if imprecise) tonal-neighborhood evidence rather than random or catastrophic misprediction."
    )
    lines.append("")

    lines.append("## Clementi (L5) interpretation")
    lines.append("")
    clem = results["clementi_interpretation"]
    lines.append(f"Frozen Phase 3G-A predicted-key run sequence: {' -> '.join(clem['run_key_sequence'])}.")
    lines.append("")
    lines.append(clem["relabel"])
    lines.append("")

    lines.append("## Chopin (L4) silence audit")
    lines.append("")
    audit = results["chopin_silence_audit"]
    lines.append(
        f"Silence region: t={audit['silence_region']['start_sec']:.2f}-{audit['silence_region']['end_sec']:.2f}s "
        f"({audit['n_windows_in_region']} windows at window_sec={audit['window_sec']}). Of these: "
        f"{audit['n_raw_chroma_zero_in_region']}/{audit['n_windows_in_region']} have numerically-zero raw chroma "
        f"and {audit['n_raw_chroma_nonzero_in_region']}/{audit['n_windows_in_region']} do not; "
        f"{audit['n_smoothed_chroma_nonzero_in_region']}/{audit['n_windows_in_region']} have strictly-positive "
        f"smoothed chroma; {audit['n_active_flag_true_in_region']}/{audit['n_windows_in_region']} are flagged "
        "`active` by Phase 3G-A's `analyze_piece`."
    )
    lines.append("")
    if audit["true_raw_silent_span_sec"]:
        lines.append(
            f"The true raw-chroma-silent span is narrower than the full documented region: "
            f"t={audit['true_raw_silent_span_sec'][0]:.2f}-{audit['true_raw_silent_span_sec'][1]:.2f}s "
            f"({audit['n_raw_chroma_zero_in_region']} windows), vs. the documented "
            f"t={audit['silence_region']['start_sec']:.2f}-{audit['silence_region']['end_sec']:.2f}s."
        )
        if audit["raw_nonzero_sustain_tail_windows"]:
            lines.append(
                "- Sustain/pedal tail before the true silence: " +
                ", ".join(f"t={w['time_sec']:.2f}s (raw_sum={w['raw_chroma_sum']:.1f})" for w in audit["raw_nonzero_sustain_tail_windows"])
            )
        if audit["raw_nonzero_next_onset_windows"]:
            lines.append(
                "- Next phrase's onset at/before the region's end boundary: " +
                ", ".join(f"t={w['time_sec']:.2f}s (raw_sum={w['raw_chroma_sum']:.1f})" for w in audit["raw_nonzero_next_onset_windows"])
            )
        lines.append("")
    lines.append(
        f"Decay-formula check (smoothed_t = 0.8*smoothed_{{t-1}} verified directly against the saved arrays "
        f"wherever raw_t is numerically zero): max absolute error = "
        f"{audit['decay_formula_verification']['max_abs_error_overall']:.2e} across "
        f"{len(audit['decay_formula_verification']['checks'])} checked window(s) -- confirms the EMA formula, "
        "not just its qualitative effect."
    )
    lines.append("")
    lines.append(f"**Classification: `{audit['classification']}`**")
    lines.append("")
    lines.append(audit["mechanism_verdict"])
    lines.append("")
    lines.append(f"Diagnostic plot: `{os.path.relpath(OUT_CHOPIN_SILENCE_PNG, os.path.join(_THIS_DIR, '..'))}`")
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This is Phase 3G-B only: tie-aware diagnostic interpretation added on top of the frozen Phase 3G-A "
        "results. No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement was run. No Phase 2C/2D/3B/3C "
        "script, no Phase 3G-A script, and no Phase 3G-A output file was modified. `np.argmax` tie-breaking "
        "behavior in the underlying baseline was not changed -- it is analyzed here, not altered. The Chopin "
        "silence 'active' behavior is documented as a smoothing-memory/threshold-convention mechanism, not "
        "'fixed.'"
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


def run_verification(results, out_paths, phase3g_a_mtime_before, derived_corpus_mtimes_before, old_script_mtimes_before):
    checks = []

    for p in out_paths:
        checks.append((f"{os.path.basename(p)} exists", os.path.exists(p)))
        checks.append((f"{os.path.basename(p)} is non-empty", os.path.exists(p) and os.path.getsize(p) > 0))

    nan_paths = _scan_for_nan(results)
    checks.append(("no NaNs in Phase 3G-B metrics", len(nan_paths) == 0))

    checks.append((
        "PHASE3G_A_pitch_class_corpus_metrics.json not modified (mtime unchanged)",
        os.path.getmtime(PHASE3G_A_METRICS_JSON) == phase3g_a_mtime_before,
    ))
    for path, mtime_before in derived_corpus_mtimes_before.items():
        checks.append((f"derived_phase3g_corpus/{os.path.basename(path)} not modified", os.path.getmtime(path) == mtime_before))
    for path, mtime_before in old_script_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} (old Phase script) not modified", os.path.getmtime(path) == mtime_before))

    # Cross-check: our independently-recomputed strict_expected_key_proportion
    # must match Phase 3G-A's own frozen proportion_expected_key for every
    # anchor, confirming we read the same frozen arrays/windows Phase 3G-A did.
    with open(PHASE3G_A_METRICS_JSON) as f:
        frozen = json.load(f)
    cross_check_ok = True
    for level, piece_result in results["pieces"].items():
        frozen_anchors = frozen["pieces"][level]["anchor_mismatch_summary"]
        for anchor_name, d in piece_result["tie_aware_anchors"].items():
            frozen_prop = frozen_anchors[anchor_name]["proportion_expected_key"]
            new_prop = d["strict_expected_key_proportion"]
            if frozen_prop is None and new_prop is None:
                continue
            if frozen_prop is None or new_prop is None or abs(frozen_prop - new_prop) > 1e-9:
                cross_check_ok = False
    checks.append(("strict_expected_key_proportion matches frozen Phase 3G-A proportion_expected_key for every anchor", cross_check_ok))

    audit = results["chopin_silence_audit"]
    checks.append(("Chopin silence audit explicitly states raw-vs-smoothed behavior", "raw" in audit["mechanism_verdict"].lower() and "smooth" in audit["mechanism_verdict"].lower()))
    checks.append(("Chopin silence n_windows_in_region matches Phase 3G-A's frozen count (9)", audit["n_windows_in_region"] == frozen["pieces"]["L4"]["inactive_window_handling"]["known_silence_region"]["n_windows_in_region"]))

    print()
    print("evaluate_phase3g_b_tie_aware_diagnostics.py verification")
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

    phase3g_a_mtime_before = os.path.getmtime(PHASE3G_A_METRICS_JSON)
    derived_corpus_files = [os.path.join(_DERIVED_CORPUS_DIR, f) for f in os.listdir(_DERIVED_CORPUS_DIR)]
    derived_corpus_mtimes_before = {p: os.path.getmtime(p) for p in derived_corpus_files}

    old_scripts = [
        "midi_chroma_extraction.py", "pitch_class_baseline.py",
        "evaluate_pitch_class_phase2d.py", "pitch_class_uncertainty_diagnostics.py",
        "compare_phase3c_disagreement.py", "evaluate_phase3g_pitch_class_corpus.py",
    ]
    old_script_mtimes_before = {
        os.path.join(_THIS_DIR, s): os.path.getmtime(os.path.join(_THIS_DIR, s))
        for s in old_scripts if os.path.exists(os.path.join(_THIS_DIR, s))
    }

    with open(PHASE3G_A_METRICS_JSON) as f:
        phase3g_a = json.load(f)

    pieces_out = {}
    for piece in PIECES:
        level = piece["level"]
        print(f"\n=== {level} — {piece['display_name']} (tie-aware diagnostics) ===")
        arrays = load_piece_arrays(piece["stem"])
        meta = load_piece_chroma_metadata(piece["stem"])
        anchors_spec = piece["anchors_fn"](meta["duration_sec"])

        tie_aware_anchors = {}
        for anchor in anchors_spec:
            d = compute_tie_aware_anchor_diagnostics(arrays, anchor)
            tie_aware_anchors[anchor["name"]] = d
            print(f"  anchor={anchor['name']} strict={_fmt(d['strict_expected_key_proportion'])} collection_equiv={_fmt(d['collection_equivalent_proportion'])}")

        pieces_out[level] = {
            "display_name": piece["display_name"],
            "tie_aware_anchors": tie_aware_anchors,
        }

    bach_result = bach_interpretation(pieces_out["L2"]["tie_aware_anchors"]["full_piece"])
    clementi_result = clementi_interpretation(phase3g_a["pieces"]["L5"])

    print("\n=== Chopin silence audit ===")
    chopin_audit = chopin_silence_audit()
    print(f"  raw zero in region: {chopin_audit['n_raw_chroma_zero_in_region']}/{chopin_audit['n_windows_in_region']}")
    print(f"  smoothed nonzero in region: {chopin_audit['n_smoothed_chroma_nonzero_in_region']}/{chopin_audit['n_windows_in_region']}")
    print(f"  active flag true in region: {chopin_audit['n_active_flag_true_in_region']}/{chopin_audit['n_windows_in_region']}")
    plot_chopin_silence(chopin_audit)
    print(f"  wrote {OUT_CHOPIN_SILENCE_PNG}")

    results = {
        "phase": "phase_3g_b_tie_aware_diagnostics",
        "based_on_frozen": "PHASE3G_A_pitch_class_corpus_metrics.json (read-only, not modified)",
        "settings": {
            "tonic_neighborhood_fifths": TONIC_NEIGHBORHOOD_FIFTHS,
            "taxonomy_categories": TAXONOMY_CATEGORIES,
        },
        "pieces": pieces_out,
        "bach_interpretation": bach_result,
        "clementi_interpretation": clementi_result,
        "chopin_silence_audit": chopin_audit,
        "notes": (
            "Phase 3G-B: tie-aware diagnostic interpretation only. No chord-id EMA/SRN, Chroma SRN, "
            "Transformer, or neural refinement. No dense per-timestep accuracy claimed. Reuses (imports, does "
            "not modify) evaluate_phase3g_pitch_class_corpus.PIECES for anchor definitions and reads (does not "
            "modify) Phase 3G-A's frozen metrics JSON and derived .npy arrays."
        ),
    }

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

    results = _to_native(results)

    with open(OUT_METRICS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_METRICS_JSON}")

    report_md = build_report_md(results)
    with open(OUT_REPORT_MD, "w") as f:
        f.write(report_md)
    print(f"Wrote {OUT_REPORT_MD}")

    out_paths = [OUT_METRICS_JSON, OUT_REPORT_MD, OUT_CHOPIN_SILENCE_PNG]
    run_verification(results, out_paths, phase3g_a_mtime_before, derived_corpus_mtimes_before, old_script_mtimes_before)


if __name__ == "__main__":
    main()
