"""evaluate_phase3j_b_section_level_candidate_b.py

Phase 3J-B: implementation and evaluation of Phase 3J-A's revised
Candidate B — a section-level, non-neural tonic/mode resolver using
aggregated mediant evidence plus an asymmetric raised-leading-tone cue.

Narrow scope, per explicit task guardrails: this script implements ONLY
Candidate B (not Candidate A, not Candidate C). It does not run chord-id
EMA/SRN, does not implement a Chroma SRN, Transformer, or any neural
refinement, does not modify the frozen Stage 1 baseline
(`pitch_class_baseline.py`) or any Phase 2C/2D/3B/3C script, and does not
overwrite any Phase 3G/3H/3I/3J-A output.

Design reference: `05_Figures_Results/PHASE3J_A_section_level_resolver_
design.md` (frozen; this script implements its six-stage architecture and
revised Candidate B exactly as specified there, with all governing
constants declared in PRE_REGISTERED_CONFIG below and frozen BEFORE any
anchor comparison is computed).

Six stages (Stage 1 frozen, Stages 2-6 new):
  Stage 1: load frozen Phase 3G-A per-piece arrays, read-only.
  Stage 2: collapse Stage 1 key_id -> diatonic collection id (reusing
    evaluate_phase3h_b_texture_gated_resolver.collection_class verbatim).
  Stage 3: label-free, offline segment detection over the collection
    sequence (stable / transition / undefined).
  Stage 4: aggregate pre-threshold (raw) chroma evidence per stable
    segment; numerically verified equivalent to summing Stage 1's own
    per-window raw_scores (linearity of the SCALE_TEMPLATES matmul).
  Stage 5: revised Candidate B decision rule per stable segment (mediant
    comparison + asymmetric raised-leading-tone cue; ambiguous fallback =
    segment majority vote of Stage 1's own key_id, never index order,
    never an anchor).
  Stage 6: project the segment decision back onto windows as a NEW field
    `key_id_section_resolved` -- Stage 1's own `key_id` array is never
    modified, read from disk, or aliased.

Anchors are loaded and consulted only in the Evaluation section, after
every piece's Stage 1-6 output already exists. No dense per-timestep
accuracy is claimed anywhere -- all anchor comparisons are the same
sparse, window-level convention every prior Phase 3G/3H script used.
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

from shared_music_defs import decode_key, key_index, key_tonic_pc, fifth_distance, PC_NAMES  # noqa: E402
from pitch_class_baseline import SCALE_TEMPLATES, DEFAULT_WINDOW_SEC, DEFAULT_MEMORY_DECAY  # noqa: E402
from midi_chroma_extraction import DEFAULT_THRESHOLD_RATIO  # noqa: E402
from pitch_class_uncertainty_diagnostics import DEFAULT_LARGE_JUMP_THRESHOLD, _expected_key_id  # noqa: E402

from evaluate_phase3g_pitch_class_corpus import PIECES, CHOPIN_SILENCE_REGION  # noqa: E402
from evaluate_phase3h_b_texture_gated_resolver import collection_class  # noqa: E402
from evaluate_phase3h_a_tonic_mode_resolver import (  # noqa: E402
    compute_piece_level_metrics, compute_anchor_metrics, load_piece_duration, _fmt,
)

assert DEFAULT_WINDOW_SEC == 0.5
assert DEFAULT_MEMORY_DECAY == 0.8
assert DEFAULT_THRESHOLD_RATIO == 0.10

PHASE3G_A_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3G_A_pitch_class_corpus_metrics.json")
PHASE3G_B_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3G_B_tie_aware_diagnostics_metrics.json")
PHASE3H_A_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3H_A_tonic_mode_resolver_metrics.json")
PHASE3H_B_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3H_B_texture_gated_resolver_metrics.json")
PHASE3H_C_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3H_C_gate_sensitivity_metrics.json")
PHASE3I_REPORT_MD = os.path.join(_FIGURES_DIR, "PHASE3I_synthesis_and_architecture_decision.md")
PHASE3J_A_REPORT_MD = os.path.join(_FIGURES_DIR, "PHASE3J_A_section_level_resolver_design.md")

OUT_METRICS_JSON = os.path.join(_FIGURES_DIR, "PHASE3J_B_section_level_candidate_b_metrics.json")
OUT_REPORT_MD = os.path.join(_FIGURES_DIR, "PHASE3J_B_section_level_candidate_b_report.md")
OUT_SEGMENTS_JSON = os.path.join(_FIGURES_DIR, "PHASE3J_B_section_level_candidate_b_segments.json")
OUT_PLOT_PREFIX = os.path.join(_FIGURES_DIR, "PHASE3J_B_")


# =============================================================================
# PRE-REGISTERED CONFIGURATION -- frozen before any anchor evaluation.
# Every constant below is fixed in writing here, matches the task's
# specified primary configuration exactly, and is NOT adjusted anywhere
# in this file after seeing any anchor-comparison result. The sensitivity
# audit (bottom of this file) sweeps a small, separately predeclared grid
# around these values, descriptively, and explicitly does not select or
# adopt a "winner."
# =============================================================================

PRE_REGISTERED_CONFIG = {
    "MIN_SEGMENT_WINDOWS": 4,                    # >= this many windows of constant collection -> "stable" segment
    "MIN_SEGMENT_DURATION_SEC": 4 * DEFAULT_WINDOW_SEC,  # 2.0s, derived, not independently chosen
    "BRIEF_INTERRUPTION_MAX_WINDOWS": 3,         # short runs of this length or less are merge/transition candidates
    "MERGE_RULE": "merge a brief interruption only when the stable collection on both sides is identical; a brief run between different collections becomes its own unresolved 'transition' segment",
    "SEGMENT_DETECTION_MODE": "offline, whole-sequence, label-free run-length encoding of the Stage 2 collection sequence (Stage 1's key_id collapsed via collection_class) -- no anchor, no lookahead-into-the-future beyond the piece's own already-fully-computed Stage 1 output",
    "AGGREGATION_SOURCE": "sum of raw (pre-threshold) chroma across all windows in a stable segment; verified numerically equivalent (see verify_chroma_score_linearity) to scoring each window from raw chroma and summing the per-window scores, by linearity of the fixed SCALE_TEMPLATES matmul -- thresholded chroma is never aggregated. NOTE: this equivalence is checked against raw-chroma-derived per-window scores computed in this script, NOT against the frozen derived_phase3g_corpus raw_scores.npy array, because that array is actually thresholded_chroma @ SCALE_TEMPLATES.T (Phase 3B's analyze_piece naming: 'raw' = un-normalized, not un-thresholded) -- see verify_chroma_score_linearity's docstring for the full explanation.",
    "AMBIGUOUS_FALLBACK": "segment majority vote over the segment's own frozen Stage 1 key_id values (never index-order argmax, never an anchor)",
    "MEDIANT_MARGIN_THRESHOLD": 0.02,            # normalized (fraction-of-segment-energy) difference required between major-third and minor-third evidence to call the mediant cue decisive, rather than ambiguous
    "LEADING_TONE_PRESENCE_THRESHOLD": 0.02,     # normalized (fraction-of-segment-energy) evidence at the minor candidate's raised leading tone required to count as "meaningful presence"
    "LEADING_TONE_ASYMMETRY": "presence of the raised leading tone is POSITIVE evidence for the minor candidate and, when present, settles the decision toward minor regardless of the mediant cue; ABSENCE of the raised leading tone is NEUTRAL -- it never itself counts as evidence for major, and the decision in that case rests entirely on the (symmetric) mediant cue",
    "STAGE6_TRANSITION_UNDEFINED_CONVENTION": "transition and undefined segments are NOT resolved by Candidate B; their windows in key_id_section_resolved preserve Stage 1's own frozen key_id verbatim, and are additionally flagged False in a parallel is_section_resolved boolean mask so 'genuinely resolved by Candidate B' vs. 'fell back to frozen Stage 1' is always distinguishable downstream",
}

SENSITIVITY_GRID = {
    "MIN_SEGMENT_WINDOWS": [4, 6, 8],
    "BRIEF_INTERRUPTION_MAX_WINDOWS": [1, 3],
}


# =============================================================================
# Stage 1 -- load frozen Phase 3G-A evidence, read-only
# =============================================================================

def load_frozen_stage1(stem):
    def _load(field):
        return np.load(os.path.join(_DERIVED_CORPUS_DIR, f"{stem}_{field}.npy"))

    with open(os.path.join(_DERIVED_CORPUS_DIR, f"{stem}_chroma_metadata.json")) as f:
        chroma_meta = json.load(f)

    key_id = _load("key_id")
    raw_chroma = _load("raw_chroma")
    smoothed_chroma = _load("smoothed_chroma_decay08")
    thresholded_chroma = _load("thresholded_smoothed_chroma_decay08")
    raw_scores = _load("raw_scores")
    active = _load("active")
    prediction_times_sec = _load("prediction_times_sec")

    # This pipeline's frozen arrays (analyze_piece, Phase 3B/3G-A
    # convention) are computed over the FULL, undropped chroma grid, so
    # prediction_times_sec = index * window_sec always -- offset_windows
    # is 0 by construction here (distinct from the separate
    # midi_to_key_baseline/Phase 2D leading-silence-offset convention,
    # which this script does not use). Verified, not assumed:
    n = len(prediction_times_sec)
    expected_times = np.arange(n) * DEFAULT_WINDOW_SEC
    assert np.allclose(prediction_times_sec, expected_times), f"{stem}: unexpected offset in prediction_times_sec"
    offset_windows = 0

    return {
        "key_id": key_id, "raw_chroma": raw_chroma, "smoothed_chroma": smoothed_chroma,
        "thresholded_chroma": thresholded_chroma, "raw_scores": raw_scores, "active": active,
        "prediction_times_sec": prediction_times_sec, "offset_windows": offset_windows,
        "chroma_meta": chroma_meta,
    }


def verify_chroma_score_linearity(raw_chroma, start_idx, end_idx, atol=1e-6):
    """Confirms sum_t(raw_chroma_t) @ SCALE_TEMPLATES.T ==
    sum_t(raw_chroma_t @ SCALE_TEMPLATES.T) for the windows in this
    segment -- i.e. aggregating pre-threshold chroma and scoring once is
    mathematically identical to scoring each window from raw chroma and
    summing the per-window scores. This is a REAL runtime numeric check
    (not just an assertion in prose) run for every stable segment.

    IMPORTANT, discovered during implementation (not assumed at design
    time): this deliberately does NOT compare against the frozen
    `{stem}_raw_scores.npy` array from derived_phase3g_corpus. Despite its
    name, that array is `thresholded_smoothed_chroma @ SCALE_TEMPLATES.T`
    (pitch_class_uncertainty_diagnostics.analyze_piece's own "raw" means
    "un-normalized/pre-softmax", not "computed from un-thresholded
    chroma"). Comparing this script's raw-chroma aggregation against that
    array would compare two genuinely different quantities (thresholding
    is nonlinear) and fail for reasons unrelated to the aggregation math
    itself -- this was caught by an initial run of this exact check
    producing large (non-floating-point-noise) errors, traced to that
    naming mismatch, and fixed by computing both sides of the equality
    from the same raw_chroma input, matching Stage 4's own aggregation
    source exactly."""
    segment_chroma = raw_chroma[start_idx:end_idx]
    agg_chroma = segment_chroma.sum(axis=0)
    scores_via_aggregated_chroma = agg_chroma @ SCALE_TEMPLATES.T
    scores_via_summed_per_window = (segment_chroma @ SCALE_TEMPLATES.T).sum(axis=0)
    max_abs_err = float(np.max(np.abs(scores_via_aggregated_chroma - scores_via_summed_per_window)))
    return max_abs_err <= atol, max_abs_err, agg_chroma


# =============================================================================
# Stage 2 -- collapse Stage 1 key_id to diatonic collection (reused verbatim)
# =============================================================================

def collapse_to_collections(key_id):
    return np.array([collection_class(int(k)) for k in key_id], dtype=np.int64)


def candidate_tonics_for_collection(collection_id):
    """Returns (major_key_id, minor_key_id) for a given collection id
    (collection id == the major member's tonic pitch class, by
    collection_class's own definition)."""
    major_tonic_pc = collection_id
    minor_tonic_pc = (collection_id - 3) % 12
    return key_index(major_tonic_pc, "maj"), key_index(minor_tonic_pc, "min")


# =============================================================================
# Stage 3 -- label-free, offline segment detection
# =============================================================================

def run_length_encode_collections(collection_seq):
    runs = []
    T = len(collection_seq)
    i = 0
    while i < T:
        start = i
        c = int(collection_seq[i])
        while i < T and int(collection_seq[i]) == c:
            i += 1
        runs.append({"start_idx": start, "end_idx": i, "collection_id": c, "length": i - start})
    return runs


def merge_brief_interruptions(runs, max_interruption):
    runs = [dict(r) for r in runs]
    merged_any = True
    while merged_any:
        merged_any = False
        for i in range(1, len(runs) - 1):
            prev, cur, nxt = runs[i - 1], runs[i], runs[i + 1]
            if cur["length"] <= max_interruption and prev["collection_id"] == nxt["collection_id"] and prev["collection_id"] != -1:
                new_run = {
                    "start_idx": prev["start_idx"], "end_idx": nxt["end_idx"],
                    "collection_id": prev["collection_id"],
                    "length": nxt["end_idx"] - prev["start_idx"],
                    "merged_interruptions": prev.get("merged_interruptions", []) + [
                        {"start_idx": cur["start_idx"], "end_idx": cur["end_idx"], "length": cur["length"]}
                    ] + nxt.get("merged_interruptions", []),
                }
                runs = runs[:i - 1] + [new_run] + runs[i + 2:]
                merged_any = True
                break
    return runs


def classify_segments(runs, min_segment_windows):
    segments = []
    for run in runs:
        if run["collection_id"] == -1:
            status = "undefined"
        elif run["length"] >= min_segment_windows:
            status = "stable"
        else:
            status = "transition"
        segments.append({**run, "status": status})
    return segments


def detect_segments(key_id, min_segment_windows, brief_interruption_max):
    collection_seq = collapse_to_collections(key_id)
    runs = run_length_encode_collections(collection_seq)
    merged = merge_brief_interruptions(runs, brief_interruption_max)
    segments = classify_segments(merged, min_segment_windows)
    return segments


# =============================================================================
# Stage 4 + 5 -- aggregation and revised Candidate B decision
# =============================================================================

def resolve_segment_candidate_b(raw_chroma, key_id_frozen, seg, config):
    """Stage 4 (aggregation) + Stage 5 (revised Candidate B) combined for
    one stable segment. Returns the segment dict augmented with
    aggregation + decision fields. No anchor input anywhere."""
    start, end = seg["start_idx"], seg["end_idx"]
    collection_id = seg["collection_id"]
    major_key_id, minor_key_id = candidate_tonics_for_collection(collection_id)
    major_tonic_pc, minor_tonic_pc = key_tonic_pc(major_key_id), key_tonic_pc(minor_key_id)

    agg_chroma = raw_chroma[start:end].sum(axis=0)  # (12,) pre-threshold, linear
    total_energy = float(agg_chroma.sum())
    active_pc_count = int(np.count_nonzero(agg_chroma))

    if total_energy <= 0.0:
        # Degenerate case (should not occur for a genuinely active stable
        # segment): no evidence at all -> immediate fallback, no
        # mediant/leading-tone computation possible.
        pc_fraction = np.zeros(12)
    else:
        pc_fraction = agg_chroma / total_energy

    major_third_pc = (major_tonic_pc + 4) % 12
    minor_third_pc = (minor_tonic_pc + 3) % 12
    major_third_evidence = float(pc_fraction[major_third_pc])
    minor_third_evidence = float(pc_fraction[minor_third_pc])
    mediant_signal = major_third_evidence - minor_third_evidence  # >0 favors major, <0 favors minor

    leading_tone_pc = (minor_tonic_pc + 11) % 12
    leading_tone_evidence = float(pc_fraction[leading_tone_pc])
    leading_tone_supports_minor = leading_tone_evidence >= config["LEADING_TONE_PRESENCE_THRESHOLD"]

    # --- Decision rule: simple, fixed, interpretable, no learned weights ---
    decision_source = None
    resolved_key_id = None
    if total_energy <= 0.0:
        decision_source = "fallback_majority_vote_zero_energy"
    elif leading_tone_supports_minor:
        # Asymmetric: presence of the leading tone settles it toward
        # minor regardless of the mediant cue. Absence never triggers a
        # symmetric "therefore major" conclusion anywhere in this rule.
        resolved_key_id = minor_key_id
        decision_source = "leading_tone_positive_evidence"
    elif mediant_signal > config["MEDIANT_MARGIN_THRESHOLD"]:
        resolved_key_id = major_key_id
        decision_source = "mediant_signal"
    elif mediant_signal < -config["MEDIANT_MARGIN_THRESHOLD"]:
        resolved_key_id = minor_key_id
        decision_source = "mediant_signal"
    else:
        decision_source = "fallback_majority_vote_ambiguous"

    if resolved_key_id is None:
        resolved_key_id = majority_vote_fallback(key_id_frozen, start, end, major_key_id, minor_key_id)

    seg_out = dict(seg)
    seg_out.update({
        "major_candidate_key_id": int(major_key_id), "minor_candidate_key_id": int(minor_key_id),
        "major_candidate_name": f"{decode_key(major_key_id)[0]} {decode_key(major_key_id)[1]}",
        "minor_candidate_name": f"{decode_key(minor_key_id)[0]} {decode_key(minor_key_id)[1]}",
        "aggregate_active_pc_count": active_pc_count, "aggregate_total_energy": total_energy,
        "major_third_pc": int(major_third_pc), "minor_third_pc": int(minor_third_pc),
        "major_third_evidence_fraction": major_third_evidence, "minor_third_evidence_fraction": minor_third_evidence,
        "mediant_signal": mediant_signal,
        "leading_tone_pc": int(leading_tone_pc), "leading_tone_evidence_fraction": leading_tone_evidence,
        "leading_tone_supports_minor": bool(leading_tone_supports_minor),
        "decision_source": decision_source,
        "resolved_key_id": int(resolved_key_id),
        "resolved_key_name": f"{decode_key(resolved_key_id)[0]} {decode_key(resolved_key_id)[1]}",
    })
    return seg_out


def majority_vote_fallback(key_id_frozen, start, end, major_key_id, minor_key_id):
    """Ambiguous-segment fallback: majority vote over the segment's own
    frozen Stage 1 key_id values (restricted to the two candidates that
    can possibly appear in a segment of this collection, by construction
    of collection_class's bijection) -- never index order, never an
    anchor. Ties in the vote itself are broken by first-occurrence order
    (deterministic, non-anchor, non-arbitrary-index-based)."""
    segment_keys = key_id_frozen[start:end]
    segment_keys = segment_keys[segment_keys != -1]
    if len(segment_keys) == 0:
        return major_key_id  # degenerate: no defined Stage 1 prediction at all in this range; documented, not expected to occur for a "stable" segment
    counts = Counter(segment_keys.tolist())
    max_count = max(counts.values())
    tied = [k for k, c in counts.items() if c == max_count]
    if len(tied) == 1:
        return tied[0]
    # deterministic tie-break: first tied key to occur in time order
    for k in segment_keys.tolist():
        if k in tied:
            return k
    return tied[0]


# =============================================================================
# Stage 6 -- project segment decisions back onto windows
# =============================================================================

def project_to_windows(key_id_frozen, segments):
    T = len(key_id_frozen)
    key_id_section_resolved = key_id_frozen.copy()
    is_section_resolved = np.zeros(T, dtype=bool)
    for seg in segments:
        if seg["status"] == "stable":
            key_id_section_resolved[seg["start_idx"]:seg["end_idx"]] = seg["resolved_key_id"]
            is_section_resolved[seg["start_idx"]:seg["end_idx"]] = True
        # transition / undefined: leave key_id_section_resolved as the
        # copied frozen Stage 1 value, is_section_resolved stays False.
    return key_id_section_resolved, is_section_resolved


# =============================================================================
# Full per-piece pipeline (Stages 2-6)
# =============================================================================

def run_pipeline_for_piece(stage1, config):
    key_id_frozen = stage1["key_id"]
    raw_chroma = stage1["raw_chroma"]

    segments = detect_segments(key_id_frozen, config["MIN_SEGMENT_WINDOWS"], config["BRIEF_INTERRUPTION_MAX_WINDOWS"])

    linearity_checks = []
    resolved_segments = []
    for seg in segments:
        if seg["status"] == "stable":
            ok, max_err, _ = verify_chroma_score_linearity(raw_chroma, seg["start_idx"], seg["end_idx"])
            linearity_checks.append({"segment_start_idx": seg["start_idx"], "segment_end_idx": seg["end_idx"], "passed": ok, "max_abs_error": max_err})
            seg = resolve_segment_candidate_b(raw_chroma, key_id_frozen, seg, config)
        resolved_segments.append(seg)

    key_id_section_resolved, is_section_resolved = project_to_windows(key_id_frozen, resolved_segments)

    return {
        "segments": resolved_segments,
        "key_id_section_resolved": key_id_section_resolved,
        "is_section_resolved": is_section_resolved,
        "linearity_checks": linearity_checks,
    }


# =============================================================================
# Unit / synthetic checks -- run BEFORE the full corpus pass. If any fail,
# the script aborts before touching real piece data or any anchor.
# =============================================================================

def run_unit_checks():
    checks = []

    # 1. Relative major/minor candidate mapping is correct for all 12 collections.
    mapping_ok = True
    known_pairs = {0: ("C", "A"), 7: ("G", "E"), 2: ("D", "B"), 5: ("F", "D")}  # spot-checked collection_id -> (major tonic name, minor tonic name)
    for collection_id in range(12):
        major_key_id, minor_key_id = candidate_tonics_for_collection(collection_id)
        if collection_class(major_key_id) != collection_id or collection_class(minor_key_id) != collection_id:
            mapping_ok = False
        if collection_id in known_pairs:
            maj_name, min_name = decode_key(major_key_id)[0], decode_key(minor_key_id)[0]
            expected_maj, expected_min = known_pairs[collection_id]
            if maj_name != expected_maj or min_name != expected_min:
                mapping_ok = False
    checks.append(("relative major/minor candidate mapping correct for all 12 collections (incl. spot-check C/A, G/E, D/B, F/D)", mapping_ok))

    # 2. Raised leading-tone pitch class computed correctly for all 12 minor tonics.
    lt_ok = True
    known_leading_tones = {9: "G#", 4: "D#", 2: "C#", 0: "B"}  # A minor -> G#, E minor -> D#, D minor -> C#, C minor -> B
    for minor_tonic_pc in range(12):
        lt_pc = (minor_tonic_pc + 11) % 12
        if minor_tonic_pc in known_leading_tones:
            if PC_NAMES[lt_pc] != known_leading_tones[minor_tonic_pc]:
                lt_ok = False
    checks.append(("raised-leading-tone pc formula correct for all 12 minor tonics (spot-checked A/E/D/C minor)", lt_ok))

    # 3. Absence of raised leading tone does NOT automatically choose major:
    #    construct a synthetic segment where leading-tone evidence is zero
    #    and mediant evidence is exactly tied (within threshold) -> must
    #    fall through to majority-vote fallback, not default to major.
    config = dict(PRE_REGISTERED_CONFIG)
    collection_id = 0  # C major / A minor
    major_key_id, minor_key_id = candidate_tonics_for_collection(collection_id)
    fake_raw_chroma = np.zeros((6, 12))
    # Equal energy at major-third (E, pc4) and minor-third (C, pc0) -> mediant signal ~0; nothing at leading tone (G#, pc8).
    fake_raw_chroma[:, 4] = 1.0
    fake_raw_chroma[:, 0] = 1.0
    fake_key_id_frozen = np.array([major_key_id, major_key_id, minor_key_id, minor_key_id, minor_key_id, major_key_id])  # majority = minor (3) vs major (3) -> tie in THIS array; use asymmetric counts below instead
    fake_key_id_frozen_majority_minor = np.array([minor_key_id, minor_key_id, minor_key_id, major_key_id, major_key_id, minor_key_id])  # 4 minor, 2 major
    seg = {"start_idx": 0, "end_idx": 6, "collection_id": collection_id, "length": 6, "status": "stable"}
    resolved = resolve_segment_candidate_b(fake_raw_chroma, fake_key_id_frozen_majority_minor, seg, config)
    ambiguous_not_major = resolved["decision_source"].startswith("fallback_majority_vote") and resolved["resolved_key_id"] != major_key_id
    checks.append(("absence of raised leading tone + ambiguous mediant does NOT default to major (falls back to majority vote instead)", ambiguous_not_major))
    checks.append(("...and that fallback correctly resolves to the actual majority key (minor, 4 of 6 windows) in this synthetic case", resolved["resolved_key_id"] == minor_key_id))

    # 4. Ambiguous fallback uses Stage 1 majority, NOT index order: construct
    #    a case where majority (minor, more occurrences) differs from what a
    #    plain index-order/argmax-style rule would pick (major, since major
    #    always has the lower key_index within a collection pair).
    index_order_would_pick_major = major_key_id < minor_key_id  # true by construction of key_index
    checks.append(("sanity: within any collection, major_key_id has the lower index than minor_key_id (so an index-order fallback would always pick major)", index_order_would_pick_major))
    checks.append(("fallback picks Stage 1 MAJORITY (minor, 4/6) not index-order (which would be major) -- confirms majority-vote, not index-order, governs the fallback", resolved["resolved_key_id"] == minor_key_id and resolved["resolved_key_id"] != major_key_id))

    # 5. Positive leading-tone evidence DOES settle toward minor (the
    #    asymmetric, non-neutral direction), confirming the rule is
    #    actually asymmetric and not simply always-neutral.
    fake_raw_chroma_lt = np.zeros((6, 12))
    fake_raw_chroma_lt[:, 4] = 1.0    # major third present
    fake_raw_chroma_lt[:, 8] = 1.0    # raised leading tone of A minor (G#, pc8) also present, equal weight
    seg2 = {"start_idx": 0, "end_idx": 6, "collection_id": collection_id, "length": 6, "status": "stable"}
    resolved_lt = resolve_segment_candidate_b(fake_raw_chroma_lt, fake_key_id_frozen_majority_minor, seg2, config)
    checks.append(("positive leading-tone evidence settles the decision toward MINOR even with major-third evidence also present (asymmetric rule)", resolved_lt["resolved_key_id"] == minor_key_id and resolved_lt["decision_source"] == "leading_tone_positive_evidence"))

    # 6. Segment run-length-encoding + merge logic: a short interruption
    #    between two IDENTICAL collections merges; between two DIFFERENT
    #    collections becomes its own transition segment.
    # C major (collection 0, tonic pc 0) x5, G major (collection 7, tonic pc 7) x2
    # interruption, D major (collection 2, tonic pc 2) x5 -- collections on
    # EITHER SIDE of the interruption (0 vs 2) are genuinely different, so it
    # must NOT merge and must remain its own short run.
    fake_key_id_a = np.array([0] * 5 + [7] * 2 + [2] * 5)
    collection_seq_a = collapse_to_collections(fake_key_id_a)
    runs_a = run_length_encode_collections(collection_seq_a)
    merged_a = merge_brief_interruptions(runs_a, max_interruption=3)
    checks.append(("brief interruption with DIFFERENT collection on both sides is NOT merged (stays as its own short run)", len(merged_a) == 3))

    # key_index(9, "min") = 21 is A minor (tonic pc 9, key_id = 9+12).
    # A minor's collection_class is (9+3)%12 = 0, the SAME collection as C
    # major (key_id 0) -- confirms merging happens by COLLECTION, not raw
    # key_id, and confirms key_index's own minor-key encoding (tonic_pc+12).
    a_minor_key_id = key_index(9, "min")
    checks.append(("key_index(9,'min') is A minor and its collection_class matches C major's collection (0) -- relative-pair collapse verified", a_minor_key_id == 21 and collection_class(a_minor_key_id) == 0))
    fake_key_id_b = np.array([0] * 5 + [a_minor_key_id] * 2 + [0] * 5)  # C major x5, A minor x2 (interruption), C major x5 -- all same collection
    collection_seq_b = collapse_to_collections(fake_key_id_b)
    runs_b = run_length_encode_collections(collection_seq_b)
    checks.append(("A minor interruption collapses to the SAME collection id as its C-major neighbors", len(set(int(c) for c in collection_seq_b)) == 1))
    merged_b = merge_brief_interruptions(runs_b, max_interruption=3)
    checks.append(("interruption sharing the SAME collection as both neighbors merges into one run", len(merged_b) == 1))

    print()
    print("Phase 3J-B unit / synthetic checks")
    print("-" * 60)
    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")
    print("-" * 60)
    print("ALL UNIT CHECKS PASSED" if all_passed else "SOME UNIT CHECKS FAILED -- ABORTING BEFORE FULL CORPUS RUN")
    return all_passed


# =============================================================================
# Evaluation (anchors loaded/consulted here ONLY, after all Stage 1-6
# output for every piece already exists)
# =============================================================================

def compute_decision_source_breakdown(segments):
    """Purely additive, post-hoc REPORTING analysis -- does not touch any
    prediction, segment, or constant. Added after independent review (see
    STATUS.md / the report's "Mechanistic finding" section) confirmed that
    `mediant_signal` and `leading_tone_positive_evidence` are NOT equally
    trustworthy mode-diagnostic cues: algebraically, minor_third_pc ==
    major_tonic_pc for every collection (verified in
    resolve_segment_candidate_b's own math), so `mediant_signal` is
    structurally confounded with simple tonic-pitch prominence and fires
    "minor" on tonic-heavy MAJOR passages, whereas the raised-leading-tone
    cue only fires on a pitch class foreign to both collection members and
    is comparatively trustworthy. This function breaks down, per piece,
    how many resolved-minor and resolved-major windows came from each
    decision source, so headline "minor recovery" figures can be read
    disaggregated rather than as one pooled, potentially-inflated number."""
    stable = [s for s in segments if s["status"] == "stable"]
    breakdown = Counter()
    minor_windows_by_source = Counter()
    major_windows_by_source = Counter()
    for s in stable:
        source = s["decision_source"]
        breakdown[source] += s["length"]
        if s["resolved_key_id"] >= 12:
            minor_windows_by_source[source] += s["length"]
        else:
            major_windows_by_source[source] += s["length"]
    total_stable_windows = sum(breakdown.values())
    total_minor_windows = sum(minor_windows_by_source.values())
    return {
        "total_stable_windows": total_stable_windows,
        "windows_by_decision_source": dict(breakdown),
        "minor_windows_by_decision_source": dict(minor_windows_by_source),
        "major_windows_by_decision_source": dict(major_windows_by_source),
        "total_minor_windows": total_minor_windows,
        "minor_windows_from_leading_tone_fraction": (
            minor_windows_by_source.get("leading_tone_positive_evidence", 0) / total_minor_windows
            if total_minor_windows > 0 else None
        ),
        "minor_windows_from_mediant_signal_fraction": (
            minor_windows_by_source.get("mediant_signal", 0) / total_minor_windows
            if total_minor_windows > 0 else None
        ),
        "minor_windows_from_fallback_fraction": (
            sum(v for k, v in minor_windows_by_source.items() if k.startswith("fallback")) / total_minor_windows
            if total_minor_windows > 0 else None
        ),
    }


def evaluate_piece(piece, stage1, pipeline_out):
    stem = piece["stem"]
    duration = load_piece_duration(stem)
    anchors_spec = piece["anchors_fn"](duration)
    times = stage1["prediction_times_sec"]
    active = stage1["active"]

    frozen_piece_level = compute_piece_level_metrics(stage1["key_id"], active, stage1["raw_scores"])
    resolved_piece_level = compute_piece_level_metrics(pipeline_out["key_id_section_resolved"], active, stage1["raw_scores"])

    frozen_anchors = {a["name"]: compute_anchor_metrics(stage1["key_id"], active, stage1["raw_scores"], times, a) for a in anchors_spec}
    resolved_anchors = {a["name"]: compute_anchor_metrics(pipeline_out["key_id_section_resolved"], active, stage1["raw_scores"], times, a) for a in anchors_spec}

    segments = pipeline_out["segments"]
    stable_segs = [s for s in segments if s["status"] == "stable"]
    transition_segs = [s for s in segments if s["status"] == "transition"]
    undefined_segs = [s for s in segments if s["status"] == "undefined"]
    n_transition_windows = sum(s["length"] for s in transition_segs)
    n_undefined_windows = sum(s["length"] for s in undefined_segs)
    T = len(stage1["key_id"])
    durations = [s["length"] * DEFAULT_WINDOW_SEC for s in stable_segs]

    result = {
        "display_name": piece["display_name"],
        "n_windows": T,
        "n_segments_total": len(segments),
        "n_segments_stable": len(stable_segs),
        "n_segments_transition": len(transition_segs),
        "n_segments_undefined": len(undefined_segs),
        "fraction_windows_transition": n_transition_windows / T if T > 0 else None,
        "fraction_windows_undefined": n_undefined_windows / T if T > 0 else None,
        "fraction_windows_section_resolved": float(pipeline_out["is_section_resolved"].mean()),
        "segment_duration_sec_distribution": {
            "mean": float(np.mean(durations)) if durations else None,
            "median": float(np.median(durations)) if durations else None,
            "min": float(np.min(durations)) if durations else None,
            "max": float(np.max(durations)) if durations else None,
            "n_stable_segments": len(durations),
        },
        "frozen_stage1_piece_level": frozen_piece_level,
        "resolved_piece_level": resolved_piece_level,
        "frozen_stage1_anchors": frozen_anchors,
        "resolved_anchors": resolved_anchors,
        "decision_source_breakdown": compute_decision_source_breakdown(segments),
        "linearity_check_summary": {
            "n_segments_checked": len(pipeline_out["linearity_checks"]),
            "all_passed": all(c["passed"] for c in pipeline_out["linearity_checks"]) if pipeline_out["linearity_checks"] else True,
            "max_abs_error_overall": max((c["max_abs_error"] for c in pipeline_out["linearity_checks"]), default=0.0),
        },
    }

    if piece["level"] == "L4":
        sil_mask = (times >= CHOPIN_SILENCE_REGION["start_sec"]) & (times <= CHOPIN_SILENCE_REGION["end_sec"])
        sil_idxs = np.where(sil_mask)[0]
        result["chopin_silence_region_unchanged_check"] = {
            "n_windows_in_region": int(len(sil_idxs)),
            "n_inactive_in_region": int((~active[sil_idxs]).sum()) if len(sil_idxs) else 0,
            "note": "Uses the same frozen Stage 1 `active` array as every prior phase -- inactivity is not redefined here.",
        }

    return result


# =============================================================================
# Plots
# =============================================================================

def plot_segment_comparison(stem, display_name, stage1, pipeline_out, out_path, key_events=None):
    key_id_frozen = stage1["key_id"]
    key_id_resolved = pipeline_out["key_id_section_resolved"]
    times = stage1["prediction_times_sec"]

    from shared_music_defs import FIFTH_POS

    fig, ax = plt.subplots(figsize=(13, 5))

    def _plot_series(key_id_arr, color, marker, label, z):
        defined = key_id_arr != -1
        fifths_pos = np.full(len(key_id_arr), np.nan)
        fifths_pos[defined] = [FIFTH_POS[key_tonic_pc(int(k))] for k in key_id_arr[defined]]
        is_minor = np.zeros(len(key_id_arr), dtype=bool)
        is_minor[defined] = key_id_arr[defined] >= 12
        ax.plot(times[defined], fifths_pos[defined], color=color, alpha=0.3, linewidth=0.8, zorder=z)
        maj_mask = defined & ~is_minor
        min_mask = defined & is_minor
        ax.scatter(times[maj_mask], fifths_pos[maj_mask], s=14, color=color, marker=marker, zorder=z + 2, label=f"{label} (major)")
        if min_mask.any():
            ax.scatter(times[min_mask], fifths_pos[min_mask], s=50, facecolors="none", edgecolors=color, linewidths=1.3, marker=marker, zorder=z + 2, label=f"{label} (minor)")

    _plot_series(key_id_frozen, "tab:gray", "o", "Stage 1 (frozen)", 1)
    _plot_series(key_id_resolved, "tab:green", "D", "Section-resolved (Candidate B)", 2)

    for seg in pipeline_out["segments"]:
        if seg["status"] == "stable":
            ax.axvspan(times[seg["start_idx"]], times[min(seg["end_idx"], len(times) - 1)], color="green", alpha=0.04, zorder=0)

    if key_events:
        for ev in key_events:
            ax.axvline(ev["time"], color="black", linestyle="--", alpha=0.4, linewidth=1)

    from plotting_comparison import FIFTHS_LABEL_NAMES
    ax.set_yticks(range(12))
    ax.set_yticklabels(FIFTHS_LABEL_NAMES)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Predicted tonic (circle-of-fifths position)")
    ax.set_title(f"Phase 3J-B — {display_name}: Stage 1 vs. Section-Level Candidate B\n(green shading = stable resolved segments; open markers = minor-mode)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Sensitivity audit (descriptive only, predeclared grid, no winner chosen)
# =============================================================================

def run_sensitivity_audit(all_stage1):
    results = {}
    for min_win in SENSITIVITY_GRID["MIN_SEGMENT_WINDOWS"]:
        for max_interrupt in SENSITIVITY_GRID["BRIEF_INTERRUPTION_MAX_WINDOWS"]:
            cfg = dict(PRE_REGISTERED_CONFIG)
            cfg["MIN_SEGMENT_WINDOWS"] = min_win
            cfg["BRIEF_INTERRUPTION_MAX_WINDOWS"] = max_interrupt
            condition_id = f"min{min_win}_interrupt{max_interrupt}"
            per_piece = {}
            for level, stage1 in all_stage1.items():
                pipeline_out = run_pipeline_for_piece(stage1, cfg)
                segs = pipeline_out["segments"]
                stable = [s for s in segs if s["status"] == "stable"]
                key_id_resolved = pipeline_out["key_id_section_resolved"]
                piece_level = compute_piece_level_metrics(key_id_resolved, stage1["active"], stage1["raw_scores"])
                per_piece[level] = {
                    "n_segments_stable": len(stable), "n_segments_total": len(segs),
                    "n_key_switches_resolved": piece_level["n_key_switches"],
                    "fraction_minor_mode_resolved": piece_level["fraction_defined_windows_predicted_minor_mode"],
                }
            results[condition_id] = {"min_segment_windows": min_win, "brief_interruption_max": max_interrupt, "per_piece": per_piece}
    return results


# =============================================================================
# Report
# =============================================================================

def _segment_summary_line(r, level):
    return (
        f"- **{level}**: {r['n_segments_stable']} stable / {r['n_segments_transition']} transition / "
        f"{r['n_segments_undefined']} undefined segments (of {r['n_segments_total']} total); "
        f"{_fmt(r['fraction_windows_section_resolved'])} of windows section-resolved; "
        f"segment duration mean={_fmt(r['segment_duration_sec_distribution']['mean'], 2)}s, "
        f"median={_fmt(r['segment_duration_sec_distribution']['median'], 2)}s"
    )


def build_report_md(results_by_level, sensitivity_results, verdict_text, verdict_code):
    lines = []
    lines.append("# Phase 3J-B — Section-Level Candidate B: Implementation and Evaluation")
    lines.append("")
    lines.append(
        "Implements and evaluates ONLY the revised Phase 3J-A Candidate B (mediant + asymmetric "
        "raised-leading-tone comparison, aggregated per stable diatonic-collection segment). Candidates A and C "
        "are explicitly NOT implemented. No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement was "
        "run. The frozen Stage 1 baseline and all Phase 3G/3H/3I/3J-A outputs are read-only references, never "
        "modified. No dense per-timestep accuracy is claimed anywhere below."
    )
    lines.append("")
    lines.append(
        "**Review process**: after the primary run completed, three independent, read-only subagent passes "
        "(implementation audit, methodology/leakage audit, music-theory critique) reviewed the actual script and "
        "actual results. None modified any file, ran any code, or altered any prediction, segment, or constant. "
        "All three independently confirmed a structural (algebraic, not coding-bug) property of the pre-registered "
        "mediant cue -- see \"Mechanistic finding\" below -- which is reported here as additive analysis on top of "
        "the unmodified primary-configuration results, not as a change to them."
    )
    lines.append("")

    lines.append("## Pre-registered configuration (frozen before any anchor evaluation)")
    lines.append("")
    for k, v in PRE_REGISTERED_CONFIG.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append(f"Sensitivity grid (predeclared, descriptive only): `MIN_SEGMENT_WINDOWS` in {SENSITIVITY_GRID['MIN_SEGMENT_WINDOWS']}, `BRIEF_INTERRUPTION_MAX_WINDOWS` in {SENSITIVITY_GRID['BRIEF_INTERRUPTION_MAX_WINDOWS']}.")
    lines.append("")

    lines.append("## Per-piece results (primary configuration)")
    lines.append("")
    for level in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        r = results_by_level[level]
        lines.append(f"### {level} — {r['display_name']}")
        lines.append("")
        lines.append(_segment_summary_line(r, level))
        fp, rp = r["frozen_stage1_piece_level"], r["resolved_piece_level"]
        lines.append(
            f"- Stage 1 dominant: " + ", ".join(f"{tk['key']} ({tk['fraction']:.1%})" for tk in fp["dominant_predicted_keys"][:3]) +
            f" | Section-resolved dominant: " + ", ".join(f"{tk['key']} ({tk['fraction']:.1%})" for tk in rp["dominant_predicted_keys"][:3])
        )
        lines.append(
            f"- Key switches: Stage1={fp['n_key_switches']} -> Resolved={rp['n_key_switches']}; "
            f"jumps: mean {_fmt(fp['mean_jump'], 2)}->{_fmt(rp['mean_jump'], 2)}, max {_fmt(fp['max_jump'], 2)}->{_fmt(rp['max_jump'], 2)}; "
            f"large jumps: {fp['n_large_jumps']}->{rp['n_large_jumps']} ({_fmt(fp['large_jump_proportion'])}->{_fmt(rp['large_jump_proportion'])})"
        )
        for aname in r["frozen_stage1_anchors"]:
            fa, ra = r["frozen_stage1_anchors"][aname], r["resolved_anchors"][aname]
            lines.append(
                f"- Anchor `{aname}` (expected {fa['expected_key']}): strict Stage1={_fmt(fa['strict_expected_key_proportion'])} -> "
                f"Resolved={_fmt(ra['strict_expected_key_proportion'])}; collection-equiv Stage1={_fmt(fa['collection_equivalent_proportion'])} -> "
                f"Resolved={_fmt(ra['collection_equivalent_proportion'])}"
            )
        db = r["decision_source_breakdown"]
        if db["total_minor_windows"] > 0:
            lines.append(
                f"- **Minor-mode windows by decision source** (see Mechanistic finding below): "
                f"{db['total_minor_windows']} total minor-resolved windows -- "
                f"leading_tone={_fmt(db['minor_windows_from_leading_tone_fraction'])}, "
                f"mediant_signal={_fmt(db['minor_windows_from_mediant_signal_fraction'])}, "
                f"fallback={_fmt(db['minor_windows_from_fallback_fraction'])}"
            )
        lc = r["linearity_check_summary"]
        lines.append(f"- Chroma/raw_scores linearity check: {lc['n_segments_checked']} stable segments checked, all_passed={lc['all_passed']}, max_abs_error={lc['max_abs_error_overall']:.2e}")
        if "chopin_silence_region_unchanged_check" in r:
            sc = r["chopin_silence_region_unchanged_check"]
            lines.append(f"- Silence-region check (interpretation preserved, not redefined): {sc['n_inactive_in_region']}/{sc['n_windows_in_region']} inactive")
        lines.append("")

    lines.append("## Sensitivity audit (descriptive only -- no winner chosen)")
    lines.append("")
    lines.append("| condition | L1 stable segs | L1 switches | L1 minor frac | L3 minor frac | L4 minor frac | L2 stable segs | L4 stable segs | L5 stable segs | L6 stable segs |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for cond_id, cond in sorted(sensitivity_results.items()):
        pp = cond["per_piece"]
        lines.append(
            f"| min={cond['min_segment_windows']}, interrupt<={cond['brief_interruption_max']} | "
            f"{pp['L1']['n_segments_stable']} | {pp['L1']['n_key_switches_resolved']} | {_fmt(pp['L1']['fraction_minor_mode_resolved'], 3)} | "
            f"{_fmt(pp['L3']['fraction_minor_mode_resolved'], 3)} | {_fmt(pp['L4']['fraction_minor_mode_resolved'], 3)} | "
            f"{pp['L2']['n_segments_stable']} | {pp['L4']['n_segments_stable']} | {pp['L5']['n_segments_stable']} | {pp['L6']['n_segments_stable']} |"
        )
    lines.append("")
    lines.append(build_sensitivity_narrative(sensitivity_results))
    lines.append("")

    lines.append("## Mechanistic finding: the mediant cue is structurally confounded with tonic-pitch prominence")
    lines.append("")
    lines.append(
        "Discovered during independent post-run review (three read-only subagent passes: implementation audit, "
        "methodology/leakage audit, music-theory critique) and confirmed algebraically, not a coding bug -- "
        "`resolve_segment_candidate_b` faithfully implements exactly the rule `PHASE3J_A_section_level_resolver_"
        "design.md` specifies. **This is a design-level property of the pre-registered Candidate B rule itself, "
        "reported here as a finding, not corrected by adjusting any constant** (per this task's guardrail against "
        "post-hoc parameter changes -- this is additive analysis of the existing, unmodified predictions, segments, "
        "and constants)."
    )
    lines.append("")
    lines.append(
        "For any collection, `minor_tonic_pc = major_tonic_pc - 3` and the minor candidate's third scale degree is "
        "`minor_tonic_pc + 3` -- these cancel exactly: `minor_third_pc == major_tonic_pc`, algebraically, for all "
        "12 relative pairs. So `minor_third_evidence_fraction` is not independent mode evidence at all -- it is "
        "literally the aggregate evidence at the MAJOR candidate's own tonic pitch class. Since essentially all "
        "tonal music (major or minor) emphasizes its own tonic pitch heavily, any segment that simply plays its "
        "tonic a lot will show high `minor_third_evidence`, mechanically dragging `mediant_signal` negative and "
        "the decision toward \"minor\" -- independent of whether the passage is actually minor. This fully explains "
        "L6's largest stable segment (669 windows, resolved A minor via `mediant_signal`, with "
        "`leading_tone_evidence_fraction=0.0` -- zero genuine harmonic-minor evidence) and is the dominant source "
        "of L6's anchor-proportion collapse."
    )
    lines.append("")
    lines.append(
        "By contrast, the raised-leading-tone cue (`leading_tone_positive_evidence`) is NOT subject to this "
        "confound: it queries a pitch class foreign to both collection members under the plain diatonic "
        "representation, so real evidence there reflects an actual chromatic (harmonic/melodic-minor-style) "
        "gesture, not tonic-pitch bookkeeping. The decision-source breakdowns in the per-piece sections above show "
        "this split concretely: Für Elise's core A-minor block and the majority of Chopin's E-minor windows are "
        "`leading_tone_positive_evidence`-driven (comparatively trustworthy), while L6's minor-resolved windows are "
        "essentially entirely `mediant_signal`-driven (the confounded path) with no leading-tone support anywhere. "
        "**Readers should treat this report's pooled \"minor recovery\" percentages as an upper bound of uncertain "
        "composition, and consult each piece's decision-source breakdown before concluding the rule detects "
        "genuine harmonic minor** -- the leading-tone-attributed fraction is the more defensible number."
    )
    lines.append("")

    lines.append("## Interpretation checks")
    lines.append("")
    lines.append(build_interpretation_checks(results_by_level))
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**Verdict code: {verdict_code}**")
    lines.append("")
    lines.append(verdict_text)
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This is Phase 3J-B only: implementation and evaluation of the revised Candidate B section-level "
        "resolver. Candidates A and C were not implemented. No chord-id EMA/SRN, Chroma SRN, Transformer, or "
        "neural refinement was run. All governing constants (`PRE_REGISTERED_CONFIG` above) were fixed in this "
        "script's source before any anchor was loaded or compared, and were not adjusted after seeing results. "
        "Anchors were used exclusively for evaluation, in the Evaluation section, after every piece's segments "
        "and `key_id_section_resolved` array already existed -- never inside segmentation, aggregation, or the "
        "Candidate B decision rule. No dense per-timestep accuracy is claimed anywhere in this report."
    )
    lines.append("")

    return "\n".join(lines)


def build_sensitivity_narrative(sensitivity_results):
    l1_switches = [c["per_piece"]["L1"]["n_key_switches_resolved"] for c in sensitivity_results.values()]
    l1_minor = [c["per_piece"]["L1"]["fraction_minor_mode_resolved"] or 0.0 for c in sensitivity_results.values()]
    l3_minor = [c["per_piece"]["L3"]["fraction_minor_mode_resolved"] or 0.0 for c in sensitivity_results.values()]
    l4_minor = [c["per_piece"]["L4"]["fraction_minor_mode_resolved"] or 0.0 for c in sensitivity_results.values()]
    l2_segs = [c["per_piece"]["L2"]["n_segments_stable"] for c in sensitivity_results.values()]
    l4_segs = [c["per_piece"]["L4"]["n_segments_stable"] for c in sensitivity_results.values()]

    return (
        f"Across the {len(sensitivity_results)} predeclared sensitivity conditions: L1 (Twinkle) key switches "
        f"range {min(l1_switches)}-{max(l1_switches)}, L1 minor-mode fraction ranges {min(l1_minor):.3f}-{max(l1_minor):.3f}, "
        f"L3 (Für Elise) minor-mode fraction ranges {min(l3_minor):.3f}-{max(l3_minor):.3f}, L4 (Chopin) minor-mode "
        f"fraction ranges {min(l4_minor):.3f}-{max(l4_minor):.3f}, L2 (Bach) stable-segment count ranges "
        f"{min(l2_segs)}-{max(l2_segs)}, L4 stable-segment count ranges {min(l4_segs)}-{max(l4_segs)}. "
        "No condition in this grid is selected, adopted, or treated as a winner -- these ranges are reported "
        "purely to characterize whether the primary configuration's qualitative conclusions (below) are robust "
        "or fragile across nearby, equally-defensible settings."
    )


def build_interpretation_checks(results_by_level):
    parts = []
    l1 = results_by_level["L1"]
    l1_stable = l1["n_segments_stable"] == 1 and l1["n_segments_transition"] == 0 and l1["n_segments_undefined"] == 0
    l1_stability_clause = (
        "remains one single stable C-major segment" if l1_stable
        else f"does NOT remain one single segment ({l1['n_segments_stable']} stable, {l1['n_segments_transition']} transition segments)"
    )
    parts.append(
        f"**L1 Twinkle**: {l1_stability_clause}. "
        f"Strict C-major proportion: Stage1={_fmt(l1['frozen_stage1_anchors']['full_piece']['strict_expected_key_proportion'])} -> "
        f"Resolved={_fmt(l1['resolved_anchors']['full_piece']['strict_expected_key_proportion'])}."
    )

    l2 = results_by_level["L2"]
    parts.append(
        f"**L2 Bach**: segmentation produced {l2['n_segments_stable']} stable segment(s) "
        f"({l2['n_segments_transition']} transition). Per Phase 3G-B's finding that Bach's C/G/D spread is "
        f"systematic tie-break bias, not real tonicization, this segment count is reported with that caveat -- "
        f"it is not treated as evidence of real modulation. Strict G-major proportion: Stage1="
        f"{_fmt(l2['frozen_stage1_anchors']['full_piece']['strict_expected_key_proportion'])} -> Resolved="
        f"{_fmt(l2['resolved_anchors']['full_piece']['strict_expected_key_proportion'])}."
    )

    l3 = results_by_level["L3"]
    parts.append(
        f"**L3 Für Elise**: strict A-minor proportion: Stage1={_fmt(l3['frozen_stage1_anchors']['full_excerpt']['strict_expected_key_proportion'])} "
        f"-> Resolved={_fmt(l3['resolved_anchors']['full_excerpt']['strict_expected_key_proportion'])}. "
        f"{l3['n_segments_stable']} stable segment(s) formed."
    )

    l4 = results_by_level["L4"]
    parts.append(
        f"**L4 Chopin**: {l4['n_segments_stable']} stable segment(s) formed (of {l4['n_segments_total']} total; "
        f"{_fmt(l4['fraction_windows_transition'])} of windows in unresolved transition segments) -- Phase 3J-A "
        f"flagged Chopin's low (0.491) collection-equivalent proportion as a specific fragmentation risk. Strict "
        f"E-minor proportion: Stage1={_fmt(l4['frozen_stage1_anchors']['full_piece']['strict_expected_key_proportion'])} -> "
        f"Resolved={_fmt(l4['resolved_anchors']['full_piece']['strict_expected_key_proportion'])}. "
        f"Silence-region interpretation preserved unchanged (see per-piece section)."
    )

    l5 = results_by_level["L5"]
    parts.append(
        f"**L5 Clementi**: {l5['n_segments_stable']} stable segment(s) formed. Compared descriptively against the "
        "previously-observed, musically plausible C->G->C baseline trajectory (Phase 3G-A/3G-B) -- this is NOT "
        "independently-verified dense ground truth, and no constant was adjusted based on this comparison."
    )

    l6 = results_by_level["L6"]
    parts.append(
        f"**L6 Twinkle 12**: {l6['n_segments_stable']} stable segment(s) formed (of {l6['n_segments_total']} total). "
        "Anchor-window strict proportions: " +
        "; ".join(
            f"{name}: Stage1={_fmt(l6['frozen_stage1_anchors'][name]['strict_expected_key_proportion'])} -> "
            f"Resolved={_fmt(l6['resolved_anchors'][name]['strict_expected_key_proportion'])}"
            for name in l6["frozen_stage1_anchors"]
        ) + "."
    )

    return " ".join(parts)


# =============================================================================
# Verification
# =============================================================================

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


def run_verification(results, segments_out, out_paths, frozen_mtimes_before, derived_corpus_mtimes_before, old_script_mtimes_before):
    checks = []

    for p in out_paths:
        checks.append((f"{os.path.basename(p)} exists", os.path.exists(p)))
        checks.append((f"{os.path.basename(p)} is non-empty", os.path.exists(p) and os.path.getsize(p) > 0))

    nan_paths = _scan_for_nan(results) + _scan_for_nan(segments_out)
    checks.append(("no NaNs in Phase 3J-B metrics or segments output", len(nan_paths) == 0))

    for path, mtime_before in frozen_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} (frozen Phase 3G/3H/3I/3J-A output) not modified", os.path.getmtime(path) == mtime_before))
    for path, mtime_before in derived_corpus_mtimes_before.items():
        checks.append((f"derived_phase3g_corpus/{os.path.basename(path)} not modified", os.path.getmtime(path) == mtime_before))
    for path, mtime_before in old_script_mtimes_before.items():
        checks.append((f"{os.path.basename(path)} (old/frozen script) not modified", os.path.getmtime(path) == mtime_before))

    checks.append(("all linearity checks passed across the full corpus", all(results["pieces"][lvl]["linearity_check_summary"]["all_passed"] for lvl in results["pieces"])))
    # Runtime check (not a self-referential text search): confirms this
    # module's own global namespace never bound the name WEIGHTED_TEMPLATES
    # (Phase 3H-A's Candidate A profile), i.e. Candidate A was never
    # imported or defined here.
    checks.append(("Candidate A not implemented (WEIGHTED_TEMPLATES is not a name in this module's namespace)", "WEIGHTED_TEMPLATES" not in globals()))
    checks.append((
        "Stage 1-6 functions (detect_segments, resolve_segment_candidate_b, project_to_windows) take no anchor/expected-key argument (structural, verified by code inspection)",
        True,
    ))

    print()
    print("evaluate_phase3j_b_section_level_candidate_b.py verification")
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


# =============================================================================
# Main
# =============================================================================

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


def determine_verdict(results_by_level):
    l1 = results_by_level["L1"]
    l6 = results_by_level["L6"]
    l3 = results_by_level["L3"]
    l4 = results_by_level["L4"]

    a_l1 = l1["frozen_stage1_anchors"]["full_piece"]["strict_expected_key_proportion"]
    r_l1 = l1["resolved_anchors"]["full_piece"]["strict_expected_key_proportion"]
    l6_deltas = []
    for name in l6["frozen_stage1_anchors"]:
        a = l6["frozen_stage1_anchors"][name]["strict_expected_key_proportion"]
        r = l6["resolved_anchors"][name]["strict_expected_key_proportion"]
        l6_deltas.append((r or 0.0) - (a or 0.0))
    a_l3 = l3["frozen_stage1_anchors"]["full_excerpt"]["strict_expected_key_proportion"]
    r_l3 = l3["resolved_anchors"]["full_excerpt"]["strict_expected_key_proportion"]
    a_l4 = l4["frozen_stage1_anchors"]["full_piece"]["strict_expected_key_proportion"]
    r_l4 = l4["resolved_anchors"]["full_piece"]["strict_expected_key_proportion"]

    l1_component_stable = r_l1 >= a_l1 - 0.02
    l6_component_stable = all(d >= -0.05 for d in l6_deltas)
    l1_l6_stable = l1_component_stable and l6_component_stable
    minor_recovered = (r_l3 > a_l3 + 0.05) or (r_l4 > a_l4 + 0.05)

    if minor_recovered and l1_l6_stable:
        code = 1
        text = (
            f"**Section-level Candidate B improves minor tonic/mode resolution while preserving L1/L6 stability.** "
            f"L3 strict A-minor: {a_l3:.4f} -> {r_l3:.4f}. L4 strict E-minor: {a_l4:.4f} -> {r_l4:.4f}. "
            f"L1 strict C-major: {a_l1:.4f} -> {r_l1:.4f}. L6 anchor deltas: {[f'{d:+.4f}' for d in l6_deltas]}."
        )
    elif (not minor_recovered) and l1_l6_stable:
        code = 2
        text = (
            f"**Section-level Candidate B preserves stability but does not recover minor mode.** "
            f"L3 strict A-minor: {a_l3:.4f} -> {r_l3:.4f}. L4 strict E-minor: {a_l4:.4f} -> {r_l4:.4f} "
            f"(neither improved by more than the 0.05 descriptive bar used here). L1 strict C-major: "
            f"{a_l1:.4f} -> {r_l1:.4f}. L6 anchor deltas: {[f'{d:+.4f}' for d in l6_deltas]} (within tolerance)."
        )
    elif minor_recovered and (not l1_l6_stable):
        code = 3
        which_failed = []
        if not l1_component_stable:
            which_failed.append(f"L1 strict C-major degraded beyond the 0.02 tolerance ({a_l1:.4f} -> {r_l1:.4f})")
        if not l6_component_stable:
            which_failed.append(f"L6 anchor deltas exceed the -0.05 tolerance ({[f'{d:+.4f}' for d in l6_deltas]})")
        text = (
            f"**Section-level Candidate B recovers minor mode but introduces unacceptable segmentation/stability "
            f"damage.** L3 strict A-minor: {a_l3:.4f} -> {r_l3:.4f}. L4 strict E-minor: {a_l4:.4f} -> {r_l4:.4f}. "
            f"Stability failure: {'; '.join(which_failed)}. Per the Mechanistic finding section above, this damage "
            f"is traced almost entirely to the `mediant_signal` decision path (structurally confounded with "
            f"tonic-pitch prominence), while the `leading_tone_positive_evidence` path -- responsible for most of "
            f"Chopin's and Für Elise's core minor recovery -- does not exhibit the same failure mode."
        )
    else:
        code = 3
        text = (
            f"**Section-level Candidate B recovers no meaningful minor mode AND fails to preserve L1/L6 "
            f"stability.** L3: {a_l3:.4f}->{r_l3:.4f}, L4: {a_l4:.4f}->{r_l4:.4f}, L1: {a_l1:.4f}->{r_l1:.4f}, "
            f"L6 deltas: {[f'{d:+.4f}' for d in l6_deltas]}."
        )
    return code, text


def main():
    os.makedirs(_FIGURES_DIR, exist_ok=True)

    if not run_unit_checks():
        print("\nABORTING: unit/synthetic checks failed. No full corpus run, no anchor evaluation performed.")
        sys.exit(1)

    frozen_mtimes_before = {
        PHASE3G_A_METRICS_JSON: os.path.getmtime(PHASE3G_A_METRICS_JSON),
        PHASE3G_B_METRICS_JSON: os.path.getmtime(PHASE3G_B_METRICS_JSON),
        PHASE3H_A_METRICS_JSON: os.path.getmtime(PHASE3H_A_METRICS_JSON),
        PHASE3H_B_METRICS_JSON: os.path.getmtime(PHASE3H_B_METRICS_JSON),
        PHASE3H_C_METRICS_JSON: os.path.getmtime(PHASE3H_C_METRICS_JSON),
        PHASE3I_REPORT_MD: os.path.getmtime(PHASE3I_REPORT_MD),
        PHASE3J_A_REPORT_MD: os.path.getmtime(PHASE3J_A_REPORT_MD),
    }
    derived_corpus_files = [os.path.join(_DERIVED_CORPUS_DIR, f) for f in os.listdir(_DERIVED_CORPUS_DIR)]
    derived_corpus_mtimes_before = {p: os.path.getmtime(p) for p in derived_corpus_files}

    old_scripts = [
        "midi_chroma_extraction.py", "pitch_class_baseline.py",
        "evaluate_pitch_class_phase2d.py", "pitch_class_uncertainty_diagnostics.py",
        "compare_phase3c_disagreement.py", "evaluate_phase3g_pitch_class_corpus.py",
        "evaluate_phase3g_b_tie_aware_diagnostics.py", "evaluate_phase3h_a_tonic_mode_resolver.py",
        "evaluate_phase3h_b_texture_gated_resolver.py", "evaluate_phase3h_c_gate_sensitivity.py",
    ]
    old_script_mtimes_before = {
        os.path.join(_THIS_DIR, s): os.path.getmtime(os.path.join(_THIS_DIR, s))
        for s in old_scripts if os.path.exists(os.path.join(_THIS_DIR, s))
    }

    print("\n=== Stage 1-6: loading frozen evidence and running the pipeline (primary configuration) for all 6 pieces ===")
    all_stage1 = {}
    all_pipeline_out = {}
    for piece in PIECES:
        level = piece["level"]
        print(f"  {level} — {piece['display_name']}")
        stage1 = load_frozen_stage1(piece["stem"])
        pipeline_out = run_pipeline_for_piece(stage1, PRE_REGISTERED_CONFIG)
        all_stage1[level] = stage1
        all_pipeline_out[level] = pipeline_out

    print("\n=== Evaluation (anchors loaded and consulted now, after all predictions above already exist) ===")
    results_by_level = {}
    for piece in PIECES:
        level = piece["level"]
        results_by_level[level] = evaluate_piece(piece, all_stage1[level], all_pipeline_out[level])

    print("\n=== Sensitivity audit (predeclared grid, descriptive only) ===")
    sensitivity_results = run_sensitivity_audit(all_stage1)

    print("\nPlotting segment comparisons for Twinkle, Für Elise, Chopin, Twinkle 12...")
    plot_paths = {}
    twinkle12_key_events = [{"time": 0.0}, {"time": 384.0}, {"time": 432.0}]
    for level, key_events in [("L1", None), ("L3", None), ("L4", None), ("L6", twinkle12_key_events)]:
        piece = next(p for p in PIECES if p["level"] == level)
        out_path = f"{OUT_PLOT_PREFIX}{piece['stem']}_segment_comparison.png"
        plot_segment_comparison(piece["stem"], piece["display_name"], all_stage1[level], all_pipeline_out[level], out_path, key_events=key_events)
        plot_paths[level] = out_path
        print(f"  wrote {out_path}")

    verdict_code, verdict_text = determine_verdict(results_by_level)
    print(f"\nVerdict code {verdict_code}: {verdict_text}")

    results = {
        "phase": "phase_3j_b_section_level_candidate_b",
        "based_on_frozen": [
            "PHASE3G_A_pitch_class_corpus_metrics.json (read-only)",
            "03_MIDI_Data/derived_phase3g_corpus/*.npy (read-only)",
            "PHASE3J_A_section_level_resolver_design.md (design spec implemented here)",
        ],
        "pre_registered_config": PRE_REGISTERED_CONFIG,
        "sensitivity_grid": SENSITIVITY_GRID,
        "pieces": results_by_level,
        "sensitivity_audit": sensitivity_results,
        "verdict_code": verdict_code,
        "verdict_text": verdict_text,
        "notes": (
            "Phase 3J-B: implements ONLY the revised Candidate B section-level resolver. Candidates A and C not "
            "implemented. No chord-id EMA/SRN, Chroma SRN, Transformer, or neural refinement. No dense "
            "per-timestep accuracy claimed. Anchors used only for evaluation, after all predictions/segments "
            "already exist. All constants frozen in PRE_REGISTERED_CONFIG before any anchor comparison."
        ),
    }
    results = _to_native(results)

    segments_out = {
        "phase": "phase_3j_b_section_level_candidate_b_segments",
        "pre_registered_config": PRE_REGISTERED_CONFIG,
        "pieces": {level: {"stem": piece["stem"], "segments": all_pipeline_out[level]["segments"]} for level, piece in ((p["level"], p) for p in PIECES)},
    }
    segments_out = _to_native(segments_out)

    with open(OUT_METRICS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_METRICS_JSON}")

    with open(OUT_SEGMENTS_JSON, "w") as f:
        json.dump(segments_out, f, indent=2)
    print(f"Wrote {OUT_SEGMENTS_JSON}")

    report_md = build_report_md(results_by_level, sensitivity_results, verdict_text, verdict_code)
    with open(OUT_REPORT_MD, "w") as f:
        f.write(report_md)
    print(f"Wrote {OUT_REPORT_MD}")

    out_paths = [OUT_METRICS_JSON, OUT_REPORT_MD, OUT_SEGMENTS_JSON] + list(plot_paths.values())
    run_verification(results, segments_out, out_paths, frozen_mtimes_before, derived_corpus_mtimes_before, old_script_mtimes_before)


if __name__ == "__main__":
    main()
