"""midi_chroma_extraction.py

Phase 2C: extract and save raw and smoothed 12-dim chroma sequences from
the existing MIDI files, for the representation Phase 2's pitch-class /
chroma experiments need.

This is extraction only. It does not run the pitch-class/scale-template
baseline (pitch_class_baseline.py, Phase 2B -- already frozen, not
modified here), does not evaluate anything (Phase 2D, not started), and
does not train or define any Chroma SRN (Phase 2E, not started). It also
does not convert chroma to chord ids -- that is the separate, already-done
Phase 1.5A path (midi_chord_extraction.py), which this module does not
touch or duplicate.

Reuses (imports only, does not modify) nothing from the existing
04_Recurrent_Implementation/ modules -- this module only needs pretty_midi
and numpy, since chroma extraction/smoothing is representation-level, not
music-vocabulary-level (unlike shared_music_defs.py's chord/key indexing).
"""

import json
import os
import sys

import numpy as np
import pretty_midi

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_MIDI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "03_MIDI_Data"))
_DERIVED_DIR = os.path.join(_MIDI_DIR, "derived_chroma_sequences")
_FIGURES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "05_Figures_Results"))

DEFAULT_WINDOW_SEC = 0.5
DEFAULT_MEMORY_DECAY = 0.8  # matches pitch_class_baseline.py's DEFAULT_MEMORY_DECAY (the notebook's pitch-class cell, not the chord path's 0.6)
DEFAULT_THRESHOLD_RATIO = 0.10

REPORT_MD = os.path.join(_FIGURES_DIR, "PHASE2C_MIDI_chroma_extraction_report.md")


# ---------------------------------------------------------------------------
# extract_chroma_sequence
#
# Produces three parallel (T, 12) arrays:
#   raw_chroma          -- pm.get_chroma() output, transposed to (T, 12)
#   smoothed_chroma      -- chroma-level EMA (memory_decay), same formula as
#                            pitch_class_baseline.py / the notebook's cells
#   thresholded_smoothed_chroma -- smoothed_chroma with per-timestep values
#                            below threshold_ratio * that timestep's max
#                            zeroed out (same 10%-of-max convention used
#                            throughout this workspace)
#
# No chord-id conversion, no scale-template matching, no MLP/SRN -- this
# function only produces the chroma representation itself.
# ---------------------------------------------------------------------------

def extract_chroma_sequence(midi_path, window_sec=DEFAULT_WINDOW_SEC, memory_decay=DEFAULT_MEMORY_DECAY, threshold_ratio=DEFAULT_THRESHOLD_RATIO, return_metadata=True):
    pm = pretty_midi.PrettyMIDI(midi_path)
    chroma = pm.get_chroma(fs=(1.0 / window_sec))  # (12, T)
    raw_chroma = chroma.T.copy()  # (T, 12)

    T = raw_chroma.shape[0]
    smoothed_chroma = np.zeros((T, 12))
    prev = np.zeros(12)
    for t in range(T):
        prev = memory_decay * prev + (1 - memory_decay) * raw_chroma[t]
        smoothed_chroma[t] = prev

    thresholded_smoothed_chroma = smoothed_chroma.copy()
    for t in range(T):
        row = thresholded_smoothed_chroma[t]
        row_max = np.max(row)
        if row_max > 0.0:
            threshold = threshold_ratio * row_max
            row[row < threshold] = 0

    if not return_metadata:
        return raw_chroma, smoothed_chroma, thresholded_smoothed_chroma

    total_notes = sum(len(inst.notes) for inst in pm.instruments)

    key_signature_changes = [
        {
            "time": float(ks.time),
            "key_number": int(ks.key_number),
            "key_name": pretty_midi.key_number_to_key_name(ks.key_number),
        }
        for ks in pm.key_signature_changes
    ]
    time_signature_changes = [
        {"time": float(ts.time), "numerator": int(ts.numerator), "denominator": int(ts.denominator)}
        for ts in pm.time_signature_changes
    ]

    nonzero_window_count = int(np.sum(np.any(raw_chroma > 0, axis=1)))
    mean_chroma_energy = float(raw_chroma.mean())

    first_5_raw_summary = [
        {"t": t, "nonzero_pcs": int(np.sum(raw_chroma[t] > 0)), "max_val": float(raw_chroma[t].max())}
        for t in range(min(5, T))
    ]

    metadata = {
        "midi_path": os.path.relpath(midi_path, os.path.join(_THIS_DIR, "..")),
        "duration_sec": float(pm.get_end_time()),
        "window_sec": window_sec,
        "memory_decay": memory_decay,
        "threshold_ratio": threshold_ratio,
        "raw_chroma_shape": list(raw_chroma.shape),
        "smoothed_chroma_shape": list(smoothed_chroma.shape),
        "thresholded_smoothed_chroma_shape": list(thresholded_smoothed_chroma.shape),
        "total_notes": int(total_notes),
        "n_instruments": len(pm.instruments),
        "key_signature_changes": key_signature_changes,
        "time_signature_changes": time_signature_changes,
        "first_5_raw_chroma_rows_summary": first_5_raw_summary,
        "mean_chroma_energy": mean_chroma_energy,
        "nonzero_window_count": nonzero_window_count,
    }

    return raw_chroma, smoothed_chroma, thresholded_smoothed_chroma, metadata


# ---------------------------------------------------------------------------
# Extraction + save
# ---------------------------------------------------------------------------

def extract_and_save(midi_filename, out_stem):
    midi_path = os.path.join(_MIDI_DIR, midi_filename)
    raw_chroma, smoothed_chroma, thresholded_smoothed_chroma, metadata = extract_chroma_sequence(
        midi_path, window_sec=DEFAULT_WINDOW_SEC, memory_decay=DEFAULT_MEMORY_DECAY, threshold_ratio=DEFAULT_THRESHOLD_RATIO, return_metadata=True
    )

    os.makedirs(_DERIVED_DIR, exist_ok=True)

    decay_tag = f"decay{int(round(DEFAULT_MEMORY_DECAY * 10)):02d}"
    raw_path = os.path.join(_DERIVED_DIR, f"{out_stem}_raw_chroma.npy")
    smoothed_path = os.path.join(_DERIVED_DIR, f"{out_stem}_smoothed_chroma_{decay_tag}.npy")
    thresholded_path = os.path.join(_DERIVED_DIR, f"{out_stem}_thresholded_smoothed_chroma_{decay_tag}.npy")
    metadata_path = os.path.join(_DERIVED_DIR, f"{out_stem}_chroma_metadata.json")

    np.save(raw_path, raw_chroma)
    np.save(smoothed_path, smoothed_chroma)
    np.save(thresholded_path, thresholded_smoothed_chroma)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    paths = {"raw": raw_path, "smoothed": smoothed_path, "thresholded": thresholded_path, "metadata": metadata_path}
    return raw_chroma, smoothed_chroma, thresholded_smoothed_chroma, metadata, paths


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _metadata_section(title, metadata, paths):
    lines = [f"### {title}", ""]
    lines.append(f"- MIDI path: `{metadata['midi_path']}`")
    lines.append(f"- Duration: {metadata['duration_sec']:.2f} sec")
    lines.append(f"- window_sec: {metadata['window_sec']}")
    lines.append(f"- memory_decay: {metadata['memory_decay']}")
    lines.append(f"- threshold_ratio: {metadata['threshold_ratio']}")
    lines.append(f"- raw_chroma_shape: {tuple(metadata['raw_chroma_shape'])}")
    lines.append(f"- smoothed_chroma_shape: {tuple(metadata['smoothed_chroma_shape'])}")
    lines.append(f"- thresholded_smoothed_chroma_shape: {tuple(metadata['thresholded_smoothed_chroma_shape'])}")
    lines.append(f"- Total notes: {metadata['total_notes']}")
    lines.append(f"- Instruments: {metadata['n_instruments']}")
    lines.append(f"- Mean chroma energy: {metadata['mean_chroma_energy']:.4f}")
    lines.append(f"- Nonzero window count: {metadata['nonzero_window_count']}")
    lines.append(
        f"- Saved: `{os.path.relpath(paths['raw'], os.path.join(_FIGURES_DIR, '..'))}`, "
        f"`{os.path.relpath(paths['smoothed'], os.path.join(_FIGURES_DIR, '..'))}`, "
        f"`{os.path.relpath(paths['thresholded'], os.path.join(_FIGURES_DIR, '..'))}`, "
        f"`{os.path.relpath(paths['metadata'], os.path.join(_FIGURES_DIR, '..'))}`"
    )
    lines.append("")

    if metadata["key_signature_changes"]:
        lines.append("Key signature changes:")
        lines.append("")
        for ks in metadata["key_signature_changes"]:
            lines.append(f"- t={ks['time']:.1f}s: {ks['key_name']} (key_number={ks['key_number']})")
        lines.append("")
    else:
        lines.append("Key signature changes: none present in this MIDI file.")
        lines.append("")

    if metadata["time_signature_changes"]:
        lines.append("Time signature changes:")
        lines.append("")
        for ts in metadata["time_signature_changes"]:
            lines.append(f"- t={ts['time']:.1f}s: {ts['numerator']}/{ts['denominator']}")
        lines.append("")
    else:
        lines.append("Time signature changes: none present in this MIDI file.")
        lines.append("")

    lines.append("First 5 raw chroma windows (summary):")
    lines.append("")
    for row in metadata["first_5_raw_chroma_rows_summary"]:
        lines.append(f"- t={row['t']}: nonzero_pcs={row['nonzero_pcs']}, max_val={row['max_val']:.3f}")
    lines.append("")

    return lines


def build_report_md(twinkle_meta, twinkle_paths, mozart_meta, mozart_paths, verification_checks):
    lines = []
    lines.append("# Phase 2C — MIDI Chroma Extraction Report")
    lines.append("")
    lines.append(
        "Phase 2C extracts and saves raw and smoothed 12-dim chroma sequences from `Twinkle.mid` and "
        "`Twinkle 12.mid`, for use by Phase 2's pitch-class/chroma experiments. **This is extraction "
        "only.** No pitch-class evaluation, Chroma SRN, or Phase 2D work has started here -- this module "
        "produces chroma arrays and metadata, nothing else."
    )
    lines.append("")

    lines.append("## Extraction settings")
    lines.append("")
    lines.append(f"- window_sec: {DEFAULT_WINDOW_SEC}")
    lines.append(f"- memory_decay: {DEFAULT_MEMORY_DECAY} (matches `pitch_class_baseline.py`'s pitch-class-path smoothing constant)")
    lines.append(f"- threshold_ratio: {DEFAULT_THRESHOLD_RATIO}")
    lines.append(
        "- Formula: `smoothed_chroma_t = memory_decay * smoothed_chroma_{t-1} + (1 - memory_decay) * raw_chroma_t`; "
        "`thresholded_smoothed_chroma` zeroes values below `threshold_ratio * max` for each timestep independently."
    )
    lines.append("")

    lines.append("## Twinkle.mid")
    lines.append("")
    lines.extend(_metadata_section("Twinkle.mid metadata", twinkle_meta, twinkle_paths))

    lines.append("## Twinkle 12.mid")
    lines.append("")
    lines.extend(_metadata_section("Twinkle 12.mid metadata", mozart_meta, mozart_paths))

    lines.append("## Verification results")
    lines.append("")
    for label, passed in verification_checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"- [{status}] {label}")
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This is **Phase 2C only**: chroma extraction and verification. No pitch-class evaluation "
        "(Phase 2D), Chroma SRN (Phase 2E), or any other model/evaluation code has been run on these "
        "chroma sequences yet. Original MIDI files were not modified."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_verification(twinkle_raw, twinkle_smoothed, twinkle_thresholded, twinkle_meta,
                      mozart_raw, mozart_smoothed, mozart_thresholded, mozart_meta,
                      twinkle_paths, mozart_paths, midi_mtimes_before):
    checks = []

    for name, arr in [
        ("Twinkle.mid raw_chroma", twinkle_raw),
        ("Twinkle.mid smoothed_chroma", twinkle_smoothed),
        ("Twinkle.mid thresholded_smoothed_chroma", twinkle_thresholded),
        ("Twinkle 12.mid raw_chroma", mozart_raw),
        ("Twinkle 12.mid smoothed_chroma", mozart_smoothed),
        ("Twinkle 12.mid thresholded_smoothed_chroma", mozart_thresholded),
    ]:
        checks.append((f"{name} exists and is non-empty", arr is not None and arr.size > 0))
        checks.append((f"{name} shape is (T, 12)", arr.ndim == 2 and arr.shape[1] == 12))

    checks.append(("Twinkle.mid has about 106 timesteps", abs(twinkle_raw.shape[0] - 106) <= 2))
    checks.append(("Twinkle 12.mid has about 1374 timesteps", abs(mozart_raw.shape[0] - 1374) <= 2))

    all_arrays = [twinkle_raw, twinkle_smoothed, twinkle_thresholded, mozart_raw, mozart_smoothed, mozart_thresholded]
    checks.append(("no NaNs in any chroma array", not any(np.isnan(a).any() for a in all_arrays)))

    checks.append(("Twinkle.mid smoothed differs from raw", not np.array_equal(twinkle_raw, twinkle_smoothed)))
    checks.append(("Twinkle 12.mid smoothed differs from raw", not np.array_equal(mozart_raw, mozart_smoothed)))

    checks.append(("Twinkle.mid thresholded shape matches smoothed shape", twinkle_thresholded.shape == twinkle_smoothed.shape))
    checks.append(("Twinkle 12.mid thresholded shape matches smoothed shape", mozart_thresholded.shape == mozart_smoothed.shape))

    checks.append(("Twinkle 12.mid key signature events recorded", len(mozart_meta["key_signature_changes"]) > 0))

    for label, path in [("Twinkle.mid MIDI file", os.path.join(_MIDI_DIR, "Twinkle.mid")), ("Twinkle 12.mid MIDI file", os.path.join(_MIDI_DIR, "Twinkle 12.mid"))]:
        checks.append((f"{label} not modified (mtime unchanged)", os.path.getmtime(path) == midi_mtimes_before[path]))

    for label, path in twinkle_paths.items():
        checks.append((f"Twinkle.mid {label} file exists", os.path.exists(path)))
    for label, path in mozart_paths.items():
        checks.append((f"Twinkle 12.mid {label} file exists", os.path.exists(path)))

    print()
    print("midi_chroma_extraction.py verification")
    print("-" * 50)
    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")
    print("-" * 50)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")

    return checks, all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(_FIGURES_DIR, exist_ok=True)

    twinkle_midi_path = os.path.join(_MIDI_DIR, "Twinkle.mid")
    mozart_midi_path = os.path.join(_MIDI_DIR, "Twinkle 12.mid")
    midi_mtimes_before = {
        twinkle_midi_path: os.path.getmtime(twinkle_midi_path),
        mozart_midi_path: os.path.getmtime(mozart_midi_path),
    }

    print("Extracting Twinkle.mid chroma...")
    twinkle_raw, twinkle_smoothed, twinkle_thresholded, twinkle_meta, twinkle_paths = extract_and_save("Twinkle.mid", "Twinkle_mid")
    print(f"  duration={twinkle_meta['duration_sec']:.2f}s raw_shape={tuple(twinkle_meta['raw_chroma_shape'])}")

    print("\nExtracting Twinkle 12.mid chroma...")
    mozart_raw, mozart_smoothed, mozart_thresholded, mozart_meta, mozart_paths = extract_and_save("Twinkle 12.mid", "Twinkle_12_mid")
    print(f"  duration={mozart_meta['duration_sec']:.2f}s raw_shape={tuple(mozart_meta['raw_chroma_shape'])}")
    print(f"  key_signature_changes: {mozart_meta['key_signature_changes']}")

    checks, all_passed = run_verification(
        twinkle_raw, twinkle_smoothed, twinkle_thresholded, twinkle_meta,
        mozart_raw, mozart_smoothed, mozart_thresholded, mozart_meta,
        twinkle_paths, mozart_paths, midi_mtimes_before,
    )

    report_md = build_report_md(twinkle_meta, twinkle_paths, mozart_meta, mozart_paths, checks)
    with open(REPORT_MD, "w") as f:
        f.write(report_md)
    print(f"\nWrote {REPORT_MD}")


if __name__ == "__main__":
    main()
