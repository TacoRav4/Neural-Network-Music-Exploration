"""midi_chord_extraction.py

Phase 1.5A: MIDI-to-chord-id extraction, reproducing the notebook's
representation path exactly:

    MIDI -> chroma windows -> chroma-level EMA -> triad template matching -> chord_ids

This is the same `midi_to_chord_ids` logic already used in
02_Baseline_Pipeline/Mini Capstone_Project_A_walking_machine_of_the_music.ipynb
(cell 6) and already frozen/reused for the true-Twinkle baseline figures --
copied here, not reinterpreted, so Phase 1.5B can later evaluate the
EMA+MLP baseline and the SRN on the exact same MIDI-derived chord-id
sequences (both still consuming the 24-dim one-hot chord/triad
representation -- raw chroma is never fed to either model).

This module only extracts and saves chord-id sequences + metadata. It does
not train, load, or evaluate either model, and does not produce any
evaluation plots -- that is Phase 1.5B, not started here.
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
_DERIVED_DIR = os.path.join(_MIDI_DIR, "derived_chord_sequences")
_FIGURES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "05_Figures_Results"))

from shared_music_defs import CHORD_TEMPLATES, decode_chord

DEFAULT_WINDOW_SEC = 0.5
DEFAULT_MEMORY_DECAY = 0.6

REPORT_MD = os.path.join(_FIGURES_DIR, "PHASE1_5A_MIDI_chord_extraction_report.md")


# ---------------------------------------------------------------------------
# midi_to_chord_ids
#
# Formula copied as-is from the notebook's cell 6 -- do not change:
#   smoothed_chroma = memory_decay * smoothed_chroma + (1 - memory_decay) * window_chroma
#   threshold values below 10% of the smoothed window's max to 0
#   best_chord_id = argmax(CHORD_TEMPLATES . working_chroma)
# Preserves the original limitation: no explicit "no chord"/ambiguity
# output -- an all-zero window just repeats the previous chord id (or is
# skipped if it's the very first window), exactly as the notebook does.
# ---------------------------------------------------------------------------

def midi_to_chord_ids(midi_path, window_sec=DEFAULT_WINDOW_SEC, memory_decay=DEFAULT_MEMORY_DECAY, return_metadata=True):
    """
    Returns:
      chord_ids: (N,) numpy int64 array of chord indices (0..23), N <= number
        of chroma windows (fewer only if the very first window(s) are silent
        and there is no previous chord yet to repeat -- matches notebook
        behavior).
      metadata: dict (only if return_metadata=True; see module docstring /
        report for the fields included).
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    chroma = pm.get_chroma(fs=(1.0 / window_sec))

    chord_sequence = []
    smoothed_chroma = np.zeros(12)

    for t in range(chroma.shape[1]):
        window_chroma = chroma[:, t].copy()
        smoothed_chroma = (memory_decay * smoothed_chroma) + ((1 - memory_decay) * window_chroma)

        working_chroma = smoothed_chroma.copy()
        if np.max(working_chroma) > 0:
            threshold = 0.1 * np.max(working_chroma)
            working_chroma[working_chroma < threshold] = 0

        if np.sum(working_chroma) == 0:
            if len(chord_sequence) > 0:
                chord_sequence.append(chord_sequence[-1])
            continue

        scores = CHORD_TEMPLATES.dot(working_chroma)
        best_chord_id = int(np.argmax(scores))
        chord_sequence.append(best_chord_id)

    chord_ids = np.array(chord_sequence, dtype=np.int64)

    if not return_metadata:
        return chord_ids

    total_notes = sum(len(inst.notes) for inst in pm.instruments)

    histogram = {}
    for c in chord_ids.tolist():
        name, qual = decode_chord(c)
        label = f"{name}:{qual}"
        histogram[label] = histogram.get(label, 0) + 1
    top_chords = sorted(histogram.items(), key=lambda kv: -kv[1])[:8]

    first_15_decoded = [f"{decode_chord(c)[0]}:{decode_chord(c)[1]}" for c in chord_ids[:15].tolist()]

    metadata = {
        "midi_path": os.path.relpath(midi_path, os.path.join(_THIS_DIR, "..")),
        "duration_sec": float(pm.get_end_time()),
        "window_sec": window_sec,
        "memory_decay": memory_decay,
        "n_chroma_windows": int(chroma.shape[1]),
        "n_chord_ids": int(len(chord_ids)),
        "total_notes": int(total_notes),
        "n_instruments": len(pm.instruments),
        "chord_histogram": histogram,
        "top_chords": top_chords,
        "first_15_decoded_chords": first_15_decoded,
    }

    return chord_ids, metadata


# ---------------------------------------------------------------------------
# Extraction + save
# ---------------------------------------------------------------------------

def extract_and_save(midi_filename, out_stem):
    midi_path = os.path.join(_MIDI_DIR, midi_filename)
    chord_ids, metadata = midi_to_chord_ids(midi_path, window_sec=DEFAULT_WINDOW_SEC, memory_decay=DEFAULT_MEMORY_DECAY, return_metadata=True)

    os.makedirs(_DERIVED_DIR, exist_ok=True)
    npy_path = os.path.join(_DERIVED_DIR, f"{out_stem}_chord_ids.npy")
    json_path = os.path.join(_DERIVED_DIR, f"{out_stem}_chord_extraction_metadata.json")

    np.save(npy_path, chord_ids)
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return chord_ids, metadata, npy_path, json_path


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _metadata_section(title, metadata, npy_path, json_path):
    lines = [f"### {title}", ""]
    lines.append(f"- MIDI path: `{metadata['midi_path']}`")
    lines.append(f"- Duration: {metadata['duration_sec']:.2f} sec")
    lines.append(f"- window_sec: {metadata['window_sec']}")
    lines.append(f"- memory_decay: {metadata['memory_decay']}")
    lines.append(f"- Chroma windows: {metadata['n_chroma_windows']}")
    lines.append(f"- Chord ids extracted: {metadata['n_chord_ids']}")
    lines.append(f"- Total notes: {metadata['total_notes']}")
    lines.append(f"- Instruments: {metadata['n_instruments']}")
    lines.append(f"- Saved: `{os.path.relpath(npy_path, os.path.join(_FIGURES_DIR, '..'))}`, `{os.path.relpath(json_path, os.path.join(_FIGURES_DIR, '..'))}`")
    lines.append("")
    lines.append("Top chords (chord:quality -> count):")
    lines.append("")
    for label, count in metadata["top_chords"]:
        lines.append(f"- {label}: {count}")
    lines.append("")
    lines.append("First 15 decoded chord predictions:")
    lines.append("")
    lines.append("`" + " | ".join(metadata["first_15_decoded_chords"]) + "`")
    lines.append("")
    return lines


def build_report_md(twinkle_meta, twinkle_npy, twinkle_json, mozart_meta, mozart_npy, mozart_json, verification_checks):
    lines = []
    lines.append("# Phase 1.5A — MIDI Chord-ID Extraction Report")
    lines.append("")
    lines.append(
        "Phase 1.5A extracts MIDI-derived chord-id sequences using the exact same representation path "
        "as the COGS 202 notebook: MIDI -> chroma windows -> chroma-level EMA -> triad template matching "
        "-> chord_ids. **This is extraction only.** No SRN or EMA+MLP MIDI evaluation has been run yet -- "
        "that is Phase 1.5B, not started here. Raw chroma is never used as model input anywhere in this "
        "workspace; only the resulting 24-dim one-hot chord/triad ids are (or will be) fed to either model."
    )
    lines.append("")

    lines.append("## Extraction settings")
    lines.append("")
    lines.append(f"- window_sec: {DEFAULT_WINDOW_SEC}")
    lines.append(f"- memory_decay: {DEFAULT_MEMORY_DECAY}")
    lines.append(
        "- Formula (unchanged from the notebook): `smoothed_chroma = memory_decay * smoothed_chroma + "
        "(1 - memory_decay) * window_chroma`; values below 10% of the smoothed window's max are thresholded "
        "to 0; chord id = argmax(`CHORD_TEMPLATES` . thresholded chroma)."
    )
    lines.append("")

    lines.append("## Twinkle.mid")
    lines.append("")
    lines.extend(_metadata_section("Twinkle.mid metadata", twinkle_meta, twinkle_npy, twinkle_json))

    lines.append("## Twinkle 12.mid")
    lines.append("")
    lines.extend(_metadata_section("Twinkle 12.mid metadata", mozart_meta, mozart_npy, mozart_json))

    lines.append("## Verification results")
    lines.append("")
    for label, passed in verification_checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"- [{status}] {label}")
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This is **Phase 1.5A only**: MIDI-to-chord-id extraction and verification. No SRN or EMA+MLP "
        "model has been run on these sequences yet. Raw chroma is not used as model input -- only the "
        "extracted chord ids (24-dim one-hot, same representation as the synthetic Phase 1 comparison) "
        "will be used in Phase 1.5B."
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_verification(twinkle_ids, twinkle_meta, mozart_ids, mozart_meta, twinkle_npy, twinkle_json, mozart_npy, mozart_json):
    checks = []

    checks.append(("Twinkle.mid duration is around 53.05s", abs(twinkle_meta["duration_sec"] - 53.05) < 1.0))
    checks.append(("Twinkle.mid has about 106 chroma windows at window_sec=0.5", abs(twinkle_meta["n_chroma_windows"] - 106) <= 2))
    checks.append(("Twinkle.mid chord_ids length is reasonable and nonzero", len(twinkle_ids) > 0))

    checks.append(("Twinkle 12.mid is much longer than Twinkle.mid", mozart_meta["duration_sec"] > 5 * twinkle_meta["duration_sec"]))

    all_ids = np.concatenate([twinkle_ids, mozart_ids])
    checks.append(("all chord ids are in 0..23", bool(np.all((all_ids >= 0) & (all_ids <= 23)))))

    checks.append(("Twinkle.mid first-15 decoded predictions available", len(twinkle_meta["first_15_decoded_chords"]) > 0))

    checks.append(("no NaNs in extracted chord ids", not np.isnan(all_ids.astype(np.float64)).any()))

    checks.append(("Twinkle.mid .npy exists", os.path.exists(twinkle_npy)))
    checks.append(("Twinkle.mid metadata .json exists", os.path.exists(twinkle_json)))
    checks.append(("Twinkle 12.mid .npy exists", os.path.exists(mozart_npy)))
    checks.append(("Twinkle 12.mid metadata .json exists", os.path.exists(mozart_json)))

    print()
    print("midi_chord_extraction.py verification")
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

    print("Extracting Twinkle.mid...")
    twinkle_ids, twinkle_meta, twinkle_npy, twinkle_json = extract_and_save("Twinkle.mid", "Twinkle_mid")
    print(f"  duration={twinkle_meta['duration_sec']:.2f}s windows={twinkle_meta['n_chroma_windows']} chord_ids={twinkle_meta['n_chord_ids']}")
    print("  First 15 decoded chords:", " | ".join(twinkle_meta["first_15_decoded_chords"]))

    print("\nExtracting Twinkle 12.mid...")
    mozart_ids, mozart_meta, mozart_npy, mozart_json = extract_and_save("Twinkle 12.mid", "Twinkle_12_mid")
    print(f"  duration={mozart_meta['duration_sec']:.2f}s windows={mozart_meta['n_chroma_windows']} chord_ids={mozart_meta['n_chord_ids']}")
    print("  First 15 decoded chords:", " | ".join(mozart_meta["first_15_decoded_chords"]))

    checks, all_passed = run_verification(twinkle_ids, twinkle_meta, mozart_ids, mozart_meta, twinkle_npy, twinkle_json, mozart_npy, mozart_json)

    report_md = build_report_md(twinkle_meta, twinkle_npy, twinkle_json, mozart_meta, mozart_npy, mozart_json, checks)
    with open(REPORT_MD, "w") as f:
        f.write(report_md)
    print(f"\nWrote {REPORT_MD}")


if __name__ == "__main__":
    main()
