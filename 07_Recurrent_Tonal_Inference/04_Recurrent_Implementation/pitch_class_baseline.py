"""pitch_class_baseline.py

Phase 2B: freezes the notebook's existing pitch-class / scale-template
baseline into a reusable standalone module. This is the same baseline
identified in ../05_Figures_Results/PHASE2_REPRESENTATION_PLAN.md section 3,
copied verbatim from
02_Baseline_Pipeline/Mini Capstone_Project_A_walking_machine_of_the_music.ipynb
(cells 40-41, "Pitch Class Baseline" -- "bypasses our neural network
entirely and compares the raw audio directly against 24 musical scales").

This baseline is fundamentally different from everything else in
04_Recurrent_Implementation/ up to this point:
- It uses the raw 12-dim chroma vector directly, not a 24-way chord id.
- It compares against full 7-note major/natural-minor SCALE templates, not
  triad templates -- there is no chord-template matching step at all.
- It has no MLP and no learned model of any kind -- key_id is a direct
  argmax(SCALE_TEMPLATES . chroma) each window.
- Its chroma-level smoothing constant is memory_decay=0.8 in the notebook,
  not the chord path's 0.6 -- this is an existing, already-baked-in
  difference being preserved here, not something introduced by this module.

This is Phase 2B only: modularization and verification of the existing
baseline. It does not save any derived chroma/key arrays (Phase 2C), does
not evaluate against Twinkle.mid/Twinkle 12.mid beyond this file's own
verification block (Phase 2D), and does not implement any Chroma SRN or
other learned model (Phase 2E). No MLP, no chord-template matching, no SRN
training happens anywhere in this file.
"""

import os
import sys

import numpy as np
import pretty_midi

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_MIDI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "03_MIDI_Data"))

from shared_music_defs import decode_key

DEFAULT_WINDOW_SEC = 0.5
DEFAULT_MEMORY_DECAY = 0.8  # matches the notebook's pitch-class baseline cell exactly (not the chord path's 0.6)


# ---------------------------------------------------------------------------
# SCALE_TEMPLATES
# (notebook cell 41)
#
# 24 full 7-note scale templates (12 major, 12 natural minor), one row per
# key, one column per pitch class -- NOT triads. This is what lets the
# baseline "bypass triadic forcing": a scale template accepts any of the 7
# diatonic pitch classes as evidence for its key, rather than requiring an
# exact 3-note chord match.
# ---------------------------------------------------------------------------

def build_scale_templates():
    """Builds the 24x12 scale-template matrix exactly as the notebook does."""
    templates = np.zeros((24, 12))
    for root in range(12):
        # Major scale intervals
        maj_intervals = [0, 2, 4, 5, 7, 9, 11]
        for i in maj_intervals:
            templates[root, (root + i) % 12] = 1

        # Natural minor scale intervals
        min_intervals = [0, 2, 3, 5, 7, 8, 10]
        for i in min_intervals:
            templates[root + 12, (root + i) % 12] = 1
    return templates


SCALE_TEMPLATES = build_scale_templates()


# ---------------------------------------------------------------------------
# midi_to_key_baseline
# (notebook cell 41: "Building the Pitch-Class Bridge")
#
# Formula copied as-is -- do not change:
#   smoothed_chroma = memory_decay * smoothed_chroma + (1 - memory_decay) * window_chroma
#   threshold values below 10% of the smoothed window's max to 0
#   key_id = argmax(SCALE_TEMPLATES . working_chroma)
# No MLP, no chord-template matching, no SRN anywhere in this function.
# ---------------------------------------------------------------------------

def midi_to_key_baseline(midi_path, window_sec=DEFAULT_WINDOW_SEC, memory_decay=DEFAULT_MEMORY_DECAY, return_metadata=True):
    """
    Returns:
      key_ids: (N,) numpy int64 array of key indices (0..23), N <= number of
        chroma windows (fewer only if the very first window(s) are silent
        and there is no previous key yet to repeat -- matches notebook
        behavior).
      metadata: dict (only if return_metadata=True).
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    chroma = pm.get_chroma(fs=(1.0 / window_sec))

    key_sequence = []
    smoothed_chroma = np.zeros(12)

    for t in range(chroma.shape[1]):
        window_chroma = chroma[:, t].copy()
        smoothed_chroma = memory_decay * smoothed_chroma + (1 - memory_decay) * window_chroma

        working_chroma = smoothed_chroma.copy()

        # Noise threshold
        if np.max(working_chroma) > 0.0:
            threshold = 0.1 * np.max(working_chroma)
            working_chroma[working_chroma < threshold] = 0

        # Handling silent windows
        if np.sum(working_chroma) == 0:
            if len(key_sequence) > 0:
                key_sequence.append(key_sequence[-1])
            continue

        # Compare raw (smoothed) chroma directly to the 24 scales
        scores = SCALE_TEMPLATES.dot(working_chroma)
        key_id = int(np.argmax(scores))
        key_sequence.append(key_id)

    key_ids = np.array(key_sequence, dtype=np.int64)

    if not return_metadata:
        return key_ids

    total_notes = sum(len(inst.notes) for inst in pm.instruments)

    counts = {}
    for k in key_ids.tolist():
        name, mode = decode_key(k)
        label = f"{name} {mode}"
        counts[label] = counts.get(label, 0) + 1
    top_keys = sorted(counts.items(), key=lambda kv: -kv[1])[:8]

    first_15_decoded = [f"{decode_key(k)[0]} {decode_key(k)[1]}" for k in key_ids[:15].tolist()]

    metadata = {
        "midi_path": os.path.relpath(midi_path, os.path.join(_THIS_DIR, "..")),
        "duration_sec": float(pm.get_end_time()),
        "window_sec": window_sec,
        "memory_decay": memory_decay,
        "n_chroma_windows": int(chroma.shape[1]),
        "n_key_predictions": int(len(key_ids)),
        "total_notes": int(total_notes),
        "n_instruments": len(pm.instruments),
        "first_15_decoded_keys": first_15_decoded,
        "top_predicted_keys": top_keys,
    }

    return key_ids, metadata


# ---------------------------------------------------------------------------
# Verification block
#
# Sanity-checks that this module reproduces the notebook's pitch-class
# baseline behavior on both existing MIDI files. Not a derived-data save
# script (Phase 2C), not an evaluation/report script (Phase 2D), and not a
# Chroma SRN (Phase 2E) -- run this file directly
# (`python pitch_class_baseline.py`) to re-check the freeze at any time.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    checks = []

    checks.append(("SCALE_TEMPLATES shape == (24, 12)", SCALE_TEMPLATES.shape == (24, 12)))

    twinkle_path = os.path.join(_MIDI_DIR, "Twinkle.mid")
    twinkle12_path = os.path.join(_MIDI_DIR, "Twinkle 12.mid")

    print("Running pitch-class baseline on Twinkle.mid...")
    twinkle_keys, twinkle_meta = midi_to_key_baseline(twinkle_path)
    print(f"  duration={twinkle_meta['duration_sec']:.2f}s windows={twinkle_meta['n_chroma_windows']} predictions={twinkle_meta['n_key_predictions']}")
    print("  First 15 decoded keys:", " | ".join(twinkle_meta["first_15_decoded_keys"]))

    print("\nRunning pitch-class baseline on Twinkle 12.mid...")
    twinkle12_keys, twinkle12_meta = midi_to_key_baseline(twinkle12_path)
    print(f"  duration={twinkle12_meta['duration_sec']:.2f}s windows={twinkle12_meta['n_chroma_windows']} predictions={twinkle12_meta['n_key_predictions']}")
    print("  First 15 decoded keys:", " | ".join(twinkle12_meta["first_15_decoded_keys"]))

    checks.append(("Twinkle.mid duration is around 53.05s", abs(twinkle_meta["duration_sec"] - 53.05) < 1.0))
    checks.append(("Twinkle.mid produces about 106 predictions", abs(twinkle_meta["n_key_predictions"] - 106) <= 2))
    checks.append(("Twinkle 12.mid is much longer than Twinkle.mid", twinkle12_meta["duration_sec"] > 5 * twinkle_meta["duration_sec"]))

    all_ids = np.concatenate([twinkle_keys, twinkle12_keys])
    checks.append(("all key ids are in 0..23", bool(np.all((all_ids >= 0) & (all_ids <= 23)))))

    checks.append(("Twinkle.mid first-15 decoded predictions available", len(twinkle_meta["first_15_decoded_keys"]) > 0))
    checks.append(("Twinkle 12.mid first-15 decoded predictions available", len(twinkle12_meta["first_15_decoded_keys"]) > 0))

    checks.append(("no NaNs in key ids", not np.isnan(all_ids.astype(np.float64)).any()))

    # "Output is stable and non-empty": re-running produces identical
    # results (this baseline is fully deterministic -- no randomness
    # anywhere in the pipeline) and both outputs are non-empty.
    twinkle_keys_rerun = midi_to_key_baseline(twinkle_path, return_metadata=False)
    checks.append(("Twinkle.mid output is non-empty", len(twinkle_keys) > 0))
    checks.append(("Twinkle.mid output is stable across re-runs (deterministic)", np.array_equal(twinkle_keys, twinkle_keys_rerun)))
    checks.append(("Twinkle 12.mid output is non-empty", len(twinkle12_keys) > 0))

    print()
    print("pitch_class_baseline.py verification")
    print("-" * 50)
    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")
    print("-" * 50)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")

    print()
    print("NOTE: this is Phase 2B (modularization + verification) only.")
    print("No derived .npy files, plots, or Phase 2 evaluation reports were created.")
    print("No Chroma SRN or other learned model was trained.")
