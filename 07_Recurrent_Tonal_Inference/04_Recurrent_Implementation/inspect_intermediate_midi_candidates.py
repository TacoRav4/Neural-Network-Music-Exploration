"""inspect_intermediate_midi_candidates.py

Phase 3F: QA inspection of candidate intermediate-difficulty MIDI files
(per PHASE3E_INTERMEDIATE_MIDI_CORPUS_SELECTION_PLAN.md). This script only
loads each candidate with pretty_midi and reports metadata/QA flags -- it
does NOT run any tonal-inference extraction or evaluation (no Phase 2C
chroma extraction, no Phase 2D pitch-class baseline, no Phase 3B
diagnostics). That is Phase 3G, not this script.

Does not modify any existing implementation script, does not touch
Twinkle.mid / Twinkle 12.mid, and does not train or run any model.
"""

import json
import os
import sys

import pretty_midi

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MIDI_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "03_MIDI_Data"))
_CANDIDATE_DIR = os.path.join(_MIDI_DIR, "candidate_intermediate_midi")

# --- QA thresholds (documented, not tuned) ---
LONG_FILE_THRESHOLD_SEC = 300.0        # > 5 minutes flagged "very long"
DENSE_FILE_NOTES_PER_SEC = 8.0         # > 8 notes/sec flagged "very dense"
MANY_INSTRUMENTS_THRESHOLD = 3         # > 3 instruments flagged


def inspect_midi_file(path):
    """Returns a QA dict for one MIDI file. Never raises -- load failures
    are captured as a 'fail' entry instead."""
    filename = os.path.basename(path)
    entry = {"filename": filename, "path": os.path.relpath(path, os.path.join(_MIDI_DIR, "..")), "load_ok": False}

    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception as e:
        entry["load_ok"] = False
        entry["load_error"] = f"{type(e).__name__}: {e}"
        entry["qa_recommendation"] = "fail"
        entry["warnings"] = ["suspicious_or_corrupted_load_failure"]
        return entry

    entry["load_ok"] = True
    duration_sec = float(pm.get_end_time())
    n_instruments = len(pm.instruments)
    instruments = [
        {"index": i, "name": inst.name or None, "program": int(inst.program), "is_drum": bool(inst.is_drum), "n_notes": len(inst.notes)}
        for i, inst in enumerate(pm.instruments)
    ]
    total_notes = sum(len(inst.notes) for inst in pm.instruments)
    notes_per_sec = (total_notes / duration_sec) if duration_sec > 0 else None

    key_signature_changes = [
        {"time": float(ks.time), "key_number": int(ks.key_number), "key_name": pretty_midi.key_number_to_key_name(ks.key_number)}
        for ks in pm.key_signature_changes
    ]
    time_signature_changes = [
        {"time": float(ts.time), "numerator": int(ts.numerator), "denominator": int(ts.denominator)}
        for ts in pm.time_signature_changes
    ]

    try:
        tempo_times, tempo_values = pm.get_tempo_changes()
        tempo_changes = [{"time": float(t), "bpm": float(b)} for t, b in zip(tempo_times, tempo_values)]
    except Exception:
        tempo_changes = []

    # Approximate max simultaneous notes (polyphony): sweep note on/off events.
    events = []
    for inst in pm.instruments:
        for note in inst.notes:
            events.append((note.start, 1))
            events.append((note.end, -1))
    events.sort(key=lambda e: (e[0], -e[1]))  # process note-ons before note-offs at the same instant
    current = 0
    max_polyphony = 0
    for _, delta in events:
        current += delta
        max_polyphony = max(max_polyphony, current)

    warnings = []
    if duration_sec > LONG_FILE_THRESHOLD_SEC:
        warnings.append(f"very_long_file (>{LONG_FILE_THRESHOLD_SEC:.0f}s)")
    if notes_per_sec is not None and notes_per_sec > DENSE_FILE_NOTES_PER_SEC:
        warnings.append(f"very_dense_file (>{DENSE_FILE_NOTES_PER_SEC:.0f} notes/sec)")
    if len(key_signature_changes) == 0:
        warnings.append("no_key_signature_metadata")
    if n_instruments > MANY_INSTRUMENTS_THRESHOLD:
        warnings.append(f"many_instruments (>{MANY_INSTRUMENTS_THRESHOLD})")

    if warnings:
        qa_recommendation = "warn"
    else:
        qa_recommendation = "pass"

    entry.update({
        "duration_sec": duration_sec,
        "n_instruments": n_instruments,
        "instruments": instruments,
        "total_notes": total_notes,
        "notes_per_sec": notes_per_sec,
        "max_polyphony_approx": max_polyphony,
        "key_signature_changes": key_signature_changes,
        "time_signature_changes": time_signature_changes,
        "tempo_changes": tempo_changes,
        "warnings": warnings,
        "qa_recommendation": qa_recommendation,
    })
    return entry


def main(midi_paths):
    results = []
    for path in midi_paths:
        print(f"Inspecting {os.path.basename(path)} ...")
        entry = inspect_midi_file(path)
        status = entry["qa_recommendation"]
        print(f"  -> {status.upper()}" + (f" ({', '.join(entry.get('warnings', []))})" if entry.get("warnings") else ""))
        results.append(entry)
    return results


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        if os.path.isdir(_CANDIDATE_DIR):
            paths = sorted(
                os.path.join(_CANDIDATE_DIR, f) for f in os.listdir(_CANDIDATE_DIR) if f.lower().endswith(".mid")
            )
        else:
            print(f"No candidate directory at {_CANDIDATE_DIR} and no paths given.")
            sys.exit(1)

    results = main(paths)
    print(json.dumps(results, indent=2))
