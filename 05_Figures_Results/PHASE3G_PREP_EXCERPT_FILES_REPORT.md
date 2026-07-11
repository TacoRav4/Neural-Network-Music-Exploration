# Phase 3G-Prep — Confirmed Excerpt MIDI Files

**Excerpt creation and verification only. No tonal-inference evaluation (chroma extraction, pitch-class baseline, chord-id EMA/SRN, uncertainty diagnostics, or disagreement analysis) has been run on either excerpt. No existing scripts, notebooks, or Phase 1/1.5/2/3 outputs have been modified. Original full candidate MIDI files are unchanged.**

Creates the two excerpt MIDI files corresponding to the boundaries the user confirmed via listening in Phase 3F.8, so Phase 3G can evaluate them directly rather than the full pieces.

## Method

Both excerpts were cropped using `pretty_midi.PrettyMIDI.crop(0.0, end_time)`, which correctly rebuilds the tempo map, time-signature, and key-signature event lists for the cropped range. `crop()`'s own note-filtering rule drops any note whose end time exceeds the crop boundary — since the task requires sustained notes to be **clipped cleanly, not dropped**, each file's original (uncropped) note list was checked for notes that start before the boundary but sustain past it; any such note was re-added to the cropped output with its `end` truncated to exactly the boundary time. Both files needed exactly one such correction.

## Für Elise opening excerpt

- **Source file:** `03_MIDI_Data/candidate_intermediate_midi/for_elise_by_beethoven.mid`
- **Excerpt file:** `03_MIDI_Data/candidate_intermediate_midi/excerpts/Fur_Elise_opening_0_54s.mid`
- **Confirmed boundary:** `[0.0, 54.0]` seconds (user-confirmed via listening; independently corroborated in Phase 3F.8 by pitch-class metadata — F-natural absent 42-54s, present in every window from 54s onward)

| Metric | Value |
|---|---|
| Duration | 53.99995s (≈54.0s; sub-millisecond rounding from MIDI tick quantization) |
| Instruments | 1 ("Für Elise") — preserved |
| Notes (excerpt / original) | 278 / 1068 |
| Notes/sec | 5.15 |
| Max polyphony (approx.) | 3 |
| Boundary-crossing notes clipped | 1 (a note originally ending at t≈54.00001s, truncated to end exactly at 54.0s — negligible, ~9 microseconds) |
| Tempo changes preserved | 1 event, 70.0 bpm flat (unchanged from source) |
| Time signature changes preserved | 1 event, 4/4 at t=0.0s |
| Key signature changes preserved | none (source file has none — not invented) |

## Clementi exposition (no repeat) excerpt

- **Source file:** `03_MIDI_Data/candidate_intermediate_midi/clementi_opus36_1_1.mid`
- **Excerpt file:** `03_MIDI_Data/candidate_intermediate_midi/excerpts/Clementi_Op36_No1_I_exposition_norepeat_0_17p3s.mid`
- **Confirmed boundary:** `[0.0, 17.3]` seconds (user-confirmed via listening — "a repeat of the first introduction" at ~17s; matches the Phase 3F.7 onset-gap/harmonic-repeat metadata evidence exactly)

| Metric | Value |
|---|---|
| Duration | 17.29999s (≈17.3s; sub-millisecond rounding from MIDI tick quantization) |
| Instruments | 2 ("Piano right", "Piano left") — both preserved |
| Notes (excerpt / original) | 135 / 666 |
| Notes/sec | 7.80 |
| Max polyphony (approx.) | 4 |
| Boundary-crossing notes clipped | 1 (a note originally spanning t≈17.258-17.582s, truncated to end at 17.3s) |
| Tempo changes preserved | 161 events (of 686 in the source — this piece is an expressive/humanized performance MIDI with continuous rubato; all events within `[0.0, 17.3]` were retained), bpm range 173.9-210.8 within this window |
| Time signature changes preserved | 1 event, 4/4 at t=0.0s |
| Key signature changes preserved | 1 event, C Major at t=0.0s (the source's only embedded key signature) |

## Verification checks

| Check | Für Elise | Clementi |
|---|---|---|
| Excerpt loads with `pretty_midi` | ✅ pass | ✅ pass |
| Duration ≈ target | ✅ 53.99995s ≈ 54.0s | ✅ 17.29999s ≈ 17.3s |
| No notes start before t=0 | ✅ pass (min start = 0.857s / 0.0s respectively) | ✅ pass |
| No notes extend beyond excerpt duration | ✅ pass (max end = duration, both files) | ✅ pass |
| Sustained boundary-crossing notes clipped, not dropped | ✅ 1 note clipped | ✅ 1 note clipped |
| Original full MIDI files unchanged | ✅ confirmed via mtime — `for_elise_by_beethoven.mid` and `clementi_opus36_1_1.mid` untouched | |

## Reminder: Chopin Op. 28 No. 4 silence

Not part of this task's scope, but restated per the task's instruction: the Chopin Prelude in E minor, Op. 28 No. 4 (Level 4) contains a genuine, compositionally-intentional 4.29-second silence at t≈95.4-99.6s (Chopin's well-known dramatic pause near the piece's end). This is **not** a data or file-corruption issue. When Phase 3G eventually processes this piece, an "inactive"/undefined-key stretch at that timestamp reflects real musical silence, not a pipeline failure — already documented in the registry's `structural_ambiguity_note` for L4.

## Scope note

**No tonal evaluation has started.** No chroma extraction, pitch-class baseline run, chord-id EMA/SRN run, uncertainty diagnostics, or disagreement analysis has been performed on either excerpt file. This is excerpt creation and verification only — Phase 3G (running the existing, unmodified Phase 2C/2D/3B pipeline) is the next step, now unblocked for Level 3 and Level 5.
