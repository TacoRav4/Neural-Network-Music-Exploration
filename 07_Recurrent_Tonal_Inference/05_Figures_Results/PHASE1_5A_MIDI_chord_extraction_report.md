# Phase 1.5A — MIDI Chord-ID Extraction Report

Phase 1.5A extracts MIDI-derived chord-id sequences using the exact same representation path as the COGS 202 notebook: MIDI -> chroma windows -> chroma-level EMA -> triad template matching -> chord_ids. **This is extraction only.** No SRN or EMA+MLP MIDI evaluation has been run yet -- that is Phase 1.5B, not started here. Raw chroma is never used as model input anywhere in this workspace; only the resulting 24-dim one-hot chord/triad ids are (or will be) fed to either model.

## Extraction settings

- window_sec: 0.5
- memory_decay: 0.6
- Formula (unchanged from the notebook): `smoothed_chroma = memory_decay * smoothed_chroma + (1 - memory_decay) * window_chroma`; values below 10% of the smoothed window's max are thresholded to 0; chord id = argmax(`CHORD_TEMPLATES` . thresholded chroma).

## Twinkle.mid

### Twinkle.mid metadata

- MIDI path: `03_MIDI_Data/Twinkle.mid`
- Duration: 53.05 sec
- window_sec: 0.5
- memory_decay: 0.6
- Chroma windows: 106
- Chord ids extracted: 106
- Total notes: 84
- Instruments: 1
- Saved: `03_MIDI_Data/derived_chord_sequences/Twinkle_mid_chord_ids.npy`, `03_MIDI_Data/derived_chord_sequences/Twinkle_mid_chord_extraction_metadata.json`

Top chords (chord:quality -> count):

- C:maj: 57
- D:maj: 16
- A#:maj: 12
- F:maj: 9
- C#:maj: 6
- G:maj: 6

First 15 decoded chord predictions:

`C:maj | C:maj | C:maj | C:maj | C:maj | C:maj | D:maj | C:maj | C:maj | C:maj | C:maj | C#:maj | C:maj | C:maj | C:maj`

## Twinkle 12.mid

### Twinkle 12.mid metadata

- MIDI path: `03_MIDI_Data/Twinkle 12.mid`
- Duration: 687.47 sec
- window_sec: 0.5
- memory_decay: 0.6
- Chroma windows: 1374
- Chord ids extracted: 1271
- Total notes: 6294
- Instruments: 2
- Saved: `03_MIDI_Data/derived_chord_sequences/Twinkle_12_mid_chord_ids.npy`, `03_MIDI_Data/derived_chord_sequences/Twinkle_12_mid_chord_extraction_metadata.json`

Top chords (chord:quality -> count):

- C:maj: 852
- A#:maj: 161
- G:maj: 80
- C#:maj: 70
- F:maj: 34
- E:maj: 16
- E:min: 16
- D#:maj: 16

First 15 decoded chord predictions:

`C:maj | C:maj | C#:maj | C#:maj | C#:maj | C#:maj | C#:maj | C#:maj | C#:maj | C#:maj | C#:maj | C#:maj | C#:maj | C#:maj | C#:maj`

## Verification results

- [PASS] Twinkle.mid duration is around 53.05s
- [PASS] Twinkle.mid has about 106 chroma windows at window_sec=0.5
- [PASS] Twinkle.mid chord_ids length is reasonable and nonzero
- [PASS] Twinkle 12.mid is much longer than Twinkle.mid
- [PASS] all chord ids are in 0..23
- [PASS] Twinkle.mid first-15 decoded predictions available
- [PASS] no NaNs in extracted chord ids
- [PASS] Twinkle.mid .npy exists
- [PASS] Twinkle.mid metadata .json exists
- [PASS] Twinkle 12.mid .npy exists
- [PASS] Twinkle 12.mid metadata .json exists

## Scope note

This is **Phase 1.5A only**: MIDI-to-chord-id extraction and verification. No SRN or EMA+MLP model has been run on these sequences yet. Raw chroma is not used as model input -- only the extracted chord ids (24-dim one-hot, same representation as the synthetic Phase 1 comparison) will be used in Phase 1.5B.
