# Phase 2C — MIDI Chroma Extraction Report

Phase 2C extracts and saves raw and smoothed 12-dim chroma sequences from `Twinkle.mid` and `Twinkle 12.mid`, for use by Phase 2's pitch-class/chroma experiments. **This is extraction only.** No pitch-class evaluation, Chroma SRN, or Phase 2D work has started here -- this module produces chroma arrays and metadata, nothing else.

## Extraction settings

- window_sec: 0.5
- memory_decay: 0.8 (matches `pitch_class_baseline.py`'s pitch-class-path smoothing constant)
- threshold_ratio: 0.1
- Formula: `smoothed_chroma_t = memory_decay * smoothed_chroma_{t-1} + (1 - memory_decay) * raw_chroma_t`; `thresholded_smoothed_chroma` zeroes values below `threshold_ratio * max` for each timestep independently.

## Twinkle.mid

### Twinkle.mid metadata

- MIDI path: `03_MIDI_Data/Twinkle.mid`
- Duration: 53.05 sec
- window_sec: 0.5
- memory_decay: 0.8
- threshold_ratio: 0.1
- raw_chroma_shape: (106, 12)
- smoothed_chroma_shape: (106, 12)
- thresholded_smoothed_chroma_shape: (106, 12)
- Total notes: 84
- Instruments: 1
- Mean chroma energy: 3.1792
- Nonzero window count: 101
- Saved: `03_MIDI_Data/derived_chroma_sequences/Twinkle_mid_raw_chroma.npy`, `03_MIDI_Data/derived_chroma_sequences/Twinkle_mid_smoothed_chroma_decay08.npy`, `03_MIDI_Data/derived_chroma_sequences/Twinkle_mid_thresholded_smoothed_chroma_decay08.npy`, `03_MIDI_Data/derived_chroma_sequences/Twinkle_mid_chroma_metadata.json`

Key signature changes: none present in this MIDI file.

Time signature changes:

- t=0.0s: 4/4

First 5 raw chroma windows (summary):

- t=0: nonzero_pcs=1, max_val=64.000
- t=1: nonzero_pcs=1, max_val=66.000
- t=2: nonzero_pcs=1, max_val=70.000
- t=3: nonzero_pcs=1, max_val=70.000
- t=4: nonzero_pcs=1, max_val=70.000

## Twinkle 12.mid

### Twinkle 12.mid metadata

- MIDI path: `03_MIDI_Data/Twinkle 12.mid`
- Duration: 687.47 sec
- window_sec: 0.5
- memory_decay: 0.8
- threshold_ratio: 0.1
- raw_chroma_shape: (1374, 12)
- smoothed_chroma_shape: (1374, 12)
- thresholded_smoothed_chroma_shape: (1374, 12)
- Total notes: 6294
- Instruments: 2
- Mean chroma energy: 2.1975
- Nonzero window count: 284
- Saved: `03_MIDI_Data/derived_chroma_sequences/Twinkle_12_mid_raw_chroma.npy`, `03_MIDI_Data/derived_chroma_sequences/Twinkle_12_mid_smoothed_chroma_decay08.npy`, `03_MIDI_Data/derived_chroma_sequences/Twinkle_12_mid_thresholded_smoothed_chroma_decay08.npy`, `03_MIDI_Data/derived_chroma_sequences/Twinkle_12_mid_chroma_metadata.json`

Key signature changes:

- t=0.0s: C Major (key_number=0)
- t=384.0s: Eb Major (key_number=3)
- t=392.0s: Eb Major (key_number=3)
- t=432.0s: C Major (key_number=0)
- t=440.0s: C Major (key_number=0)

Time signature changes:

- t=0.0s: 2/4
- t=8.0s: 2/4
- t=600.0s: 3/4
- t=612.0s: 3/4

First 5 raw chroma windows (summary):

- t=0: nonzero_pcs=0, max_val=0.000
- t=1: nonzero_pcs=0, max_val=0.000
- t=2: nonzero_pcs=0, max_val=0.000
- t=3: nonzero_pcs=0, max_val=0.000
- t=4: nonzero_pcs=0, max_val=0.000

## Verification results

- [PASS] Twinkle.mid raw_chroma exists and is non-empty
- [PASS] Twinkle.mid raw_chroma shape is (T, 12)
- [PASS] Twinkle.mid smoothed_chroma exists and is non-empty
- [PASS] Twinkle.mid smoothed_chroma shape is (T, 12)
- [PASS] Twinkle.mid thresholded_smoothed_chroma exists and is non-empty
- [PASS] Twinkle.mid thresholded_smoothed_chroma shape is (T, 12)
- [PASS] Twinkle 12.mid raw_chroma exists and is non-empty
- [PASS] Twinkle 12.mid raw_chroma shape is (T, 12)
- [PASS] Twinkle 12.mid smoothed_chroma exists and is non-empty
- [PASS] Twinkle 12.mid smoothed_chroma shape is (T, 12)
- [PASS] Twinkle 12.mid thresholded_smoothed_chroma exists and is non-empty
- [PASS] Twinkle 12.mid thresholded_smoothed_chroma shape is (T, 12)
- [PASS] Twinkle.mid has about 106 timesteps
- [PASS] Twinkle 12.mid has about 1374 timesteps
- [PASS] no NaNs in any chroma array
- [PASS] Twinkle.mid smoothed differs from raw
- [PASS] Twinkle 12.mid smoothed differs from raw
- [PASS] Twinkle.mid thresholded shape matches smoothed shape
- [PASS] Twinkle 12.mid thresholded shape matches smoothed shape
- [PASS] Twinkle 12.mid key signature events recorded
- [PASS] Twinkle.mid MIDI file not modified (mtime unchanged)
- [PASS] Twinkle 12.mid MIDI file not modified (mtime unchanged)
- [PASS] Twinkle.mid raw file exists
- [PASS] Twinkle.mid smoothed file exists
- [PASS] Twinkle.mid thresholded file exists
- [PASS] Twinkle.mid metadata file exists
- [PASS] Twinkle 12.mid raw file exists
- [PASS] Twinkle 12.mid smoothed file exists
- [PASS] Twinkle 12.mid thresholded file exists
- [PASS] Twinkle 12.mid metadata file exists

## Scope note

This is **Phase 2C only**: chroma extraction and verification. No pitch-class evaluation (Phase 2D), Chroma SRN (Phase 2E), or any other model/evaluation code has been run on these chroma sequences yet. Original MIDI files were not modified.
