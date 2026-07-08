# Phase 2D — Pitch-Class Baseline Evaluation Report

Evaluates the modularized pitch-class/scale-template baseline (`pitch_class_baseline.py`, Phase 2B) on `Twinkle.mid` and `Twinkle 12.mid`, cross-referenced against the saved chroma sequences from Phase 2C. **This is a non-neural baseline evaluation** -- it bypasses the MLP, chord-id template matching, EMA+MLP, SRN, Chroma SRN, and Transformer entirely. Predictions come directly from `midi_to_key_baseline` (raw/smoothed chroma -> `SCALE_TEMPLATES` -> argmax key id).

**No Chroma SRN, Transformer, or QuickBin-like staged pipeline has been implemented anywhere in this workspace.** This report is representation-only evidence (does avoiding hard triadic forcing change real-MIDI behavior), not a new model result.

## Settings

- window_sec: 0.5
- memory_decay: 0.8 (pitch-class baseline's own smoothing constant, per Phase 2B)

## Twinkle.mid

- n_timesteps: 106
- n_unique_predicted_keys: 1
- proportion_c_major: 1.0000
- fifths jump: mean=0.00, max=0.00, large jumps (>=3): 0 (0.0000 of transitions)
- Top predicted keys: C maj (100.0%)

**Vs. Phase 1.5B (chord-id EMA/SRN):** Phase 1.5B found the SRN collapsed onto F major (67.0% of predictions) with avg_prob_c_major=0.072, and EMA at F major 39.6% with avg_prob_c_major=0.147. Here, the pitch-class baseline predicts C major 1.000 of the time -- C major is the single most common prediction, in clear contrast with both chord-id models' F-major dominance in Phase 1.5B.

## Twinkle 12.mid

### Overall

- n_timesteps: 1271
- n_unique_predicted_keys: 4
- proportion_c_major: 0.9087
- fifths jump: mean=0.02, max=3.00, large jumps (>=3): 5 (0.0039 of transitions)
- Top predicted keys: C maj (90.9%), D# maj (7.1%), G maj (1.9%), A# maj (0.2%)

**Vs. Phase 1.5B:** both chord-id EMA and SRN concentrated ~60% of predictions on F major and only ~12-13% on C major despite this piece's real embedded modulations. Here, the pitch-class baseline predicts C major 0.909 of the time overall.

### Key-signature-aligned windows

Twinkle 12.mid's real embedded key-signature events (confirmed in Phase 2C via `pretty_midi`, read-only): t=0.0s C Major, t=384.0s/392.0s Eb Major, t=432.0s/440.0s C Major. The windows below are split at t=384s and t=432s accordingly, corrected for the leading-silent-window time offset (see Settings/JSON `*_offset_windows`) so window boundaries align with true wall-clock time rather than with `key_ids`' own index 0. **This is a descriptive, key-signature-aligned check, not a dense per-timestep accuracy computation** -- there is no per-timestep ground truth. Note: this workspace's `decode_key` spells pitch class 3 as "D#", so "D# maj" below is the same key as "Eb Major".

**pre_384s (expected: C Major):**
- timesteps: 665 (indices 0-665)
- proportion predicted as expected key (C Major): 1.0000
- Top predicted keys: C maj (100.0%)

**384_to_432s (expected: Eb Major):**
- timesteps: 96 (indices 665-761)
- proportion predicted as expected key (Eb Major): 0.8958
- Top predicted keys: D# maj (89.6%), C maj (10.4%)

**post_432s (expected: C Major):**
- timesteps: 510 (indices 761-1271)
- proportion predicted as expected key (C Major): 0.9412
- Top predicted keys: C maj (94.1%), G maj (4.7%), D# maj (0.8%), A# maj (0.4%)

## Interpretation

**Does the pitch-class baseline reduce the F-major bias seen in Phase 1.5B?** On Twinkle.mid, yes, clearly: the pitch-class baseline's top predicted key is C maj at 100.0%, with proportion_c_major=1.000 -- markedly higher than either chord-id model's avg_prob_c_major in Phase 1.5B (EMA 0.147, SRN 0.072), and the baseline's most common prediction is C major rather than F major. **Does Twinkle 12.mid respond to the C -> Eb -> C key-signature markers?** In the pre_384s window (expected C Major), the baseline predicts C major 1.0000 of the time. In the 384_to_432s window (expected Eb Major), it predicts Eb major ("D# maj") 0.8958 of the time. In the post_432s window (expected C Major again), it predicts C major 0.9412 of the time. Put together, **the baseline responds strongly and correctly to the real modulation**: it predicts Eb major (reported as "D# maj" above -- the same key, spelled with a sharp rather than a flat by this workspace's `decode_key` convention) for the large majority of this window, closely tracking the piece's real, embedded key-signature change rather than defaulting to C major. **Framing:** all of the above is representation-only evidence -- a non-neural, hand-coded scale-template comparison, not a new learned model result. It suggests the *representation* (full scale evidence vs. hard triads) materially changes Twinkle.mid's behavior, but does not by itself establish that a learned model over this representation (a Chroma SRN) would perform similarly, worse, or better -- that remains untested. No Chroma SRN, Transformer, or QuickBin-like staged pipeline has been implemented yet.

## Scope note

This is Phase 2D: a non-neural, representation-only baseline evaluation. No Chroma SRN (Phase 2E), Transformer, or QuickBin-like staged tonal-inference pipeline has been implemented here. Metrics are descriptive (proportions, jump statistics, key-signature-aligned window checks), not accuracy against dense ground truth, since no such ground truth exists for real MIDI.
