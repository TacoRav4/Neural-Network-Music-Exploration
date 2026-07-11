# Phase 1.5B — MIDI-Derived Chord-ID Evaluation Report

Evaluates the EMA+MLP baseline (alpha=0.20) and the best Elman SRN condition (epochs=25, lr=1e-3, hidden_size=48) on the MIDI-derived chord-id sequences extracted in Phase 1.5A (`midi_chord_extraction.py`). **This is a data-source change only** -- both models still consume the identical 24-dim one-hot chord/triad representation used throughout Phase 1; raw chroma is never fed to either model.

**This is not an accuracy test.** Real MIDI has no clean per-timestep key ground truth, so no synthetic-style accuracy is reported here. Instead, the metrics below describe each model's confidence, key diversity, and Circle-of-Fifths trajectory *stability* under real, noisy, template-matched chord sequences.

## Model settings

- EMA+MLP: alpha=0.2, MLP training seed=269, epochs=12, final val loss=1.7167, final val acc=0.2870
- SRN: epochs=25, lr=0.001, hidden_size=48, seed=269, final val loss=0.6622, final val acc=0.7800

## Twinkle.mid

- n_timesteps: 106
- Plotted keys (probability tracking): C maj, F maj, G maj, D min

### EMA+MLP

**EMA**
- n_timesteps: 106
- n_unique_predicted_keys: 4
- avg_prob_c_major: 0.1475
- mean_confidence (top-1 prob): 0.2711
- mean_entropy: 1.9089 (uniform-24 reference: 3.1781)
- fifths jump: mean=0.30, max=3.00, large jumps (>=3): 2 (0.0190 of transitions)
- Top predicted keys: G maj (45.3%), F maj (39.6%), C maj (12.3%), D min (2.8%)

### SRN

**SRN**
- n_timesteps: 106
- n_unique_predicted_keys: 3
- avg_prob_c_major: 0.0719
- mean_confidence (top-1 prob): 0.7089
- mean_entropy: 0.9444 (uniform-24 reference: 3.1781)
- fifths jump: mean=0.30, max=2.00, large jumps (>=3): 0 (0.0000 of transitions)
- Top predicted keys: F maj (67.0%), G maj (29.2%), A# min (3.8%)

## Twinkle 12.mid

- n_timesteps: 1271
- Plotted keys (probability tracking): C maj, F maj, G maj, D min

### EMA+MLP

**EMA**
- n_timesteps: 1271
- n_unique_predicted_keys: 13
- avg_prob_c_major: 0.1703
- mean_confidence (top-1 prob): 0.2431
- mean_entropy: 1.8356 (uniform-24 reference: 3.1781)
- fifths jump: mean=0.22, max=5.00, large jumps (>=3): 52 (0.0409 of transitions)
- Top predicted keys: F maj (61.4%), C maj (12.8%), A# maj (7.8%), G maj (6.3%), C# maj (3.8%)

### SRN

**SRN**
- n_timesteps: 1271
- n_unique_predicted_keys: 17
- avg_prob_c_major: 0.1435
- mean_confidence (top-1 prob): 0.7612
- mean_entropy: 0.8210 (uniform-24 reference: 3.1781)
- fifths jump: mean=0.18, max=6.00, large jumps (>=3): 30 (0.0236 of transitions)
- Top predicted keys: F maj (60.1%), C maj (12.1%), D# maj (9.8%), G maj (5.5%), C# maj (4.2%)

## Interpretation

**Twinkle.mid:** among each model's top-5 predicted keys, the fraction landing in the C/F/G/D-major neighborhood is 97.2% for EMA and 96.2% for the SRN -- both models stay clustered around the expected tonic neighborhood for this simple, low-modulation piece. Mean top-1 confidence on Twinkle.mid: EMA=0.271, SRN=0.709 -- the SRN is more confident on average here; combined with entropy (EMA=1.909, SRN=0.944, vs. a uniform-24 reference of 3.178), the SRN appears more overconfident rather than simply cleaner on this sequence. **Twinkle 12.mid:** mean Circle-of-Fifths jump distance is 0.22 for EMA vs. 0.18 for the SRN, and the fraction of large jumps (distance >= 3) is 0.0409 for EMA vs. 0.0236 for the SRN. The SRN reduces the 'spiderweb'-like instability described for this piece in the historical mislabeled-figure notes, on this specific run. Whether the SRN 'improves' Twinkle 12.mid depends on which axis is prioritized: if it reduces jump frequency/magnitude, that supports smoother tonal tracking; if predicted keys still concentrate heavily on a narrow set (see top predicted keys above) despite the piece's real embedded modulations, that is consistent with the same triadic-forcing / ornamentation-collapse limitation the COGS 202 paper describes for this piece, independent of which temporal-memory mechanism is used.

## Scope note

This is Phase 1.5B: real MIDI-derived chord-id sequences, still using the 24-dim one-hot chord/triad representation (no raw chroma as model input). **Raw-chroma Phase 2 has not started.** Any next step (a raw-chroma end-to-end SRN, or hidden-state PCA visualization) should be reviewed and approved separately before being implemented.
