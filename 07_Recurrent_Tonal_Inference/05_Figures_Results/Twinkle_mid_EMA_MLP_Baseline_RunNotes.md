# True Twinkle.mid Baseline — Run Notes (EMA + MLP pipeline)

## 1. Notebook used
`07_Recurrent_Tonal_Inference/02_Baseline_Pipeline/Mini Capstone_Project_A_walking_machine_of_the_music.ipynb`

(A pre-edit backup is preserved at `Mini Capstone_Project_A_walking_machine_of_the_music_BACKUP_before_true_twinkle_baseline.ipynb` in the same folder.)

## 2. MIDI file used
`07_Recurrent_Tonal_Inference/03_MIDI_Data/Twinkle.mid`

(Resolved at execution time via `PROJECT_PATH = "../03_MIDI_Data"`, i.e. `os.path.join(PROJECT_PATH, "Twinkle.mid")`.)

## 3. Duration of Twinkle.mid
53.05 seconds (`pretty_midi.PrettyMIDI(...).get_end_time()` = 53.052552s), 1 instrument, 84 notes total.

## 4. Number of chroma/time windows generated
106 windows (chroma matrix shape `(12, 106)` from `pm.get_chroma(fs=1/window_sec)`), yielding 106 extracted chords in `twinkle_chords`.

## 5. window_sec value
`0.5` seconds (passed to `midi_to_chord_ids(midi_file, window_sec=0.5)`).

## 6. alpha used in sequence_key_tracking
`alpha = 0.20` (leaky-integration context update rate, hand-coded — not learned).

## 7. Did the notebook retrain the MLP during nbconvert execution?
Yes. `ChordToKeyMLP` is instantiated fresh in cell 23 and there is no `torch.save`/`torch.load` anywhere in the notebook, so `train(model, epochs=12)` (cell 25) trains the model from scratch on a freshly-sampled 60,000-example synthetic diatonic-triad dataset every time the notebook runs. Final epoch of this run: train loss 1.7203, val loss 1.7167, val acc 0.287 (accuracy is expected to be low given chord/key ambiguity by design).

## 8. Are random seeds fixed?
Yes. Cell 4 sets `SEED = 269` and seeds `random.seed`, `np.random.seed`, and `torch.manual_seed` before any data sampling or model init, so the synthetic training set, weight initialization, and results are reproducible across runs (module import order and library versions held constant).

## 9. Top competing keys observed
By average P(key) across all 106 timesteps: **G major** (0.202), **F major** (0.168), **E minor** (0.155), C major (0.147), D minor (0.129). By per-timestep argmax: G major most likely 45.3% of steps, F major 39.6%, C major 12.3%, D minor 2.8%. G (dominant) and F (subdominant) are the dominant competitors to C, with their respective relative minors (E minor, D minor) trailing behind — consistent with the model's known I/IV/V ambiguity on diatonic triads.

## 10. Interpretation
The per-timestep argmax key is not overwhelmingly C major, but the Circle-of-Fifths walk visits only 4 adjacent positions the entire piece — F, C, G, D (i.e. -1 to +2 fifths from C) — starting and ending on F/C. This is a **tight, stable cluster immediately around the tonic**, not the wide, erratic jumps seen in the historical mislabeled Mozart run. This confirms Twinkle.mid is a legitimate simple-case, low-modulation baseline: the EMA/leaky-integration MLP pipeline keeps its key estimate confined to the tonic's near neighborhood throughout, making it a suitable known-good baseline to compare against a learned SRN hidden-state update in `04_Recurrent_Implementation/`.

## 11. Output figure paths
- `07_Recurrent_Tonal_Inference/05_Figures_Results/Twinkle_mid_EMA_MLP_Prob_Tracking.png`
- `07_Recurrent_Tonal_Inference/05_Figures_Results/Twinkle_mid_EMA_MLP_CircleOfFifths.png`
