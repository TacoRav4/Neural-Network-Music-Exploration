# Phase 2 Representation Plan

**Planning document only. No Phase 2 code, models, or evaluations have been implemented. No existing scripts, notebooks, or Phase 1/1.5 outputs have been modified.**

## 1. Phase 2 research question

Phase 1 and Phase 1.5B both held the input representation fixed (24-dim one-hot chord/triad) and varied only the temporal-memory mechanism (hand-coded EMA vs. learned SRN). Phase 2 flips that: **can preserving pitch-class/chroma evidence directly, instead of forcing each analysis window into a hard 24-way major/minor triad, improve real-MIDI tonal inference?**

This is deliberately a representation question, not a recurrence question. Phase 2 should isolate "does the input representation matter" the same way Phase 1 isolated "does the temporal mechanism matter" — one variable at a time.

## 2. Existing evidence motivating Phase 2

From `PHASE1_SYNTHETIC_EMA_vs_SRN_SUMMARY.md` and `PHASE1_5B_MIDI_EMA_vs_SRN_SUMMARY.md`:

- **The SRN helps on clean synthetic chord data.** On synthetic modulation sequences built from clean diatonic triads, the SRN beat the EMA+MLP baseline on both overall accuracy (0.8101 vs. 0.7529) and modulation accuracy (0.8010 vs. 0.7485).
- **The SRN cannot rescue distorted, real MIDI-derived triadic inputs.** On both `Twinkle.mid` and `Twinkle 12.mid`, the SRN became more confident (lower entropy, higher top-1 probability) than EMA, but did not recover a more correct tonic — if anything it collapsed harder onto the wrong key.
- **Twinkle.mid and Twinkle 12.mid both show F-major bias despite C-major evidence.** Per Phase 1.5A's chord-extraction histograms, C major is the single most common *extracted chord* for both pieces (57/106 windows for Twinkle.mid; 852/1271 for Twinkle 12.mid). Yet both models — EMA and SRN alike — end up predicting F major as the dominant *key* roughly 40-67% of the time, and assign C major noticeably less average probability (SRN: 0.072 on Twinkle.mid, vs. EMA's 0.147).
- **Triadic forcing is the likely bottleneck, not the temporal mechanism.** Since the bias is present in both temporal-memory conditions and the correct tonal evidence (C major chords) is present at the chord-extraction level, the distortion most plausibly enters at the hard 24-way triad-matching step and/or the resulting chord-to-key mapping the MLP was trained on — a representational issue, not a memory issue.

## 3. Existing pitch-class baseline (already in the notebook)

Inspected directly in `02_Baseline_Pipeline/Mini Capstone_Project_A_walking_machine_of_the_music.ipynb`, cells 40-41 ("Pitch Class Baseline" — "bypasses our neural network entirely and compares the raw audio directly against 24 musical scales"):

- **Function name:** `midi_to_key_baseline(midi_path, window_sec=0.5)`.
- **Input representation:** the raw 12-dim chroma vector per window, smoothed with the same style of leaky/EMA blend used elsewhere (`smoothed_chroma = memory_decay * smoothed_chroma + (1 - memory_decay) * window_chroma`) — **but with `memory_decay=0.8`**, not the chord path's `0.6`. This is a different smoothing constant already baked into the existing notebook code, worth noting explicitly since Phase 2 should not silently inherit it without deciding whether to keep it.
- **Uses chroma directly:** yes — there is no chord-template intermediate step. Chroma is compared directly against `SCALE_TEMPLATES` (a 24×12 matrix of full 7-note major/natural-minor scales, not triads).
- **Bypasses the MLP and triadic forcing:** yes, completely. `key_id = argmax(SCALE_TEMPLATES . working_chroma)` is a direct, non-neural, non-triadic key estimate.
- **Current output:** a plain Python list of key ids (`key_sequence`), and the notebook only prints the first 15 decoded key names (`decode_key`) — no saved arrays, no metadata, no plots, no metrics of any kind currently exist for this baseline anywhere in the workspace.
- **Modularizable later:** yes, cleanly. The function is already self-contained (only depends on `pretty_midi` and a local `SCALE_TEMPLATES` matrix); it can be frozen into `04_Recurrent_Implementation/` the same way `shared_music_defs.py` froze the chord/key vocabulary, with `SCALE_TEMPLATES` and `midi_to_key_baseline` copied verbatim (per the same "freeze, don't reinterpret" convention used throughout this workspace).

## 4. Phase 2 candidate models

### Option A: Pitch-class scale-template baseline only

- **Goal:** establish a non-neural, direct-chroma reference point for Twinkle.mid/Twinkle 12.mid, analogous to what Phase 1.5B did for the chord-id path, before any new model is trained.
- **Input shape:** 12-dim chroma vector per window (`window_sec=0.5`, matching the existing extraction cadence).
- **Output shape:** a single key id per window (24-way argmax), i.e. `(T,)` — no probability distribution currently, since `midi_to_key_baseline` only returns argmax ids, not softmax scores (`scores = SCALE_TEMPLATES.dot(working_chroma)` are raw dot-product scores, not normalized probabilities).
- **Training data needed:** none — this is a non-learned baseline, template-matching only.
- **Strengths:** already implemented and known to work in the notebook; zero training cost; a clean, minimal-assumption reference for "how much does simply avoiding triadic forcing already help," independent of any new model risk.
- **Risks:** raw dot-product scores aren't directly comparable to the EMA/SRN's softmax probabilities (no confidence/entropy metrics without an added normalization step, e.g. softmax over the scores); scale templates conflate all diatonic scale tones equally, which may itself be too coarse in a different way from triads (e.g. can't distinguish tonic from other scale members without additional weighting).
- **Sequencing:** **first step.** Cheapest, safest, and directly informative — should run before any new model is built.

### Option B: Chroma SRN

- **Goal:** test whether a *learned* recurrent hidden state, given the same un-triaded chroma evidence Option A uses, can track key better than either the non-neural scale-template baseline or the existing chord-id SRN.
- **Input shape:** `(T, 12)` chroma (or smoothed chroma) per sequence — same per-timestep dimensionality as the existing pitch-class baseline's input, but fed to a learned model instead of a fixed template.
- **Output shape:** `(T, 24)` key logits/probabilities — same output convention as `ElmanKeySRN`, so evaluation code (entropy, confidence, Circle-of-Fifths jump stats) can be reused with minimal change.
- **Training data needed:** new synthetic sequences that are chroma/pitch-class level, not chord-id level (see section 5) — the existing `sequence_dataset.py` generates one-hot *chord* ids, which is the wrong shape and the wrong abstraction level for this input.
- **Strengths:** directly tests the Phase 2 research question (does representation, not just recurrence, matter) with an apples-to-apples architecture change from `ElmanKeySRN`; reuses most of `srn_model.py`'s training-loop machinery conceptually (though not its 24-dim input assumption).
- **Risks:** meaningfully more implementation work than Option A (new dataset generator, new/adapted model, new training run); a synthetic chroma-level training distribution that doesn't resemble real melodic/ornamented MIDI could reproduce the same "helps on synthetic, fails on real MIDI" pattern seen in Phase 1 vs. 1.5B, just one level down — the training-data realism problem does not disappear by changing representation alone.
- **Sequencing:** **later**, only after Option A's results are in hand and reviewed — Option A tells us whether representation alone (without any new learned model) already changes the real-MIDI picture, which should inform whether Option B is worth building at all.

### Option C: Hybrid representation (chord-id + chroma, or pitch-class + chord evidence)

- **Goal:** combine the categorical chord-identity signal (useful for harmonic function, e.g. distinguishing I vs. V of the same key) with the continuous chroma signal (useful for not discarding passing tones/ornamentation) in one input.
- **Input shape:** not yet fixed — candidates include concatenating a 24-dim chord one-hot with a 12-dim chroma vector (`(T, 36)`), or using chroma as the primary input with chord id as an auxiliary/side signal.
- **Output shape:** `(T, 24)` key probabilities, consistent with everything else in this workspace.
- **Training data needed:** synthetic sequences that carry both a valid chord label and pitch-class content simultaneously — more design work than either A or B alone.
- **Strengths:** could combine the best of both signals; a natural target once both A and B have standalone results to compare against.
- **Risks:** conflates two representational changes at once, which violates this workspace's own established discipline (Phase 1's guardrail was "don't mix representation and recurrence changes"; Option C would mix two different representations together without first knowing how either behaves alone) — premature without A and B as individual baselines first.
- **Sequencing:** **explicitly a later extension, not a first Phase 2 implementation step.** Should not be attempted before Option A and Option B each have independent, reviewed results.

## 5. Training data design for a raw-chroma/pitch-class SRN (for Option B, when it is approved)

The existing `sequence_dataset.py` generates sequences of discrete chord ids sampled from clean diatonic triad sets — appropriate for the chord-id SRN, but not melody-like enough to train a chroma-level model realistically. A future chroma-level dataset generator should include:

- **Monophonic scale-degree sequences:** single pitch classes drawn from a key's scale (not full triads), to mimic melodic lines like Twinkle Twinkle's actual monophonic tune, rather than always presenting full chords.
- **Chordal windows:** windows where multiple simultaneous scale/chord tones are active (as chroma naturally aggregates), to also cover harmonically dense passages.
- **Passing tones / ornamentation:** deliberately inject occasional out-of-key or non-harmonic tones into otherwise in-key windows, to simulate the kind of "ornamentation collapse" the COGS 202 paper describes for Twinkle 12.mid (Mozart's variations), rather than only ever presenting clean in-key evidence.
- **Modulation sequences:** carried over conceptually from `sequence_dataset.make_modulation_sequence` — a key_a segment followed by a key_b segment — but generating pitch-class/chroma windows instead of chord ids.
- **No-modulation controls:** carried over from `make_no_modulation_sequence`, same rationale (a negative control so the model doesn't learn to always expect a pivot).
- **Per-timestep key labels:** same hard-label convention as `sequence_dataset.py` (pre-pivot = key_a, post-pivot = key_b), for consistency and so existing accuracy/lag evaluation code can be reused with minimal adaptation.
- **Ambiguity handling:** the chord-id dataset's `is_ambiguous` flag (chord diatonic to both keys) has a chroma-level analog — a pitch-class window consistent with both key_a's and key_b's scales. The same "flag, don't exclude by default" convention from `sequence_dataset.py` should carry over, so masked-vs-unmasked accuracy remains comparable across Phase 1 and Phase 2 datasets.

This is a design sketch for when Option B is approved — **no dataset generator code should be written yet.**

## 6. Real MIDI evaluation design (for when Option A and/or B are implemented)

### Twinkle.mid

- **Expected behavior:** given the documented, already-verified true-Twinkle baseline (stable, F-C-G-D-clustered Circle-of-Fifths walk; see `Twinkle_mid_EMA_MLP_Baseline_RunNotes.md`), a representation that better preserves tonal evidence should ideally show **stable behavior clustered on or near C major specifically** (the piece's actual, simple tonic), not merely "clustered in the tonic neighborhood" the way both Phase 1.5B models already were.
- **Comparison point:** directly against Phase 1.5B's documented F-major overconfidence (SRN: 67.0% F major, avg_prob_c_major=0.072) — did avoiding triadic forcing reduce that overconfident F-major bias, or does it persist even with richer input?

### Twinkle 12.mid

- **Embedded key-signature events, inspected directly from the MIDI file (read-only, not modified):**
  - `t=0.0s`: C major
  - `t=384.0s`: Eb major (repeated at `t=392.0s`)
  - `t=432.0s`: C major (repeated at `t=440.0s`)
  - (Time signature also changes at `t=600.0s`, from 2/4 to 3/4 — a further complication worth being aware of, though not a key change.)
  - These are exact `pretty_midi` `key_signature_changes` events, confirmed by direct inspection for this planning document; no MIDI file was modified to obtain them.
- **Evaluation design once a Phase 2 model exists:** compute the same descriptive metrics used in Phase 1.5B (confidence, entropy, Circle-of-Fifths jump statistics, top predicted keys) in three windows aligned to the real key-signature events — before `t=384s` (C major), during `t=384-432s` (Eb major), and after `t=432s` (C major again) — and report whether predicted keys shift in the correct direction and at roughly the correct times around these markers, rather than computing a single aggregate figure that could hide or wash out a correct local response. This is still descriptive, not accuracy-labeled (per guardrail 8 below), but the known ground-truth timestamps let it be a much more targeted qualitative check than Phase 1.5B could do without them.

## 7. Minimal implementation roadmap

Small, sequential steps — not a rewrite of the existing Phase 1/1.5 codebase:

- **Phase 2A (this document):** planning only. Done.
- **Phase 2B:** modularize the existing pitch-class baseline — freeze `SCALE_TEMPLATES` and `midi_to_key_baseline` (copied verbatim from the notebook, same convention as `shared_music_defs.py`) into a new, standalone module. No new logic, no new model.
- **Phase 2C:** extract and save chroma sequences for `Twinkle.mid` and `Twinkle 12.mid` (raw and/or smoothed chroma arrays), analogous to Phase 1.5A's saved chord-id `.npy` files, into a clearly separate derived-data location.
- **Phase 2D:** evaluate the modularized pitch-class baseline (Option A) on both saved chroma sequences, producing descriptive metrics/plots in a clearly Phase-2-labeled output location — no chord-id/triadic model involved at this step.
- **Phase 2E:** only after 2B-2D are reviewed, implement the Chroma SRN (Option B) — new dataset generator (section 5), new/adapted model, training, and evaluation, mirroring Phase 1's structure (diagnostic sweep, then plots, then a summary) but for the chroma representation.

Each step should be proposed and approved individually, the same way Phase 1 and Phase 1.5 were built incrementally file-by-file.

## 8. Guardrails

- **Do not modify Phase 1 / Phase 1.5 results.** All existing outputs in `05_Figures_Results/` (synthetic comparison, SRN diagnostic, visual comparison, Phase 1.5A/1.5B reports and figures) remain as historical record of those phases and must not be overwritten or edited.
- **Do not replace old scripts.** `shared_music_defs.py`, `sequence_dataset.py`, `mlp_baseline.py`, `srn_model.py`, `run_comparison.py`, `diagnose_srn_training.py`, `plotting_comparison.py`, `midi_chord_extraction.py`, and `run_midi_phase15_evaluation.py` all remain in place, unmodified; Phase 2 adds new, separate modules alongside them.
- **Do not mix chord-id and raw-chroma experiments without labeling them.** Any future script, output file, or plot must make clear in its name/title whether it is chord-id-based (Phase 1/1.5 style) or chroma/pitch-class-based (Phase 2), so results are never ambiguous about which representation produced them.
- **Do not claim accuracy on MIDI without ground-truth labels.** Twinkle 12.mid's key-signature events (section 6) provide approximate timing checkpoints, not a dense per-timestep label array — any future MIDI evaluation should continue reporting descriptive metrics (confidence, entropy, jump statistics, qualitative behavior around known key-change timestamps), not synthetic-style accuracy, consistent with Phase 1.5B's convention.
- **Keep Phase 2 outputs separate from Phase 1.5B outputs.** Future Phase 2 files should use a distinct naming convention (e.g. a `PHASE2_` prefix, mirroring `PHASE1_5A_`/`PHASE1_5B_`) and, if warranted, a separate subfolder, so the two phases' artifacts are never visually or programmatically confused.
