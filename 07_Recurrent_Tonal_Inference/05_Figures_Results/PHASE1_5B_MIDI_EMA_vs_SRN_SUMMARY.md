# Phase 1.5B Summary — EMA+MLP vs. Elman SRN on Real MIDI-Derived Chord IDs

Consolidated summary of Phase 1.5B. Synthesizes `PHASE1_5B_MIDI_EMA_vs_SRN_report.md`, `PHASE1_5B_MIDI_EMA_vs_SRN_metrics.json`, the four Phase 1.5B PNGs, `PHASE1_5A_MIDI_chord_extraction_report.md`, and `PHASE1_SYNTHETIC_EMA_vs_SRN_SUMMARY.md`. No new runs, plots, or evaluation were performed to produce this document.

## 1. Phase 1.5B question

Phase 1.5B asks whether the Phase 1 result — that a learned Elman SRN hidden state improves on the notebook's hand-coded EMA/leaky-integration baseline — carries over from clean synthetic chord sequences to **real, MIDI-derived chord-id sequences**, while holding the input representation fixed. Both models consume the identical 24-dim one-hot chord/triad vector produced by the notebook's `midi_to_chord_ids` template-matching path (chroma → chroma-level EMA → triad template argmax, extracted in Phase 1.5A). This is a **data-source change only** relative to Phase 1 — not a representation change.

## 2. Model settings

| | EMA+MLP baseline | Elman SRN |
|---|---|---|
| Temporal mechanism | Hand-coded leaky integration, `alpha=0.20` | Learned recurrent hidden state, `hidden_size=48` |
| Training | `ChordToKeyMLP`, seed=269, 12 epochs | `epochs=25`, `lr=1e-3` (Phase 1's best condition), seed=269 |
| Input | Saved chord-id sequences from Phase 1.5A (`Twinkle_mid_chord_ids.npy`, `Twinkle_12_mid_chord_ids.npy`) | Same |
| Representation | 24-dim one-hot chord/triad — **no raw chroma input** | Same |

## 3. Twinkle.mid findings

- **Both models stay in the tonic-neighborhood region.** Top predicted keys for both cluster around C/F/G/D major — neither model wanders into distant, unrelated keys on this simple piece.
- **The SRN is smoother and more confident.** Mean top-1 confidence is 0.709 for the SRN vs. 0.271 for EMA; mean entropy is 0.944 for the SRN vs. 1.909 for EMA (uniform-24 reference: 3.178). The SRN also has zero large Circle-of-Fifths jumps (distance ≥ 3) vs. EMA's 2.
- **But the SRN collapses more strongly onto F major**: 67.0% of SRN predictions land on F major vs. 39.6% for EMA.
- **The SRN assigns less average probability to C major than EMA does**: avg_prob_c_major is 0.072 for the SRN vs. 0.147 for EMA — the SRN moves further away from the piece's actual tonic, not closer to it.
- **Read together, this is overconfidence, not straightforward improvement.** The SRN's smoother, lower-entropy trajectory (visible in `PHASE1_5B_Twinkle_EMA_vs_SRN_Probability_Tracking.png` as sharp, high-amplitude swings between F and G major) reflects a more decisive commitment to a specific — and arguably less correct — interpretation, not a cleaner recovery of the true C-major tonic.

## 4. Twinkle 12.mid findings

- **Both models remain unstable and visually "spiderweb"-like.** The side-by-side Circle-of-Fifths walk (`PHASE1_5B_Twinkle12_EMA_vs_SRN_Circle_of_Fifths.png`) shows dense, crisscrossing transitions across nearly the full circle for both models — neither produces the stable, clustered trajectory seen on the simple Twinkle.mid case.
- **The SRN reduces large fifth-jumps modestly**: 30 large jumps (2.4% of transitions) vs. EMA's 52 (4.1%), and a lower mean jump distance (0.18 vs. 0.22). This is a real, measurable reduction in the largest jumps, not a resolution of the underlying instability.
- **The SRN remains highly confident** (mean top-1 prob 0.761 vs. EMA's 0.243, mean entropy 0.821 vs. 1.836) while visiting *more* unique predicted keys overall (17 vs. 13) — confidence and instability coexist here rather than trading off.
- **Neither model recovers clean C-major/key-tracking behavior.** Both concentrate roughly 60% of predictions on F major and only ~12-13% on C major, despite this piece's real, embedded key-signature changes (documented at ≈384s and ≈432s). Triadic forcing and the piece's ornamentation remain the dominant bottleneck — the choice of temporal-memory mechanism does not change which key the models default to.

## 5. Research interpretation

Read alongside `PHASE1_SYNTHETIC_EMA_vs_SRN_SUMMARY.md`, Phase 1 and Phase 1.5B together support a specific, bounded claim:

- **Phase 1 showed learned recurrence helps on clean synthetic sequence data.** The SRN improved both overall accuracy (0.8101 vs. 0.7529) and modulation accuracy (0.8010 vs. 0.7485) over the EMA baseline on synthetic modulation sequences with well-formed, unambiguous diatonic triads.
- **Phase 1.5B shows learned recurrence cannot rescue distorted MIDI-derived triadic inputs.** On both real pieces, the SRN's gains are confined to confidence/smoothness metrics, not to recovering the correct tonic or resolving genuine key-tracking instability. On Twinkle.mid it becomes more confidently *wrong* about which tonic-neighborhood key dominates; on Twinkle 12.mid it becomes more confident while remaining just as key-diverse and only marginally less jump-prone.
- **Temporal modeling helps only when the representation preserves useful tonal evidence.** Notably, the Phase 1.5A chord-level extraction itself is not obviously broken: C major is the single most common extracted chord for both pieces (57/106 windows for Twinkle.mid, 852/1271 for Twinkle 12.mid — see `PHASE1_5A_MIDI_chord_extraction_report.md`). The distortion appears downstream of chord extraction, in how the single-chord-trained MLP (and, on top of it, either temporal-memory mechanism) maps that chord evidence to a *key* — both models end up favoring F major over C major as the dominant predicted key despite C major being the dominant extracted chord. No amount of learned recurrence over the hidden state can correct a systematic chord-to-key mapping bias that exists independent of time.
- **The main remaining failure is representational, not merely recurrent-memory related.** This reframes the project's open question: the bottleneck limiting real-MIDI performance sits in how the rigid 24-way major/minor triad representation (both the template-matching that produces chord ids, and the resulting chord-to-key mapping learned on top of it) handles real, ornamented, ambiguous chord evidence — not in the choice between hand-coded EMA and learned recurrence for combining chords over time. Swapping the temporal mechanism, on its own, does not fix a bias that originates in the representation.

## 6. Current limitations

- **No ground-truth per-timestep key labels for MIDI.** Phase 1.5B's metrics are descriptive (confidence, entropy, key diversity, Circle-of-Fifths jump statistics), not accuracy — there is no labeled "correct key at time t" to compare against for either real piece.
- **Descriptive metrics only, not accuracy.** Conclusions above are about stability/confidence/trajectory shape, not correctness in a measurable sense.
- **Still uses triadic forcing.** Both models operate downstream of the same 24-way major/minor triad template-matching step used throughout this workspace; no richer chord vocabulary (e.g. sevenths, diminished, ambiguous/no-chord windows) has been tried.
- **Raw chroma / pitch-class input not tested yet.** Neither model has been given the 12-dim chroma vector (or a pitch-class-only representation) directly — Phase 1.5B, like Phase 1, deliberately restricts both models to the same 24-dim triadic input.

## 7. Next approved phase recommendation

**Recommended: Phase 2 planning only** — not implementation. Based on the representational bottleneck identified in section 5, Phase 2 planning should scope:

- A **raw chroma or pitch-class recurrent model** (SRN operating on the 12-dim chroma vector, or a pitch-class representation, instead of the hard-templated 24-way chord id) to test whether preserving more of the original tonal evidence changes the real-MIDI results.
- A **comparison against the pitch-class baseline** already present in the notebook (the bypass-the-neural-network, direct-scale-template comparison), to establish whether a richer input representation alone — independent of the recurrence question — improves real-MIDI behavior.
- Explicit avoidance of triadic forcing in whatever Phase 2 representation is chosen, since section 5 identifies the current triad-template step as the likely dominant bottleneck.

**This task does not start Phase 2.** It only records the recommendation. Phase 2 implementation should wait for explicit approval, per `../STATUS.md`.
