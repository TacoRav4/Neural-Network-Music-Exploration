# Phase 3E — Intermediate-Difficulty MIDI Corpus Selection Plan

**Planning document only. No new MIDI files added, no extraction, no evaluation, no neural model, no Transformer. No existing scripts, notebooks, or Phase 1/1.5/2/3 outputs have been modified.**

## 1. Why new MIDI examples are needed

Per `PHASE3D_STAGE4_DECISION_REVIEW.md`, the two existing test pieces cannot resolve the open Stage 4 question:

- **Twinkle.mid is too simple.** The pitch-class baseline is already perfect on it by every Phase 3B measure (100% C major, 0 key switches, 0 large jumps, 0 anchor mismatches) — there is no local failure of any kind to diagnose or refine.
- **Twinkle 12.mid is a high-stress/extreme case.** Per Phase 3C, chord-id EMA/SRN disagreement with the pitch-class baseline is global (87.7%–100% overall across both pieces), not concentrated in any flaggable local region — Twinkle 12.mid's difficulty (dense ornamentation, 1271 predictions, real modulation) produces a pervasive representation-wide bias, not a bounded local problem.
- **Neither piece tests local, diagnosable Stage 1 failures.** Phase 3B's diagnostics (`key_switch`, `large_jump`, `anchor_mismatch`, `high_tie_count`) worked exactly as designed — they just found "almost nothing" on Twinkle.mid and "not much, and not correlated with chord-id disagreement" on Twinkle 12.mid. The staged architecture's Stage 4 premise (a validated fast filter with a small number of specific, fixable failure regions) has never actually been exercised by a piece that produces that pattern.
- **Stage 4 neural refinement should stay deferred** until intermediate examples exist that can distinguish "there was nothing to refine" from "refinement helped" from "refinement didn't help" — none of which the current two-piece evidence base can distinguish.

## 2. Corpus size

A small, curated first-pass corpus: **4 to 6 new MIDI files maximum.** Not a large dataset — the goal is targeted diagnostic coverage of specific gaps (see section 3), not statistical breadth. Curate and analyze this small set first (Phase 3F/3G/3H below); only consider scaling up later, and only if the small set proves informative enough to be worth extending.

## 3. Required categories

| | Category | Purpose |
|---|---|---|
| A | Simple monophonic melody in a **non-C major** key | Tests whether the pitch-class baseline's apparent strength (100% correct on Twinkle.mid) is specific to C major or generalizes to other tonics — currently untested, since both existing pieces are C-major-centered. |
| B | Simple **minor-key** melody | Directly probes the relative-major/minor ambiguity Phase 3B identified as the representation's core structural limitation (`SCALE_TEMPLATES` gives relative pairs identical rows). Neither existing piece is in a minor key — this specific, well-understood failure mode has never been exercised. |
| C | Melody **with accompaniment**, no modulation | Tests behavior outside the sparse-chroma/high-tie-count regime Phase 3B found dominant for monophonic material — denser chroma per window may reduce template ties and produce a different uncertainty profile. |
| D | **Ornamented but non-modulating** melody | Isolates the "ornamentation collapse" variable from the "real modulation" variable, which Twinkle 12.mid currently conflates (it has both dense ornamentation *and* a real modulation at once, making it impossible to attribute Twinkle 12.mid's behavior to either cause alone). |
| E | Short piece with **one clear modulation** | A cleaner, more isolated test of modulation-tracking than Twinkle 12.mid — simpler texture, single modulation, easier to build tight anchor windows around, more likely to produce a genuinely local (rather than global) failure region if the baseline struggles. |
| F | *(Optional)* A **second classical variation** or short sonatina excerpt, less extreme than Mozart's 12 Variations | A data point between "trivially simple" (Twinkle.mid) and "maximally stressful" (Twinkle 12.mid) on the same general genre (theme-and-variations / classical figuration), to see whether Twinkle 12.mid's extremity is about genre or about degree. |

## 4. Selection criteria

For each candidate MIDI file, require:

- **Public-domain or clearly usable educational/classical source** — no ambiguous licensing.
- **Expected key documented** — tonic and mode recorded before any analysis, so later results can be checked against a known-correct expectation (the same convention used for Twinkle.mid's assumed C-major tonic and Twinkle 12.mid's embedded key signatures).
- **Texture documented**: monophonic / accompaniment / ornamented / modulation (matching the categories in section 3, so each file's role in the corpus is unambiguous).
- **Expected difficulty level documented** — a brief, honest prediction (easy / moderate / hard) for both the pitch-class baseline and the chord-id EMA/SRN, recorded *before* running any diagnostics, so later Phase 3G/3H results can be compared against the prediction rather than only interpreted after the fact.
- **Short enough for readable plots** — roughly Twinkle.mid-scale (tens of seconds to low minutes), not Twinkle-12-scale (over 10 minutes, 1374 windows) — long pieces make trajectory/timeline plots hard to read and slow down iteration.
- **Preferably includes real `key_signature_changes`** if modulation is involved (category E, and F if it modulates) — real embedded MIDI metadata is strongly preferred over an assumed/manual key, per the precedent set by Twinkle 12.mid's confirmed, `pretty_midi`-readable key-signature events.
- **No corrupted or overly dense multi-track MIDI** — single or few tracks, clean note data, no the kind of malformed-header warning `pretty_midi` already emits for Twinkle 12.mid (non-zero-track tempo/key/time-signature events) if avoidable.
- **Source URL or source note recorded manually by the user** — this workspace does not fetch external files autonomously; any candidate's provenance must be recorded by the user as part of collection, not inferred or guessed.

## 5. Metadata template (for each future MIDI, once collected)

```
filename:                          <e.g. Example_G_major_melody.mid>
source:                            <where the file came from>
license / public-domain note:      <explicit statement, not assumed>
expected tonic:                    <e.g. G>
expected mode:                     <major / minor>
known key changes:                 <list of {time, key} if any, or "none expected">
texture category:                  <A/B/C/D/E/F per section 3>
reason selected:                   <which gap in the corpus this fills>
expected challenge — pitch-class baseline:   <easy / moderate / hard, with brief reasoning>
expected challenge — chord-id EMA/SRN:       <easy / moderate / hard, with brief reasoning>
manual anchor windows needed:      <yes/no — and if yes, what they should be>
```

This mirrors the fields already established for Twinkle.mid/Twinkle 12.mid across `PHASE1_5A_MIDI_chord_extraction_report.md`, `PHASE2C_MIDI_chroma_extraction_report.md`, and the anchor-window convention from `PHASE2D_pitch_class_baseline_report.md`, so new files slot into the existing pipeline without inventing new conventions.

## 6. Folder proposal

Recommended future location: **`03_MIDI_Data/candidate_intermediate_midi/`** — kept separate from the existing `Twinkle.mid`/`Twinkle 12.mid` pair so the original two-piece evidence base (referenced throughout Phases 1.5–3D) remains unambiguous and undisturbed. **This folder is not created in this task** — only proposed, per the explicit instruction not to create it unless requested.

## 7. Evaluation plan after files are collected

Once candidate MIDI files and their metadata (section 5) exist, in order:

- **Phase 3F**: import the selected MIDI files into `03_MIDI_Data/candidate_intermediate_midi/` (or wherever finalized) and record their metadata using the section 5 template — no diagnostics yet, just intake and documentation.
- **Phase 3G**: run the existing Phase 2C (chroma extraction) → Phase 2D (pitch-class baseline evaluation) → Phase 3B (uncertainty diagnostics) pipeline on the new files, exactly as already validated on Twinkle.mid/Twinkle 12.mid — reusing existing modules, not rewriting them for the new corpus.
- **Phase 3H**: decide whether any new file produces a genuine **local** Stage 1 failure (a bounded, diagnosable difficult region, unlike either existing piece) that would make it a suitable target for reconsidering Stage 4 neural refinement — this is the actual go/no-go decision point Phase 3D's deferral was waiting on.

## 8. Guardrails

- **No new MIDI files added yet** — this document only plans what to look for.
- **No extraction/evaluation yet** — Phase 3F/3G/3H are described but not started.
- **No neural model yet.**
- **No Transformer yet.**
- **Preserve all previous outputs** — nothing in this plan implies modifying any existing script (`shared_music_defs.py` through `compare_phase3c_disagreement.py`) or any prior Phase 1/1.5/2/3 output file.
