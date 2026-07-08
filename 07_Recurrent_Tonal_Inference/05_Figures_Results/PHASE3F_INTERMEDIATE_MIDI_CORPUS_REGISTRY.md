# Phase 3F — Intermediate MIDI Corpus Registry

**Metadata QA and registry only. No tonal-inference extraction or evaluation (Phase 2C/2D/3B) has been run on any candidate file. No existing scripts, notebooks, or Phase 1/1.5/2/3 outputs have been modified.**

## 1. Candidate discovery

Files found in `03_MIDI_Data/` before organization:

| Filename | Status |
|---|---|
| `Twinkle.mid` | Existing benchmark (Level 1) |
| `Twinkle 12.mid` | Existing benchmark (Level 6) |
| `J.S. Bach - Minuet in G Major, BWV Anh. 114.mid` | New candidate |
| `Beethoven - Fur Elise.mid.mid` | New candidate (note the double `.mid.mid` extension in the original filename — preserved as-is, not renamed) |
| `f-f-chopin-prelude-op-28-no-4.mid` | New candidate (matches the expected E minor prelude) |
| `f-f-chopin-prelude-op-28-no-2.mid` | New candidate, **not previously anticipated** — the A minor prelude (Op. 28 No. 2), found alongside No. 4. Registered as an extra/unassigned-ladder-level candidate (see section 4). |

## 2. Candidate folder organization

`03_MIDI_Data/candidate_intermediate_midi/` was created (did not previously exist). The 4 new candidate files were **moved** (not copied — no root duplicates remain) from `03_MIDI_Data/` into this folder:

| Before | After |
|---|---|
| `03_MIDI_Data/J.S. Bach - Minuet in G Major, BWV Anh. 114.mid` | `03_MIDI_Data/candidate_intermediate_midi/J.S. Bach - Minuet in G Major, BWV Anh. 114.mid` |
| `03_MIDI_Data/Beethoven - Fur Elise.mid.mid` | `03_MIDI_Data/candidate_intermediate_midi/Beethoven - Fur Elise.mid.mid` |
| `03_MIDI_Data/f-f-chopin-prelude-op-28-no-4.mid` | `03_MIDI_Data/candidate_intermediate_midi/f-f-chopin-prelude-op-28-no-4.mid` |
| `03_MIDI_Data/f-f-chopin-prelude-op-28-no-2.mid` | `03_MIDI_Data/candidate_intermediate_midi/f-f-chopin-prelude-op-28-no-2.mid` |

`Twinkle.mid` and `Twinkle 12.mid` were **not moved** and remain at their original paths in `03_MIDI_Data/`.

## 3. Inspection script

`04_Recurrent_Implementation/inspect_intermediate_midi_candidates.py` loads each candidate with `pretty_midi` and reports duration, instrument count/names/programs, total notes, notes/sec, approximate max polyphony (note on/off sweep), key signature changes, time signature changes, tempo changes, and QA warning flags (very long file >300s, very dense file >8 notes/sec, no key-signature metadata, many instruments >3, or a load failure). It performs **no tonal-inference extraction** — no chroma, no pitch-class baseline, no chord-id path. Run via `python inspect_intermediate_midi_candidates.py` (defaults to scanning `candidate_intermediate_midi/`) or with explicit file paths as arguments.

## 4. Corpus ladder

| Level | File | Key/Mode | Role | Scope | Status |
|---|---|---|---|---|---|
| 1 | `Twinkle.mid` | C major | Monophonic sanity check | Full piece | Existing reference, not reprocessed |
| 2 | Bach — Minuet in G Major, BWV Anh. 114 | G major | Non-C tonic + light accompaniment | Full piece | **Candidate, moved, QA: warn (no key-signature metadata)** |
| 3 | Beethoven — Für Elise (opening theme) | A minor | Relative-major/minor ambiguity test | **Excerpt (opening theme only) — exact cut time TBD** | **Candidate, moved, QA: warn (no key-signature metadata)** |
| 4 | Chopin — Prelude in E minor, Op. 28 No. 4 | E minor | Minor + slow harmony + chromatic pressure | Full piece (tentative) | **Candidate, moved, QA: warn (no key-signature metadata)** |
| 5 | *(none yet)* | TBD | Short, clear single modulation | TBD | **MISSING — no candidate file currently fills this role** |
| 6 | `Twinkle 12.mid` | C major (→ Eb → C) | High-stress ornamented variation / modulation stress test | Full piece | Existing reference, not reprocessed |

**Extra, unassigned candidate:** Chopin — Prelude in A minor, Op. 28 No. 2 (`f-f-chopin-prelude-op-28-no-2.mid`) was also found and moved into the candidate folder, but is **not assigned a ladder level** in this plan — Level 3 (Für Elise) and Level 4 (Chopin No. 4) already cover the minor-key/relative-ambiguity and minor-plus-chromaticism roles. It is retained as a backup/extra example (this piece is independently well known for its own harmonic ambiguity, even to human listeners) in case Level 3 or 4 needs a harder replacement later.

**Level 5 is explicitly missing.** None of the 4 newly added candidates are primarily single-modulation pieces: the Bach Minuet and both Chopin preludes stay in one key throughout, and while Für Elise's later sections do modulate, the recommended excerpt deliberately excludes them (to keep the minor-key test isolated from a modulation test, per Phase 3E's category-isolation goal). This slot remains open for a future addition.

## 5. QA findings summary

All 4 new candidates **loaded successfully** with `pretty_midi` — no load failures. All 4 are flagged **`warn`**, and only for one reason each: **no embedded `key_signature_changes` metadata** (none of the 4 files carry this MIDI metadata, unlike `Twinkle 12.mid`). No file was flagged for length, density, or instrument count.

| File | Duration | Instruments | Notes | Notes/sec | Max polyphony (approx.) | Key sig. events | Warnings | Recommendation |
|---|---|---|---|---|---|---|---|---|
| Bach Minuet in G | 82.3s | 1 | 430 | 5.23 | 6 | 0 | no_key_signature_metadata | **warn** |
| Für Elise | 230.3s | 1 | 1042 | 4.52 | 10 | 0 | no_key_signature_metadata | **warn** |
| Chopin Prelude No. 4 (E minor) | 108.2s | 2 | 600 | 5.54 | 13 | 0 | no_key_signature_metadata | **warn** |
| Chopin Prelude No. 2 (A minor, extra) | 76.7s | 2 | 342 | 4.46 | 11 | 0 | no_key_signature_metadata | **warn** |

None of the "warn" flags are disqualifying — the absence of embedded key-signature metadata is expected for these sources (unlike Mozart's 12 Variations, which happens to carry it) and is exactly why each entry's `expected_key`/`expected_mode` fields in the registry (section 6) must be treated as an assumed, human-recorded anchor rather than MIDI-derived ground truth, the same convention already used for `Twinkle.mid`'s assumed C-major tonic.

Für Elise's full duration (230.3s) is well under the 300s "very long" threshold but is still ~4x longer than any other candidate and ~2x Twinkle 12.mid's per-window density scale when considered against its note count — this, combined with its later-section modulations, is why an opening-theme excerpt (not the full piece) is recommended in the ladder table, independent of the QA threshold not having triggered.

## 6. Registry file

Full structured registry, including all fields specified in the Phase 3E metadata template (source/license placeholders, expected key/mode, texture category, intended benchmark level, recommended scope, expected challenge predictions for both the pitch-class baseline and chord-id EMA/SRN, and complete QA metadata per file) is saved to:

**`03_MIDI_Data/candidate_intermediate_midi/candidate_midi_registry.json`**

`source_license` fields are placeholders (`<TBD -- user to fill in exact source URL / license note>`) — per Phase 3E's guardrail, this workspace does not infer or guess provenance; the user must record exact source/license information for each file before these candidates are used in any published analysis.

## 7. Scope note

This is Phase 3F: candidate discovery, organization, and metadata QA only. **No tonal-inference evaluation has started** — no chroma extraction, no pitch-class baseline run, no chord-id EMA/SRN run, no uncertainty diagnostics, on any candidate file. That is Phase 3G, not performed here. No neural refinement, Chroma SRN, or Transformer was implemented.
