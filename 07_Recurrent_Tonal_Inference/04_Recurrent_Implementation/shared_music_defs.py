"""shared_music_defs.py

Frozen copy of the COGS 202 chord/key vocabulary used by
02_Baseline_Pipeline/Mini Capstone_Project_A_walking_machine_of_the_music.ipynb.

This module exists so the recurrent (SRN) continuation in
04_Recurrent_Implementation/ can reuse the exact same chord/key indexing,
diatonic triad sets, and circle-of-fifths geometry as the completed COGS 202
baseline notebook, without importing from or editing the notebook itself.

Everything below is copied as closely as possible to the notebook's original
logic (same variable names, same formulas, same comments where present).
Nothing here has been improved, refactored, or musically reinterpreted --
that is intentional. This is a freeze, not a redesign. Any future changes to
the tonal vocabulary belong in a new module, not here.

Dependency-light by design: numpy only. torch is not required -- one_hot
below returns a plain numpy float32 array, matching the notebook's own
`one_hot` helper (the notebook separately wraps this in `torch.tensor(...)`
at call sites; that wrapping is not part of the frozen vocabulary itself).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Musical "vocabulary": pitch classes, chords, keys
# (notebook cells 7-8)
#
# We index pitch classes C = 0, C# = 1, ..., B = 11.
# Chords: 24 total = 12 major triads + 12 minor triads.
# Keys:   24 total = 12 major keys + 12 minor keys.
# ---------------------------------------------------------------------------

PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def chord_index(root_pc: int, quality: str) -> int:
    # quality: "maj" or "min"
    assert 0 <= root_pc <= 11
    assert quality in ["maj", "min"]
    return root_pc if quality == "maj" else root_pc + 12


def key_index(tonic_pc: int, quality: str) -> int:
    # quality: "maj" or "min"
    assert 0 <= tonic_pc <= 11
    assert quality in ["maj", "min"]
    return tonic_pc if quality == "maj" else tonic_pc + 12


def decode_key(k: int):
    tonic = k % 12
    mode = "maj" if k < 12 else "min"
    return PC_NAMES[tonic], mode


def decode_chord(c: int):
    root = c % 12
    qual = "maj" if c < 12 else "min"
    return PC_NAMES[root], qual


# ---------------------------------------------------------------------------
# Converting MIDI to Best fitting Chord: chord templates
# (notebook cell 6)
#
# Create 12-dimensional chromatic templates for our 24 chords.
# A 2D matrix:
#   rows = 24 = 12 major triads + 12 minor triads
#   columns = 12 = 12 pitch classes
# ---------------------------------------------------------------------------

CHORD_TEMPLATES = np.zeros((24, 12))

for root in range(12):
    # Major triad: root, major third(+4), perfect fifth (+7)
    # CHORD_TEMPLATES shows what perfet major and minor chords look like
    CHORD_TEMPLATES[root, root] = 1
    CHORD_TEMPLATES[root, (root + 4) % 12] = 1
    CHORD_TEMPLATES[root, (root + 7) % 12] = 1

    # Minor triad: root, minor third(+3), perfect fifth (+7)
    CHORD_TEMPLATES[root + 12, root] = 1
    CHORD_TEMPLATES[root + 12, (root + 3) % 12] = 1
    CHORD_TEMPLATES[root + 12, (root + 7) % 12] = 1


# ---------------------------------------------------------------------------
# Circle of fifths distance + key-induction gradient
# (notebook cell 28)
#
# Setting distance on the circle of fifths.
# For a tonic pc, its position on circle of fifths can be mapped by
# repeated +7 moves.
# ---------------------------------------------------------------------------

# map tonic pc -> circle-of-fifths index (0..11) relative to C=0

# brute-force solve
FIFTH_POS = {}
for k in range(12):
    FIFTH_POS[(7 * k) % 12] = k  # position(pc) = k such that (7*k) % 12 == pc


def fifth_distance(pc_a, pc_b):
    a = FIFTH_POS[pc_a]
    b = FIFTH_POS[pc_b]
    d = abs(a - b)
    return min(d, 12 - d)


def key_tonic_pc(key_id):
    return key_id % 12


# ---------------------------------------------------------------------------
# Synthetic "exposure" dataset: diatonic triads sampled by key
# (notebook cells 9-19)
#
# Major key diatonic triads (triads qualities):
#   I maj, ii min, iii min, IV maj, V maj, vi min
# Minor key (natural minor-ish):
#   i min, III maj, iv min, v min, VI maj, VII maj
# ---------------------------------------------------------------------------


def major_key_triads_set(tonic_pc: int):
    # returns list of chord indices for diatonic triads (skip diminished)
    # scale degrees: 0,2,4,5,7,9 (I, ii, iii, IV, V, vi)
    roots = [(tonic_pc + x) % 12 for x in [0, 2, 4, 5, 7, 9]]
    quals = ["maj", "min", "min", "maj", "maj", "min"]
    return [chord_index(r, q) for r, q in zip(roots, quals)]


def minor_key_triad_set(tonic_pc: int):
    # natural minor-ish: i, III, iv, v, VI, VII
    roots = [(tonic_pc + x) % 12 for x in [0, 3, 5, 7, 8, 10]]
    quals = ["min", "maj", "min", "min", "maj", "maj"]
    return [chord_index(r, q) for r, q in zip(roots, quals)]


def key_triad_set(key_id: int):
    tonic = key_id % 12
    mode = "maj" if key_id < 12 else "min"
    return major_key_triads_set(tonic) if mode == "maj" else minor_key_triad_set(tonic)


# Weights for the 6 chords we are using
# We will emphasize tonic + dominant (+ subdominant)
MAJ_WEIGHTS = np.array([0.28, 0.10, 0.08, 0.18, 0.26, 0.10])  # I, ii, iii, IV, V, vi
MIN_WEIGHTS = np.array([0.30, 0.12, 0.16, 0.14, 0.16, 0.12])  # i, III, iv, v, VI, VII


def sample_chord_from_key(key_id: int) -> int:
    triads = key_triad_set(key_id)
    w = MAJ_WEIGHTS if key_id < 12 else MIN_WEIGHTS
    return int(np.random.choice(triads, p=w / w.sum()))


# ---------------------------------------------------------------------------
# Onehot encoding
# (notebook cell 21)
# ---------------------------------------------------------------------------


def one_hot(n, idx):
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Verification block
#
# Sanity-checks that this frozen copy matches the notebook's stated
# invariants. Not a training or figure-generation script -- run this file
# directly (`python shared_music_defs.py`) to re-check the freeze at any
# time. No SRN, dataset, or model code lives here.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    checks = []

    # 24 chord templates exist
    checks.append(("CHORD_TEMPLATES has 24 rows", CHORD_TEMPLATES.shape[0] == 24))

    # CHORD_TEMPLATES shape is correct (24 chords x 12 pitch classes)
    checks.append(("CHORD_TEMPLATES shape == (24, 12)", CHORD_TEMPLATES.shape == (24, 12)))

    # 24 keys exist (0..23 all decodable, 12 maj + 12 min)
    all_keys = [decode_key(k) for k in range(24)]
    maj_keys = [k for k in all_keys if k[1] == "maj"]
    min_keys = [k for k in all_keys if k[1] == "min"]
    checks.append(("24 keys total (0..23)", len(all_keys) == 24))
    checks.append(("12 major keys + 12 minor keys", len(maj_keys) == 12 and len(min_keys) == 12))

    # decode_chord / chord_index round-trip
    chord_roundtrip_ok = True
    for root_pc in range(12):
        for quality in ["maj", "min"]:
            c = chord_index(root_pc, quality)
            name, qual = decode_chord(c)
            if name != PC_NAMES[root_pc] or qual != quality:
                chord_roundtrip_ok = False
    checks.append(("decode_chord/chord_index round-trip", chord_roundtrip_ok))

    # decode_key / key_index round-trip
    key_roundtrip_ok = True
    for tonic_pc in range(12):
        for quality in ["maj", "min"]:
            k = key_index(tonic_pc, quality)
            name, mode = decode_key(k)
            if name != PC_NAMES[tonic_pc] or mode != quality:
                key_roundtrip_ok = False
    checks.append(("decode_key/key_index round-trip", key_roundtrip_ok))

    # FIFTH_POS maps all 12 pitch classes
    checks.append(("FIFTH_POS covers all 12 pitch classes", sorted(FIFTH_POS.keys()) == list(range(12))))
    checks.append(("FIFTH_POS positions are a permutation of 0..11", sorted(FIFTH_POS.values()) == list(range(12))))

    print("shared_music_defs.py verification")
    print("-" * 50)
    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")
    print("-" * 50)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
