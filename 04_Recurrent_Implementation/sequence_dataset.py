"""sequence_dataset.py

Synthetic labeled chord-sequence generator for the recurrent (SRN)
continuation of the tonal inference project.

This module builds sequences of chords (drawn from the same 24-chord
vocabulary and weighted diatonic sampling defined in shared_music_defs.py)
paired with a per-timestep *key* label, so that a future Elman SRN can be
trained/evaluated on "what key is currently active" rather than the
single-chord "what key would sample this chord" task the COGS 269/202 MLP
was trained on.

Nothing in here trains or defines a model -- this is dataset generation
only. It imports the frozen chord/key vocabulary from shared_music_defs.py
and does not reinterpret it.
"""

import numpy as np

from shared_music_defs import (
    key_index,
    key_triad_set,
    key_tonic_pc,
    decode_key,
    fifth_distance,
    MAJ_WEIGHTS,
    MIN_WEIGHTS,
    sample_chord_from_key,
    one_hot,
)


# ---------------------------------------------------------------------------
# Chord sampling helper
#
# shared_music_defs.sample_chord_from_key draws from the module-level
# np.random state (matches the notebook's own behavior). For dataset
# generation we additionally support an explicit np.random.Generator (rng)
# so a whole dataset can be produced reproducibly from a single seed without
# depending on global random state. When rng is None, we fall back to the
# frozen sample_chord_from_key exactly as-is.
# ---------------------------------------------------------------------------

def _sample_chord_from_key(key_id: int, rng=None) -> int:
    triads = key_triad_set(key_id)
    w = MAJ_WEIGHTS if key_id < 12 else MIN_WEIGHTS
    p = w / w.sum()
    if rng is not None:
        return int(rng.choice(triads, p=p))
    return sample_chord_from_key(key_id)


def _is_ambiguous_chord(chord_id: int, key_a: int, key_b: int) -> bool:
    # A chord is "ambiguous" between two keys if it is a diatonic triad of
    # both -- i.e. it doesn't by itself disambiguate which key is active.
    return (chord_id in key_triad_set(key_a)) and (chord_id in key_triad_set(key_b))


def _key_metadata(key_id: int, prefix: str) -> dict:
    tonic_pc = key_tonic_pc(key_id)
    tonic_name, mode = decode_key(key_id)
    return {
        f"{prefix}_key_id": key_id,
        f"{prefix}_tonic_pc": tonic_pc,
        f"{prefix}_tonic_name": tonic_name,
        f"{prefix}_mode": mode,
    }


# ---------------------------------------------------------------------------
# sequence_to_onehot
# ---------------------------------------------------------------------------

def sequence_to_onehot(chord_ids) -> np.ndarray:
    """(T,) chord id sequence -> (T, 24) one-hot float32 array."""
    return np.stack([one_hot(24, c) for c in chord_ids], axis=0)


# ---------------------------------------------------------------------------
# make_sequence_in_key
# ---------------------------------------------------------------------------

def make_sequence_in_key(tonic_pc: int, mode: str, length: int, rng=None) -> dict:
    """Single-key chord sequence (no modulation), diatonic-sampled."""
    key_id = key_index(tonic_pc, mode)
    chord_ids = np.array([_sample_chord_from_key(key_id, rng) for _ in range(length)], dtype=np.int64)
    y = np.full(length, key_id, dtype=np.int64)
    is_ambiguous = np.zeros(length, dtype=bool)
    x = sequence_to_onehot(chord_ids)

    metadata = {
        "type": "single_key",
        "length": length,
        **_key_metadata(key_id, "key"),
    }

    return {
        "chord_ids": chord_ids,
        "x": x,
        "y": y,
        "pivot_idx": None,
        "is_ambiguous": is_ambiguous,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# make_no_modulation_sequence
#
# Thin, explicitly-named wrapper around make_sequence_in_key: this is the
# "negative control" case described in the SRN implementation plan -- a
# sequence that never modulates, so the SRN doesn't learn to always expect
# a pivot.
# ---------------------------------------------------------------------------

def make_no_modulation_sequence(key, length: int, rng=None) -> dict:
    """key: either a key_id (int, 0..23) or a (tonic_pc, mode) tuple."""
    if isinstance(key, tuple):
        tonic_pc, mode = key
    else:
        tonic_pc, mode = key_tonic_pc(key), decode_key(key)[1]

    seq = make_sequence_in_key(tonic_pc, mode, length, rng=rng)
    seq["metadata"]["type"] = "no_modulation"
    return seq


# ---------------------------------------------------------------------------
# make_modulation_sequence
# ---------------------------------------------------------------------------

def make_modulation_sequence(key_a: int, key_b: int, len_a: int, len_b: int, rng=None) -> dict:
    """
    Two-key modulation sequence: key_a for the first len_a chords, then
    key_b for the following len_b chords.

    Labeling (per spec):
      - labels before pivot are key_a
      - labels at and after pivot are key_b
      - pivot_idx marks the first timestep assigned to key_b (== len_a)
    """
    chords_a = [_sample_chord_from_key(key_a, rng) for _ in range(len_a)]
    chords_b = [_sample_chord_from_key(key_b, rng) for _ in range(len_b)]
    chord_ids = np.array(chords_a + chords_b, dtype=np.int64)

    T = len_a + len_b
    pivot_idx = len_a
    y = np.array([key_a] * len_a + [key_b] * len_b, dtype=np.int64)

    is_ambiguous = np.array(
        [_is_ambiguous_chord(c, key_a, key_b) for c in chord_ids], dtype=bool
    )

    x = sequence_to_onehot(chord_ids)

    tonic_a, tonic_b = key_tonic_pc(key_a), key_tonic_pc(key_b)
    metadata = {
        "type": "modulation",
        "length": T,
        "len_a": len_a,
        "len_b": len_b,
        "fifth_distance": fifth_distance(tonic_a, tonic_b),
        **_key_metadata(key_a, "a"),
        **_key_metadata(key_b, "b"),
    }

    return {
        "chord_ids": chord_ids,
        "x": x,
        "y": y,
        "pivot_idx": pivot_idx,
        "is_ambiguous": is_ambiguous,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Key pair sampling
#
# Kept deliberately simple: "close" pairs are drawn from fifth-distance
# 1-2 (immediate tonal neighbors, e.g. C -> G/F/Am/Em-ish), "distant" pairs
# from fifth-distance 3-6 (harder modulations, e.g. C -> F#). Mode (maj/min)
# for each side is chosen independently at random.
# ---------------------------------------------------------------------------

def _sample_key_pair(rng: np.random.Generator, closeness: str) -> tuple:
    tonic_a = int(rng.integers(0, 12))
    mode_a = "maj" if rng.random() < 0.5 else "min"
    key_a = key_index(tonic_a, mode_a)

    if closeness == "close":
        dist_choices = [1, 2]
    elif closeness == "distant":
        dist_choices = [3, 4, 5, 6]
    else:
        raise ValueError(f"unknown closeness: {closeness}")

    target_dist = int(rng.choice(dist_choices))
    # find all tonic_b candidates at exactly target_dist from tonic_a
    candidates = [pc for pc in range(12) if fifth_distance(tonic_a, pc) == target_dist]
    tonic_b = int(rng.choice(candidates))
    mode_b = "maj" if rng.random() < 0.5 else "min"
    key_b = key_index(tonic_b, mode_b)

    return key_a, key_b


# ---------------------------------------------------------------------------
# make_modulation_dataset
# ---------------------------------------------------------------------------

def make_modulation_dataset(
    n_sequences: int,
    length_range=(16, 48),
    pivot_range=None,
    include_no_modulation=True,
    rng_seed=269,
):
    """
    Build a list of labeled sequences (dicts, see make_* functions above)
    for training/evaluating the future SRN.

    length_range: (min, max) inclusive bounds used either for per-segment
        lengths (len_a, len_b each drawn independently from this range when
        pivot_range is None) or for total sequence length T (when
        pivot_range is given).
    pivot_range: None, or (min_frac, max_frac) in (0, 1). When given, T is
        drawn from length_range, and the pivot fraction is drawn from
        pivot_range, so pivot position is controlled directly rather than
        being an emergent side effect of two independent segment lengths.
    include_no_modulation: if True, reserve ~20% of n_sequences as
        single-key negative-control sequences (no pivot).
    rng_seed: seed for the local np.random.Generator driving this dataset's
        sampling (segment lengths, key pairs, chord draws). Uses SEED=269
        by default to match the notebook's fixed seed for continuity, not
        because it must match exactly.
    """
    rng = np.random.default_rng(rng_seed)

    n_no_mod = int(round(0.2 * n_sequences)) if include_no_modulation else 0
    n_mod = n_sequences - n_no_mod
    n_close = n_mod // 2
    n_distant = n_mod - n_close

    dataset = []

    for _ in range(n_no_mod):
        tonic = int(rng.integers(0, 12))
        mode = "maj" if rng.random() < 0.5 else "min"
        length = int(rng.integers(length_range[0], length_range[1] + 1))
        dataset.append(make_no_modulation_sequence((tonic, mode), length, rng=rng))

    for closeness, n in [("close", n_close), ("distant", n_distant)]:
        for _ in range(n):
            key_a, key_b = _sample_key_pair(rng, closeness)

            if pivot_range is None:
                len_a = int(rng.integers(length_range[0], length_range[1] + 1))
                len_b = int(rng.integers(length_range[0], length_range[1] + 1))
            else:
                total_len = int(rng.integers(length_range[0], length_range[1] + 1))
                frac = rng.uniform(pivot_range[0], pivot_range[1])
                len_a = max(1, int(round(total_len * frac)))
                len_b = max(1, total_len - len_a)

            seq = make_modulation_sequence(key_a, key_b, len_a, len_b, rng=rng)
            seq["metadata"]["closeness"] = closeness
            dataset.append(seq)

    rng.shuffle(dataset)  # shuffles the list order in place (numpy handles object arrays/lists fine)
    return dataset


# ---------------------------------------------------------------------------
# Verification block
#
# Sanity-checks the dataset generator's core invariants. Not a training or
# figure-generation script -- run this file directly
# (`python sequence_dataset.py`) to re-check at any time. No SRN, model, or
# training-loop code lives here.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    checks = []
    rng = np.random.default_rng(269)

    # --- C major -> G major example ---
    key_c_maj = key_index(0, "maj")
    key_g_maj = key_index(7, "maj")
    cg_seq = make_modulation_sequence(key_c_maj, key_g_maj, len_a=12, len_b=12, rng=rng)

    checks.append(("C->G x shape == (T, 24)", cg_seq["x"].shape == (24, 24)))
    checks.append(("C->G y shape == (T,)", cg_seq["y"].shape == (24,)))
    checks.append(("C->G pivot_idx == len_a (12)", cg_seq["pivot_idx"] == 12))
    checks.append((
        "C->G labels before pivot == key_a, at/after pivot == key_b",
        bool(np.all(cg_seq["y"][:12] == key_c_maj)) and bool(np.all(cg_seq["y"][12:] == key_g_maj)),
    ))
    checks.append(("C->G chord_ids within 0..23", bool(np.all((cg_seq["chord_ids"] >= 0) & (cg_seq["chord_ids"] <= 23)))))
    checks.append(("C->G key labels within 0..23", bool(np.all((cg_seq["y"] >= 0) & (cg_seq["y"] <= 23)))))
    checks.append(("C->G is_ambiguous length == T", len(cg_seq["is_ambiguous"]) == 24))
    checks.append(("C->G is_ambiguous dtype == bool", cg_seq["is_ambiguous"].dtype == np.bool_))

    # --- no-modulation sequence ---
    no_mod_seq = make_no_modulation_sequence((0, "maj"), length=20, rng=rng)
    checks.append(("no-modulation x shape == (T, 24)", no_mod_seq["x"].shape == (20, 24)))
    checks.append(("no-modulation y shape == (T,)", no_mod_seq["y"].shape == (20,)))
    checks.append(("no-modulation pivot_idx is None", no_mod_seq["pivot_idx"] is None))
    checks.append(("no-modulation has only one key label", len(set(no_mod_seq["y"].tolist())) == 1))
    checks.append(("no-modulation is_ambiguous all False", bool(np.all(no_mod_seq["is_ambiguous"] == False))))

    # --- generic sequence_to_onehot check ---
    chord_ids_sample = np.array([0, 5, 12, 23], dtype=np.int64)
    oh = sequence_to_onehot(chord_ids_sample)
    checks.append(("sequence_to_onehot shape == (4, 24)", oh.shape == (4, 24)))
    checks.append(("sequence_to_onehot rows sum to 1", bool(np.allclose(oh.sum(axis=1), 1.0))))

    # --- full dataset generation ---
    dataset = make_modulation_dataset(n_sequences=40, length_range=(8, 16), rng_seed=269)
    checks.append(("dataset has requested number of sequences", len(dataset) == 40))

    shapes_ok = all(seq["x"].shape == (len(seq["chord_ids"]), 24) for seq in dataset)
    checks.append(("all dataset sequences: x.shape == (T, 24)", shapes_ok))

    y_shapes_ok = all(seq["y"].shape == (len(seq["chord_ids"]),) for seq in dataset)
    checks.append(("all dataset sequences: y.shape == (T,)", y_shapes_ok))

    chord_range_ok = all(bool(np.all((seq["chord_ids"] >= 0) & (seq["chord_ids"] <= 23))) for seq in dataset)
    checks.append(("all dataset chord_ids within 0..23", chord_range_ok))

    key_range_ok = all(bool(np.all((seq["y"] >= 0) & (seq["y"] <= 23))) for seq in dataset)
    checks.append(("all dataset key labels within 0..23", key_range_ok))

    ambig_ok = all(
        len(seq["is_ambiguous"]) == len(seq["chord_ids"]) and seq["is_ambiguous"].dtype == np.bool_
        for seq in dataset
    )
    checks.append(("all dataset is_ambiguous masks correct length + dtype", ambig_ok))

    no_mod_count = sum(1 for seq in dataset if seq["metadata"]["type"] == "no_modulation")
    mod_count = sum(1 for seq in dataset if seq["metadata"]["type"] == "modulation")
    checks.append(("dataset includes both no_modulation and modulation sequences", no_mod_count > 0 and mod_count > 0))

    pivot_switch_ok = True
    for seq in dataset:
        if seq["metadata"]["type"] != "modulation":
            continue
        p = seq["pivot_idx"]
        key_a = seq["metadata"]["a_key_id"]
        key_b = seq["metadata"]["b_key_id"]
        if not (np.all(seq["y"][:p] == key_a) and np.all(seq["y"][p:] == key_b)):
            pivot_switch_ok = False
    checks.append(("all modulation sequences: pivot labels switch correctly", pivot_switch_ok))

    print("sequence_dataset.py verification")
    print("-" * 50)
    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")
    print("-" * 50)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
