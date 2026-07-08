"""mlp_baseline.py

Reusable copy of the notebook's feed-forward MLP + hand-coded hidden-state
leaky-integration ("EMA") baseline, extracted from
02_Baseline_Pipeline/Mini Capstone_Project_A_walking_machine_of_the_music.ipynb
(cells: imports/seed, ChordToKeyMLP, train, sequence_key_tracking).

This module exists so the future SRN comparison in
04_Recurrent_Implementation/srn_model.py + a later comparison script can call
the exact same baseline the notebook produces, without importing from or
editing the notebook itself. The model architecture and the leaky-integration
formula are copied as-is -- not improved, not retuned. This is a freeze of
the COGS 202 baseline, not a redesign.

Imports the frozen chord/key vocabulary from shared_music_defs.py and (for
the verification block only) the synthetic labeled sequence generator from
sequence_dataset.py.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared_music_defs import (
    key_triad_set,
    MAJ_WEIGHTS,
    MIN_WEIGHTS,
    one_hot,
)

DEFAULT_SEED = 269


# ---------------------------------------------------------------------------
# Deterministic setup
# (notebook cell 4)
# ---------------------------------------------------------------------------

def set_seed(seed: int = DEFAULT_SEED):
    """Seed random, numpy, and torch, matching the notebook's setup cell."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# ChordToKeyMLP
# (notebook cell 23)
#
# Chord -> Hidden -> Key
# Linear(24 -> 48) -> ReLU -> Linear(48 -> 24)
# ---------------------------------------------------------------------------

class ChordToKeyMLP(nn.Module):
    def __init__(self, in_dim=24, hid_dim=48, out_dim=24):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x, return_hidden=False):
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        if return_hidden:
            return logits, h
        return logits


# ---------------------------------------------------------------------------
# Synthetic single-chord training dataset
# (notebook cells 19, 21 -- sample_chord_from_key / make_dataset)
#
# NOTE: this is the single-chord "which key would sample this chord" task
# the MLP itself is trained on -- distinct from sequence_dataset.py's
# per-timestep sequence-labeling task, which is for evaluating the trained
# MLP (via sequence_key_tracking) and, later, the SRN.
# ---------------------------------------------------------------------------

def _sample_chord_from_key(key_id: int) -> int:
    triads = key_triad_set(key_id)
    w = MAJ_WEIGHTS if key_id < 12 else MIN_WEIGHTS
    return int(np.random.choice(triads, p=w / w.sum()))


def make_single_chord_dataset(n_samples=60000):
    """
    Reproduces the notebook's make_dataset(): X = one-hot chord (N, 24),
    y = key id (N,), keys sampled uniformly, chords sampled diatonically
    within each key.
    """
    X = np.zeros((n_samples, 24), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.int64)

    for i in range(n_samples):
        k = np.random.randint(0, 24)
        c = _sample_chord_from_key(k)
        X[i] = one_hot(24, c)
        y[i] = k
    return X, y


# ---------------------------------------------------------------------------
# Training
# (notebook cells 25)
# ---------------------------------------------------------------------------

def accuracy(logits, y_true):
    preds = logits.argmax(dim=1)
    return (preds == y_true).float().mean().item()

def train_mlp(model, X_train_t, y_train_t, X_val_t, y_val_t, opt, epochs=12, batch_size=256, verbose=True):
    """
    Trains `model` in place using cross-entropy loss and the given
    optimizer. Mirrors the notebook's train() loop exactly (same batching,
    same per-epoch print format).
    """
    device = X_train_t.device
    n = X_train_t.shape[0]

    for ep in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n, device=device)
        total_loss = 0.0

        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            xb = X_train_t[batch]
            yb = y_train_t[batch]

            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = F.cross_entropy(val_logits, y_val_t).item()
            val_acc = accuracy(val_logits, y_val_t)

        if verbose:
            print(f"Epoch {ep:02d} | train loss {total_loss/n:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.3f}")

    return model


def train_default_mlp(seed: int = DEFAULT_SEED, n_samples=60000, epochs=12, hid_dim=48, verbose=True):
    """
    Convenience function: seeds everything, builds the synthetic
    single-chord dataset, instantiates ChordToKeyMLP, and trains it --
    end to end, matching the notebook's cells 4, 21, 23, 25 in order.

    Returns (model, device).
    """
    set_seed(seed)
    device = get_device()

    X, y = make_single_chord_dataset(n_samples)

    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    split = int(0.9 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    X_train_t = torch.tensor(X_train).to(device)
    y_train_t = torch.tensor(y_train).to(device)
    X_val_t = torch.tensor(X_val).to(device)
    y_val_t = torch.tensor(y_val).to(device)

    model = ChordToKeyMLP(hid_dim=hid_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_mlp(model, X_train_t, y_train_t, X_val_t, y_val_t, opt, epochs=epochs, verbose=verbose)

    return model, device


# ---------------------------------------------------------------------------
# EMA / leaky hidden-state baseline
# (notebook cell 31: sequence_key_tracking)
#
# Formula copied as-is -- do not change:
#   h_in = relu(linear(x, W1, b1))          instantaneous hidden state
#   h    = (1 - alpha) * h + alpha * h_in   leaky integration ("EMA")
#   logits = linear(h, W2, b2)
#   probs  = softmax(logits)
#
# The leaky integration happens on the hidden state h, not on the output
# probabilities -- that distinction is load-bearing and must not change.
# ---------------------------------------------------------------------------

@torch.no_grad()
def sequence_key_tracking(model, chord_ids, alpha=0.20, device=None):
    """
    alpha: context update rate (smaller = more memory).
    Returns key probability per timestep: shape (T, 24) numpy array.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    W1, b1 = model.fc1.weight, model.fc1.bias
    W2, b2 = model.fc2.weight, model.fc2.bias

    h = torch.zeros((1, W1.shape[0]), device=device)  # hidden dim
    all_probs = []

    for c in chord_ids:
        x = torch.tensor(one_hot(24, c), device=device).unsqueeze(0)
        h_in = F.relu(F.linear(x, W1, b1))       # instantaneous hidden
        h = (1 - alpha) * h + alpha * h_in        # leaky integrate
        logits = F.linear(h, W2, b2)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        all_probs.append(probs)

    return np.stack(all_probs, axis=0)


def predict_sequence_probs(model, chord_ids, alpha=0.20, device=None):
    """Thin, explicitly-named alias for sequence_key_tracking."""
    return sequence_key_tracking(model, chord_ids, alpha=alpha, device=device)


# ---------------------------------------------------------------------------
# Verification block
#
# Sanity-checks that this module reproduces the notebook's baseline
# behavior. Not a figure-generation or comparison script -- run this file
# directly (`python mlp_baseline.py`) to re-check at any time. No SRN,
# comparison, or figure-regeneration code lives here.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sequence_dataset import make_modulation_sequence
    from shared_music_defs import key_index

    checks = []

    print("Training default MLP (seed=269, epochs=12)...")
    model, device = train_default_mlp(seed=DEFAULT_SEED, epochs=12, verbose=True)
    checks.append(("model trained without error", True))

    # C major -> G major modulation sequence (small, for a quick smoke test)
    key_c_maj = key_index(0, "maj")
    key_g_maj = key_index(7, "maj")
    seq = make_modulation_sequence(key_c_maj, key_g_maj, len_a=8, len_b=8)
    chord_ids = seq["chord_ids"]
    T = len(chord_ids)

    probs = sequence_key_tracking(model, chord_ids, alpha=0.20, device=device)

    checks.append(("output shape == (T, 24)", probs.shape == (T, 24)))
    checks.append(("probabilities sum to ~1 per timestep", bool(np.allclose(probs.sum(axis=1), 1.0, atol=1e-4))))
    checks.append(("no NaNs in output", bool(not np.isnan(probs).any())))
    checks.append(("alpha=0.20 produces a valid (T,24) probability array", probs.shape == (T, 24) and np.isfinite(probs).all()))

    # alpha=0.10 vs alpha=0.50 should give different trajectories (more vs
    # less memory), since the leaky-integration rate directly changes h.
    probs_low_alpha = sequence_key_tracking(model, chord_ids, alpha=0.10, device=device)
    probs_high_alpha = sequence_key_tracking(model, chord_ids, alpha=0.50, device=device)
    checks.append(("alpha=0.10 vs alpha=0.50 outputs differ", bool(not np.allclose(probs_low_alpha, probs_high_alpha))))

    print()
    print("mlp_baseline.py verification")
    print("-" * 50)
    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")
    print("-" * 50)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
