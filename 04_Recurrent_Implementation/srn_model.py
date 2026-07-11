"""srn_model.py

Phase 1 learned recurrent-memory model for key tracking: an Elman-style
simple recurrent network (SRN) whose hidden-state update is *learned*
(`W_ih`, `W_hh`) end-to-end on labeled chord sequences, instead of the
notebook's hand-coded leaky-integration blend (`alpha`) in
mlp_baseline.sequence_key_tracking.

Per the workspace guardrails (see ../STATUS.md section 8): this is phase 1
only. The SRN below consumes the exact same 24-dim one-hot chord/triad
vector the EMA+MLP baseline uses -- no representation change is bundled in
with the recurrence change. A raw-chroma end-to-end SRN would be a separate,
explicitly-labeled phase 2 experiment, not implemented here.

This module does not compare itself against the EMA+MLP baseline -- that is
run_comparison.py's job, not yet created. This module also does not import
from or edit the notebook; it builds on the frozen definitions in
shared_music_defs.py and the synthetic labeled sequences from
sequence_dataset.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sequence_dataset import make_modulation_dataset, sequence_to_onehot

# Reused, not modified: mlp_baseline.set_seed seeds random/numpy/torch
# exactly as the notebook's setup cell does. Importing it here keeps
# seeding behavior identical across the EMA baseline and the SRN without
# duplicating (and risking drifting from) that logic.
from mlp_baseline import set_seed, DEFAULT_SEED, get_device


# ---------------------------------------------------------------------------
# ElmanKeySRN
#
# Recurrent update (the learned replacement for the hand-coded leaky blend):
#   h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)
# Output:
#   logits_t = W_ho @ h_t + b_o
#
# Implemented manually with nn.Linear layers (rather than nn.RNNCell) so the
# correspondence to sequence_key_tracking's h = (1-alpha)*h + alpha*h_in
# formula is visually obvious: same "blend previous hidden state with a
# function of the current input" shape, but the blend itself (W_hh, and the
# tanh nonlinearity in place of the fixed convex combination) is learned
# rather than a fixed scalar.
# ---------------------------------------------------------------------------

class ElmanKeySRN(nn.Module):
    def __init__(self, input_size=24, hidden_size=48, output_size=24):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_ih = nn.Linear(input_size, hidden_size, bias=True)   # x_t -> hidden, carries +b
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)  # h_{t-1} -> hidden
        self.W_ho = nn.Linear(hidden_size, output_size, bias=True)   # h_t -> output logits

    def forward(self, x, h0=None, return_hidden=False):
        """
        x: (T, input_size) or (B, T, input_size)
        Returns logits: (T, output_size) or (B, T, output_size)
        If return_hidden=True, also returns hidden states of matching shape
        (T, hidden_size) or (B, T, hidden_size).
        """
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, T, input_size)
            squeeze_batch = True
        elif x.dim() != 3:
            raise ValueError(f"expected x of shape (T, input) or (B, T, input), got {tuple(x.shape)}")

        B, T, _ = x.shape
        device = x.device

        h = torch.zeros(B, self.hidden_size, device=device) if h0 is None else h0

        hiddens = []
        logits_steps = []
        for t in range(T):
            x_t = x[:, t, :]
            h = torch.tanh(self.W_ih(x_t) + self.W_hh(h))
            logits_t = self.W_ho(h)
            hiddens.append(h)
            logits_steps.append(logits_t)

        logits = torch.stack(logits_steps, dim=1)   # (B, T, output_size)
        hiddens = torch.stack(hiddens, dim=1)        # (B, T, hidden_size)

        if squeeze_batch:
            logits = logits.squeeze(0)
            hiddens = hiddens.squeeze(0)

        if return_hidden:
            return logits, hiddens
        return logits


# ---------------------------------------------------------------------------
# Training helper
#
# Sequences are variable-length (see sequence_dataset.py), so this loops
# over one sequence at a time per optimizer step (effective batch size 1)
# rather than padding/masking a batched tensor -- simplest correct approach
# for phase 1, and avoids introducing padding logic before it's needed.
# BPTT is over each full sequence (no truncation) since these sequences are
# short (tens of timesteps).
# ---------------------------------------------------------------------------

def _sequence_loss_and_correct(model, seq, device):
    x = torch.tensor(seq["x"], dtype=torch.float32, device=device)
    y = torch.tensor(seq["y"], dtype=torch.int64, device=device)
    logits = model(x)  # (T, output_size)
    loss = F.cross_entropy(logits, y)
    correct = (logits.argmax(dim=1) == y).sum().item()
    return loss, correct, len(y)


def train_srn(model, train_sequences, val_sequences=None, epochs=50, lr=1e-2, seed=DEFAULT_SEED, device=None, verbose=True):
    """
    Trains `model` in place with per-timestep cross-entropy loss, Adam,
    and full-sequence BPTT (one sequence per optimizer step).

    train_sequences / val_sequences: lists of sequence dicts as returned by
    sequence_dataset.make_modulation_dataset (or make_sequence_in_key /
    make_modulation_sequence / make_no_modulation_sequence).

    Returns the trained model.
    """
    set_seed(seed)
    if device is None:
        device = next(model.parameters()).device
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, total_correct, total_count = 0.0, 0, 0

        for seq in train_sequences:
            opt.zero_grad()
            loss, correct, n = _sequence_loss_and_correct(model, seq, device)
            loss.backward()
            opt.step()

            total_loss += loss.item() * n
            total_correct += correct
            total_count += n

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count
        msg = f"Epoch {ep:02d} | train loss {train_loss:.4f} | train acc {train_acc:.3f}"

        if val_sequences:
            model.eval()
            v_loss, v_correct, v_count = 0.0, 0, 0
            with torch.no_grad():
                for seq in val_sequences:
                    loss, correct, n = _sequence_loss_and_correct(model, seq, device)
                    v_loss += loss.item() * n
                    v_correct += correct
                    v_count += n
            val_loss = v_loss / v_count
            val_acc = v_correct / v_count
            msg += f" | val loss {val_loss:.4f} | val acc {val_acc:.3f}"

        if verbose:
            print(msg)

    return model


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict_sequence_probs(model, x_or_sequence, device=None):
    """
    Accepts a sequence dict (as from sequence_dataset.py, uses its "x"
    field), a raw chord_ids array/list (converted via sequence_to_onehot),
    or an already-one-hot (T, 24) array.

    Returns softmax key probabilities, shape (T, 24), numpy array.
    """
    if device is None:
        device = next(model.parameters()).device

    if isinstance(x_or_sequence, dict):
        x = x_or_sequence["x"]
    else:
        x = np.asarray(x_or_sequence)
        if x.ndim == 1:
            x = sequence_to_onehot(x)

    x_t = torch.tensor(np.asarray(x), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(x_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    return probs


# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------

def sequence_accuracy(logits_or_probs, y, mask=None):
    """
    logits_or_probs: (T, 24) torch tensor or numpy array (raw logits or
        softmax probabilities -- argmax is invariant to softmax, so either
        works).
    y: (T,) array-like of true key labels.
    mask: optional (T,) boolean array/tensor; True = include this timestep
        in the accuracy computation. Use this to exclude ambiguous pivot
        timesteps (mask = ~is_ambiguous) once that comparison is needed --
        this function itself does not know about ambiguity, it just applies
        whatever mask it's given.

    Returns a python float accuracy (NaN if mask excludes every timestep).
    """
    arr = logits_or_probs
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    y = np.asarray(y)

    preds = arr.argmax(axis=-1)
    correct = (preds == y)

    if mask is not None:
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return float("nan")
        return float(correct[mask].mean())

    return float(correct.mean())


# ---------------------------------------------------------------------------
# Verification block
#
# Sanity-checks the SRN's shapes, training loop, and helper functions in
# isolation. Deliberately does NOT compare against the EMA+MLP baseline --
# that comparison belongs in run_comparison.py, not yet created. Run this
# file directly (`python srn_model.py`) to re-check at any time.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    checks = []

    set_seed(DEFAULT_SEED)
    device = get_device()

    # --- tiny synthetic dataset ---
    dataset = make_modulation_dataset(n_sequences=8, length_range=(6, 10), rng_seed=DEFAULT_SEED)
    train_sequences = dataset[:5]
    val_sequences = dataset[5:]

    # --- model init + single forward pass (unbatched, (T, 24)) ---
    model = ElmanKeySRN(input_size=24, hidden_size=48, output_size=24).to(device)

    seq0 = train_sequences[0]
    x0 = torch.tensor(seq0["x"], dtype=torch.float32, device=device)
    T0 = x0.shape[0]

    logits = model(x0)
    checks.append(("forward (T,24) input -> logits shape (T,24)", tuple(logits.shape) == (T0, 24)))

    logits_h, hiddens = model(x0, return_hidden=True)
    checks.append(("return_hidden=True -> hidden shape (T,48)", tuple(hiddens.shape) == (T0, 48)))
    checks.append(("return_hidden=True -> logits shape still (T,24)", tuple(logits_h.shape) == (T0, 24)))

    # --- batched forward pass (B, T, 24), using two sequences of equal length ---
    equal_len_seqs = [s for s in dataset if len(s["chord_ids"]) == T0]
    if len(equal_len_seqs) >= 2:
        x_batch = torch.tensor(
            np.stack([equal_len_seqs[0]["x"], equal_len_seqs[1]["x"]], axis=0),
            dtype=torch.float32, device=device,
        )
        logits_batch, hiddens_batch = model(x_batch, return_hidden=True)
        checks.append(("batched forward -> logits shape (B,T,24)", tuple(logits_batch.shape) == (2, T0, 24)))
        checks.append(("batched forward -> hidden shape (B,T,48)", tuple(hiddens_batch.shape) == (2, T0, 48)))
    else:
        # Not every random tiny dataset guarantees two same-length sequences;
        # fall back to an explicit synthetic batch so the batched path is
        # still exercised.
        x_batch = x0.unsqueeze(0).repeat(2, 1, 1)
        logits_batch, hiddens_batch = model(x_batch, return_hidden=True)
        checks.append(("batched forward -> logits shape (B,T,24)", tuple(logits_batch.shape) == (2, T0, 24)))
        checks.append(("batched forward -> hidden shape (B,T,48)", tuple(hiddens_batch.shape) == (2, T0, 48)))

    # --- training loop runs without error on a tiny dataset ---
    print("Training tiny ElmanKeySRN for 3 epochs (smoke test only)...")
    model = train_srn(model, train_sequences, val_sequences=val_sequences, epochs=3, lr=1e-2, seed=DEFAULT_SEED, device=device, verbose=True)
    checks.append(("train_srn ran without error", True))

    # --- predict_sequence_probs ---
    probs = predict_sequence_probs(model, seq0, device=device)
    checks.append(("predict_sequence_probs shape == (T,24)", probs.shape == (T0, 24)))
    checks.append(("predict_sequence_probs rows sum to ~1", bool(np.allclose(probs.sum(axis=1), 1.0, atol=1e-4))))
    checks.append(("predict_sequence_probs has no NaNs", bool(not np.isnan(probs).any())))

    # predict_sequence_probs also accepts raw chord_ids and raw one-hot x
    probs_from_ids = predict_sequence_probs(model, seq0["chord_ids"], device=device)
    checks.append(("predict_sequence_probs works from raw chord_ids", probs_from_ids.shape == (T0, 24)))
    probs_from_x = predict_sequence_probs(model, seq0["x"], device=device)
    checks.append(("predict_sequence_probs works from raw one-hot x", bool(np.allclose(probs_from_ids, probs_from_x))))

    # --- sequence_accuracy ---
    acc = sequence_accuracy(probs, seq0["y"])
    checks.append(("sequence_accuracy returns a float in [0,1]", isinstance(acc, float) and 0.0 <= acc <= 1.0))

    mask = ~seq0["is_ambiguous"]
    acc_masked = sequence_accuracy(probs, seq0["y"], mask=mask)
    checks.append(("sequence_accuracy with mask returns a float", isinstance(acc_masked, float)))

    print()
    print("srn_model.py verification")
    print("-" * 50)
    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")
    print("-" * 50)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print()
    print("NOTE: this verification does not compare ElmanKeySRN against the")
    print("EMA+MLP baseline (mlp_baseline.sequence_key_tracking). That")
    print("comparison belongs in run_comparison.py, not yet created.")
