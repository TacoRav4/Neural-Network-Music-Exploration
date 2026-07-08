"""run_comparison.py

Phase 1 synthetic-only comparison: EMA+MLP baseline (mlp_baseline.py) vs.
Elman SRN (srn_model.py), both consuming the same 24-dim one-hot
chord/triad input, evaluated on synthetic labeled sequences from
sequence_dataset.py only.

Explicitly out of scope here (per workspace guardrails, see ../STATUS.md):
- No MIDI or raw chroma input -- Twinkle.mid / Twinkle 12.mid are not
  evaluated in this script. That is a separate, later step once the
  synthetic comparison itself is reviewed.
- No plotting -- that is plotting_comparison.py, not yet created.
- No changes to shared_music_defs.py, sequence_dataset.py, mlp_baseline.py,
  or srn_model.py; this script only imports and calls them.

Run from either this directory or the workspace root:
    python run_comparison.py
    python 04_Recurrent_Implementation/run_comparison.py
"""

import contextlib
import io
import json
import math
import os
import re
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_FIGURES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "05_Figures_Results"))

from sequence_dataset import make_modulation_dataset
from mlp_baseline import set_seed, DEFAULT_SEED, get_device, train_default_mlp, sequence_key_tracking
from srn_model import ElmanKeySRN, train_srn, predict_sequence_probs

SEED = DEFAULT_SEED  # 269, matches the notebook's fixed seed throughout this workspace

TRAIN_N = 400
VAL_N = 90
TEST_N = 100
LENGTH_RANGE = (16, 48)

ALPHA_VALUES = [0.10, 0.20, 0.35, 0.50]
PRIMARY_ALPHA = 0.20

SRN_HIDDEN_SIZE = 48
SRN_EPOCHS = 10
SRN_LR = 1e-2

MLP_EPOCHS = 12
MLP_N_SAMPLES = 60000

JSON_OUT = os.path.join(_FIGURES_DIR, "EMA_vs_SRN_synthetic_comparison_metrics.json")
MD_OUT = os.path.join(_FIGURES_DIR, "EMA_vs_SRN_synthetic_comparison_report.md")


# ---------------------------------------------------------------------------
# Dataset construction
#
# Three independent, deterministic splits derived from the base SEED=269
# (train uses 269 itself; val/test use fixed offsets of it) so the splits
# don't overlap in content but the whole run is reproducible end to end.
# ---------------------------------------------------------------------------

def build_datasets():
    train_seed = SEED
    val_seed = SEED + 1000
    test_seed = SEED + 2000

    train_sequences = make_modulation_dataset(
        n_sequences=TRAIN_N, length_range=LENGTH_RANGE, include_no_modulation=True, rng_seed=train_seed
    )
    val_sequences = make_modulation_dataset(
        n_sequences=VAL_N, length_range=LENGTH_RANGE, include_no_modulation=True, rng_seed=val_seed
    )
    test_sequences = make_modulation_dataset(
        n_sequences=TEST_N, length_range=LENGTH_RANGE, include_no_modulation=True, rng_seed=test_seed
    )

    dataset_info = {
        "seed_base": SEED,
        "train_seed": train_seed,
        "val_seed": val_seed,
        "test_seed": test_seed,
        "train_size": len(train_sequences),
        "val_size": len(val_sequences),
        "test_size": len(test_sequences),
        "length_range": list(LENGTH_RANGE),
        "include_no_modulation": True,
    }
    return train_sequences, val_sequences, test_sequences, dataset_info


# ---------------------------------------------------------------------------
# Metrics
#
# All accuracy fields are either a python float in [0,1] or None (never
# NaN) when the underlying count is zero (e.g. no ambiguous timesteps
# happened to occur in the test set) -- the count is always reported
# alongside so "None" is distinguishable from "computed and exactly 0".
# ---------------------------------------------------------------------------

def _acc(correct: int, count: int):
    return (float(correct) / float(count)) if count > 0 else None


def modulation_lag_for_sequence(preds, y, pivot_idx):
    """
    pivot_idx: first timestep labeled key_b.
    true post-pivot key = y[pivot_idx].
    Find the first t >= pivot_idx where preds[t] == true_post_key AND
    preds[t+1] == true_post_key (stays switched for >= 2 consecutive
    timesteps). Returns (lag_or_None, switched_bool).
    """
    true_post_key = y[pivot_idx]
    T = len(y)
    for t in range(pivot_idx, T - 1):
        if preds[t] == true_post_key and preds[t + 1] == true_post_key:
            return t - pivot_idx, True
    return None, False


def evaluate_model(dataset, predict_fn):
    """
    dataset: list of sequence dicts (from sequence_dataset.py).
    predict_fn: seq_dict -> (T, 24) probability array.

    Returns a metrics dict (see requirements 4 and 5 in the task spec).
    """
    overall_correct = overall_count = 0
    masked_correct = masked_count = 0          # excludes ambiguous timesteps
    ambig_correct = ambig_count = 0             # ambiguous timesteps only
    mod_correct = mod_count = 0                 # modulation sequences, all timesteps
    nomod_correct = nomod_count = 0              # no-modulation sequences, all timesteps
    pre_correct = pre_count = 0                  # modulation sequences, pre-pivot timesteps
    post_correct = post_count = 0                # modulation sequences, post-pivot timesteps
    pivot_correct = pivot_count = 0              # modulation sequences, exact pivot timestep only

    lags = []
    n_switch_fail = 0
    n_modulation_seqs = 0

    for seq in dataset:
        probs = predict_fn(seq)
        preds = np.argmax(probs, axis=1)
        y = np.asarray(seq["y"])
        is_amb = np.asarray(seq["is_ambiguous"], dtype=bool)
        correct = (preds == y)
        T = len(y)

        overall_correct += int(correct.sum())
        overall_count += T

        non_amb = ~is_amb
        masked_correct += int(correct[non_amb].sum())
        masked_count += int(non_amb.sum())

        if is_amb.sum() > 0:
            ambig_correct += int(correct[is_amb].sum())
            ambig_count += int(is_amb.sum())

        if seq["metadata"]["type"] == "modulation":
            n_modulation_seqs += 1
            mod_correct += int(correct.sum())
            mod_count += T

            pivot_idx = seq["pivot_idx"]
            pre_correct += int(correct[:pivot_idx].sum())
            pre_count += pivot_idx
            post_correct += int(correct[pivot_idx:].sum())
            post_count += (T - pivot_idx)
            pivot_correct += int(correct[pivot_idx])
            pivot_count += 1

            lag, switched = modulation_lag_for_sequence(preds, y, pivot_idx)
            if switched:
                lags.append(lag)
            else:
                n_switch_fail += 1
        else:
            nomod_correct += int(correct.sum())
            nomod_count += T

    metrics = {
        "n_sequences": len(dataset),
        "n_timesteps": overall_count,
        "overall_accuracy": _acc(overall_correct, overall_count),
        "masked_accuracy_excl_ambiguous": _acc(masked_correct, masked_count),
        "ambiguous_timestep_accuracy": _acc(ambig_correct, ambig_count),
        "ambiguous_timestep_count": ambig_count,
        "modulation_accuracy": _acc(mod_correct, mod_count),
        "no_modulation_accuracy": _acc(nomod_correct, nomod_count),
        "pre_pivot_accuracy": _acc(pre_correct, pre_count),
        "post_pivot_accuracy": _acc(post_correct, post_count),
        "pivot_timestep_accuracy": _acc(pivot_correct, pivot_count),
        "modulation_lag": {
            "n_modulation_sequences": n_modulation_seqs,
            "n_switch_success": len(lags),
            "n_switch_failure": n_switch_fail,
            "switch_failure_rate": _acc(n_switch_fail, n_modulation_seqs),
            "mean_lag": float(np.mean(lags)) if lags else None,
            "median_lag": float(np.median(lags)) if lags else None,
            "min_lag": float(np.min(lags)) if lags else None,
            "max_lag": float(np.max(lags)) if lags else None,
        },
    }
    return metrics


# ---------------------------------------------------------------------------
# Training log capture
#
# mlp_baseline.train_mlp / srn_model.train_srn print per-epoch progress
# (matching the notebook's style); we capture stdout during training so we
# can (a) still show it live to the user by re-printing it, and (b) parse
# the final epoch's numbers into the JSON/markdown report without touching
# either module's return signature.
# ---------------------------------------------------------------------------

def _run_capturing_stdout(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    log = buf.getvalue()
    print(log, end="")
    return result, log


def _parse_last_epoch_line(log: str) -> dict:
    lines = [l for l in log.strip().split("\n") if l.startswith("Epoch")]
    if not lines:
        return {}
    last = lines[-1]
    out = {}
    for key, pattern in [
        ("train_loss", r"train loss ([\d.]+)"),
        ("train_acc", r"train acc ([\d.]+)"),
        ("val_loss", r"val loss ([\d.]+)"),
        ("val_acc", r"val acc ([\d.]+)"),
    ]:
        m = re.search(pattern, last)
        if m:
            out[key] = float(m.group(1))
    return out


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt(x, digits=4):
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _metrics_table_row(name, m):
    lag = m["modulation_lag"]
    return (
        f"| {name} "
        f"| {_fmt(m['overall_accuracy'])} "
        f"| {_fmt(m['masked_accuracy_excl_ambiguous'])} "
        f"| {_fmt(m['modulation_accuracy'])} "
        f"| {_fmt(m['no_modulation_accuracy'])} "
        f"| {_fmt(m['pre_pivot_accuracy'])} "
        f"| {_fmt(m['post_pivot_accuracy'])} "
        f"| {_fmt(m['pivot_timestep_accuracy'])} "
        f"| {_fmt(m['ambiguous_timestep_accuracy'])} (n={m['ambiguous_timestep_count']}) "
        f"| {_fmt(lag['mean_lag'], 2)} "
        f"| {_fmt(lag['switch_failure_rate'])} |"
    )


def build_report_md(results: dict) -> str:
    ds = results["dataset"]
    base = results["baseline"]
    srn = results["srn"]

    lines = []
    lines.append("# EMA+MLP vs. Elman SRN — Synthetic Comparison Report (Phase 1)")
    lines.append("")
    lines.append(
        "Phase 1, synthetic-only comparison between the notebook's hand-coded EMA/leaky-integration "
        "baseline (`mlp_baseline.sequence_key_tracking`) and a learned Elman SRN (`srn_model.ElmanKeySRN`), "
        "both consuming the same 24-dim one-hot chord/triad input. "
        "**This run does not evaluate Twinkle.mid or Twinkle 12.mid** -- it uses only the synthetic "
        "labeled modulation sequences generated by `sequence_dataset.py`. Real-MIDI evaluation is a "
        "separate, later step."
    )
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Base seed: `{ds['seed_base']}`")
    lines.append(f"- Train sequences: {ds['train_size']} (rng_seed={ds['train_seed']})")
    lines.append(f"- Val sequences: {ds['val_size']} (rng_seed={ds['val_seed']})")
    lines.append(f"- Test sequences: {ds['test_size']} (rng_seed={ds['test_seed']})")
    lines.append(f"- Sequence length range (per segment): {tuple(ds['length_range'])}")
    lines.append(f"- Includes no-modulation negative controls: {ds['include_no_modulation']}")
    lines.append(
        "- All metrics below are computed on the held-out **test** split only, for both models."
    )
    lines.append("")

    lines.append("## EMA+MLP baseline")
    lines.append("")
    mlp_train = base["mlp_training"]
    lines.append(
        f"- MLP training: seed={mlp_train['seed']}, epochs={mlp_train['epochs']}, "
        f"n_samples={mlp_train['n_samples']}, final val loss={_fmt(mlp_train.get('final_val_loss'))}, "
        f"final val acc={_fmt(mlp_train.get('final_val_acc'))}"
    )
    lines.append(
        f"- Primary alpha (leaky-integration rate): **{base['primary_alpha']}** "
        "(fixed, matches the documented true-Twinkle baseline run notes; not tuned here)"
    )
    lines.append("")
    lines.append(
        "### Alpha sweep (diagnostic only -- reported as-is, not tuned/selected)"
    )
    lines.append("")
    lines.append("| alpha | overall_accuracy | modulation_accuracy | mean_lag | switch_failure_rate |")
    lines.append("|---|---|---|---|---|")
    for a_str, m in base["alpha_sweep"].items():
        lag = m["modulation_lag"]
        lines.append(
            f"| {a_str} | {_fmt(m['overall_accuracy'])} | {_fmt(m['modulation_accuracy'])} "
            f"| {_fmt(lag['mean_lag'], 2)} | {_fmt(lag['switch_failure_rate'])} |"
        )
    lines.append("")

    lines.append("## SRN")
    lines.append("")
    srn_train = srn["training_settings"]
    lines.append(
        f"- Training: seed={srn_train['seed']}, epochs={srn_train['epochs']}, lr={srn_train['lr']}, "
        f"hidden_size={srn_train['hidden_size']}, "
        f"final train loss={_fmt(srn_train.get('final_train_loss'))}, "
        f"final train acc={_fmt(srn_train.get('final_train_acc'))}, "
        f"final val loss={_fmt(srn_train.get('final_val_loss'))}, "
        f"final val acc={_fmt(srn_train.get('final_val_acc'))}"
    )
    lines.append(
        "- This is a modest first-run epoch count for an initial comparison, not a converged/tuned model."
    )
    lines.append("")

    lines.append("## Summary: test-set metrics (primary alpha vs. SRN)")
    lines.append("")
    lines.append(
        "| model | overall_acc | masked_acc (excl. ambig.) | modulation_acc | no_mod_acc | "
        "pre_pivot_acc | post_pivot_acc | pivot_timestep_acc | ambiguous_acc (n) | mean_lag | switch_fail_rate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    lines.append(_metrics_table_row("EMA+MLP (alpha=" + str(base["primary_alpha"]) + ")", base["primary_metrics"]))
    lines.append(_metrics_table_row("Elman SRN", srn["metrics"]))
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(build_interpretation(base["primary_metrics"], srn["metrics"]))
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This is a **synthetic-only Phase 1** result. It has not yet evaluated `Twinkle.mid` or "
        "`Twinkle 12.mid`, and does not include probability-tracking or Circle-of-Fifths plots "
        "(`plotting_comparison.py` is not yet created). Per the workspace guardrails, phase 1 "
        "deliberately keeps the input representation identical between both models (24-dim one-hot "
        "chord/triad) so the only variable under test is hand-coded EMA vs. learned recurrence; a "
        "raw-chroma end-to-end SRN would be a separate phase 2 experiment."
    )
    lines.append("")

    return "\n".join(lines)


def build_interpretation(base_m: dict, srn_m: dict) -> str:
    parts = []

    b_overall, s_overall = base_m["overall_accuracy"], srn_m["overall_accuracy"]
    if b_overall is not None and s_overall is not None:
        better = "the SRN" if s_overall > b_overall else ("the EMA+MLP baseline" if b_overall > s_overall else "neither model (tied)")
        parts.append(
            f"On overall per-timestep accuracy, {better} scores higher on this synthetic test set "
            f"(EMA+MLP={_fmt(b_overall)}, SRN={_fmt(s_overall)})."
        )

    b_lag, s_lag = base_m["modulation_lag"]["mean_lag"], srn_m["modulation_lag"]["mean_lag"]
    if b_lag is not None and s_lag is not None:
        faster = "the SRN" if s_lag < b_lag else ("the EMA+MLP baseline" if b_lag < s_lag else "neither model (tied)")
        parts.append(
            f"On modulation lag (timesteps between the true pivot and a stable switch to the new key), "
            f"{faster} switches faster on average (EMA+MLP mean_lag={_fmt(b_lag, 2)}, SRN mean_lag={_fmt(s_lag, 2)})."
        )
    else:
        parts.append(
            "Modulation lag could not be compared directly because one or both models never produced "
            "a successful stable switch on any test modulation sequence (see switch_failure_rate)."
        )

    b_fail, s_fail = base_m["modulation_lag"]["switch_failure_rate"], srn_m["modulation_lag"]["switch_failure_rate"]
    if b_fail is not None and s_fail is not None:
        parts.append(
            f"Switch failure rate (fraction of modulation sequences where the model never stably "
            f"switched to the post-pivot key): EMA+MLP={_fmt(b_fail)}, SRN={_fmt(s_fail)}."
        )

    parts.append(
        "This is a first-run, modestly-trained SRN (see training settings above) compared against an "
        "untuned, fixed-alpha baseline -- these numbers describe this specific run, not a claim about "
        "the SRN architecture's ceiling. Reviewing these results (and iterating on SRN training/epochs "
        "if needed) should happen before adding plots or moving to real-MIDI evaluation."
    )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _scan_for_nan(obj, path="root"):
    """Recursively scan a JSON-able structure for float NaN. Returns list of offending paths."""
    bad = []
    if isinstance(obj, float) and math.isnan(obj):
        bad.append(path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            bad.extend(_scan_for_nan(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad.extend(_scan_for_nan(v, f"{path}[{i}]"))
    return bad


def run_verification(results: dict, json_path: str, md_path: str):
    checks = []

    checks.append(("output JSON exists", os.path.exists(json_path)))
    checks.append(("output markdown report exists", os.path.exists(md_path)))

    checks.append(("results contains 'baseline' key", "baseline" in results))
    checks.append(("results contains 'srn' key", "srn" in results))

    with open(md_path) as f:
        md_text = f.read()
    checks.append(("markdown report mentions EMA", "EMA" in md_text))
    checks.append(("markdown report mentions SRN", "SRN" in md_text))

    nan_paths = _scan_for_nan(results)
    checks.append(("no NaNs anywhere in metrics (missing values are null, not NaN)", len(nan_paths) == 0))

    base_primary = results["baseline"]["primary_metrics"]
    srn_metrics = results["srn"]["metrics"]
    checks.append((
        "ambiguous-mask metrics present (value or explicit null) for baseline",
        "ambiguous_timestep_accuracy" in base_primary and "ambiguous_timestep_count" in base_primary,
    ))
    checks.append((
        "ambiguous-mask metrics present (value or explicit null) for SRN",
        "ambiguous_timestep_accuracy" in srn_metrics and "ambiguous_timestep_count" in srn_metrics,
    ))

    checks.append(("script located under 04_Recurrent_Implementation/", os.path.basename(_THIS_DIR) == "04_Recurrent_Implementation"))

    print()
    print("run_comparison.py verification")
    print("-" * 50)
    all_passed = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {label}")
    print("-" * 50)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    if nan_paths:
        print("NaN found at:", nan_paths)

    return all_passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(_FIGURES_DIR, exist_ok=True)

    print("Building synthetic train/val/test datasets...")
    train_sequences, val_sequences, test_sequences, dataset_info = build_datasets()
    print(
        f"  train={dataset_info['train_size']} val={dataset_info['val_size']} "
        f"test={dataset_info['test_size']} length_range={dataset_info['length_range']}"
    )

    device = get_device()

    # --- EMA+MLP baseline ---
    print("\nTraining EMA+MLP baseline (ChordToKeyMLP)...")
    (model_mlp, mlp_device), mlp_log = _run_capturing_stdout(
        train_default_mlp, seed=SEED, n_samples=MLP_N_SAMPLES, epochs=MLP_EPOCHS, verbose=True
    )
    mlp_final = _parse_last_epoch_line(mlp_log)

    print("\nEvaluating EMA+MLP baseline on test set (alpha sweep)...")
    alpha_sweep = {}
    for a in ALPHA_VALUES:
        def predict_fn(seq, _alpha=a):
            return sequence_key_tracking(model_mlp, seq["chord_ids"], alpha=_alpha, device=mlp_device)
        alpha_sweep[f"{a:.2f}"] = evaluate_model(test_sequences, predict_fn)
        print(f"  alpha={a:.2f} -> overall_accuracy={_fmt(alpha_sweep[f'{a:.2f}']['overall_accuracy'])}")

    primary_key = f"{PRIMARY_ALPHA:.2f}"
    baseline_results = {
        "primary_alpha": PRIMARY_ALPHA,
        "primary_metrics": alpha_sweep[primary_key],
        "alpha_sweep": alpha_sweep,
        "mlp_training": {
            "seed": SEED,
            "epochs": MLP_EPOCHS,
            "n_samples": MLP_N_SAMPLES,
            "final_train_loss": mlp_final.get("train_loss"),
            "final_val_loss": mlp_final.get("val_loss"),
            "final_val_acc": mlp_final.get("val_acc"),
        },
    }

    # --- Elman SRN ---
    print("\nTraining Elman SRN...")
    set_seed(SEED)
    model_srn = ElmanKeySRN(input_size=24, hidden_size=SRN_HIDDEN_SIZE, output_size=24).to(device)

    model_srn, srn_log = _run_capturing_stdout(
        train_srn,
        model_srn,
        train_sequences,
        val_sequences=val_sequences,
        epochs=SRN_EPOCHS,
        lr=SRN_LR,
        seed=SEED,
        device=device,
        verbose=True,
    )
    srn_final = _parse_last_epoch_line(srn_log)

    print("\nEvaluating Elman SRN on test set...")
    def srn_predict_fn(seq):
        return predict_sequence_probs(model_srn, seq, device=device)
    srn_metrics = evaluate_model(test_sequences, srn_predict_fn)
    print(f"  overall_accuracy={_fmt(srn_metrics['overall_accuracy'])}")

    srn_results = {
        "training_settings": {
            "seed": SEED,
            "epochs": SRN_EPOCHS,
            "lr": SRN_LR,
            "hidden_size": SRN_HIDDEN_SIZE,
            "final_train_loss": srn_final.get("train_loss"),
            "final_train_acc": srn_final.get("train_acc"),
            "final_val_loss": srn_final.get("val_loss"),
            "final_val_acc": srn_final.get("val_acc"),
        },
        "metrics": srn_metrics,
    }

    # --- Assemble + write outputs ---
    results = {
        "phase": "phase_1_synthetic_only",
        "seed": SEED,
        "dataset": dataset_info,
        "baseline": baseline_results,
        "srn": srn_results,
        "notes": (
            "Synthetic-only Phase 1 comparison. Does not yet evaluate Twinkle.mid or Twinkle 12.mid. "
            "Both models use the same 24-dim one-hot chord/triad input representation."
        ),
    }

    with open(JSON_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {JSON_OUT}")

    report_md = build_report_md(results)
    with open(MD_OUT, "w") as f:
        f.write(report_md)
    print(f"Wrote {MD_OUT}")

    run_verification(results, JSON_OUT, MD_OUT)


if __name__ == "__main__":
    main()
