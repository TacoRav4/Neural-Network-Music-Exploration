"""diagnose_srn_training.py

Focused diagnostic sweep to answer one question: was the Elman SRN's
weaker showing in the first run_comparison.py run (overall_accuracy 0.636
vs. the EMA+MLP baseline's 0.753 at alpha=0.20, after only 10 training
epochs) due to undertraining, or a more structural limitation of the
architecture/setup?

Sweeps SRN epochs x learning rate only. hidden_size and architecture are
held fixed (48, unchanged Elman update) -- a hidden-size sweep is out of
scope here. No plotting, no MIDI/chroma, no raw-chroma SRN.

Reuses (imports only, does not modify) run_comparison.py's exact dataset
construction (build_datasets, same seed/sizes/length_range) and its
evaluate_model metrics function, so results here are directly comparable
to the EMA+MLP baseline reference loaded from the prior comparison run's
output JSON.

Run from either this directory or the workspace root:
    python diagnose_srn_training.py
    python 04_Recurrent_Implementation/diagnose_srn_training.py
"""

import json
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_FIGURES_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "05_Figures_Results"))

import run_comparison as rc  # reused, not modified: build_datasets, evaluate_model, log-capture/parse helpers
from mlp_baseline import set_seed, DEFAULT_SEED, get_device, train_default_mlp, sequence_key_tracking
from srn_model import ElmanKeySRN, train_srn

SEED = DEFAULT_SEED  # 269

EPOCH_VALUES = [10, 25, 50]
LR_VALUES = [1e-3, 3e-3]
HIDDEN_SIZE = 48  # fixed; no hidden-size sweep in this diagnostic

PRIMARY_ALPHA = 0.20
PREVIOUS_COMPARISON_JSON = os.path.join(_FIGURES_DIR, "EMA_vs_SRN_synthetic_comparison_metrics.json")

JSON_OUT = os.path.join(_FIGURES_DIR, "SRN_training_diagnostic_metrics.json")
MD_OUT = os.path.join(_FIGURES_DIR, "SRN_training_diagnostic_report.md")


# ---------------------------------------------------------------------------
# Baseline reference
#
# Prefer loading the EMA+MLP alpha=0.20 metrics from the previous
# run_comparison.py output (same seed, same dataset construction) rather
# than retraining -- avoids duplicating a training run that already exists
# and is already verified. Falls back to computing it fresh only if that
# file is missing, using the exact same dataset/train/eval path
# run_comparison.py uses.
# ---------------------------------------------------------------------------

def load_or_compute_baseline_reference(test_sequences, device):
    if os.path.exists(PREVIOUS_COMPARISON_JSON):
        with open(PREVIOUS_COMPARISON_JSON) as f:
            prev = json.load(f)
        return {
            "source": "loaded_from_previous_run_comparison_json",
            "source_path": os.path.relpath(PREVIOUS_COMPARISON_JSON, _FIGURES_DIR),
            "alpha": prev["baseline"]["primary_alpha"],
            "metrics": prev["baseline"]["primary_metrics"],
        }

    print("Previous comparison JSON not found -- computing EMA+MLP baseline reference fresh.")
    (model_mlp, mlp_device), _ = rc._run_capturing_stdout(
        train_default_mlp, seed=SEED, n_samples=rc.MLP_N_SAMPLES, epochs=rc.MLP_EPOCHS, verbose=True
    )

    def predict_fn(seq):
        return sequence_key_tracking(model_mlp, seq["chord_ids"], alpha=PRIMARY_ALPHA, device=mlp_device)

    metrics = rc.evaluate_model(test_sequences, predict_fn)
    return {
        "source": "freshly_computed_in_diagnose_srn_training",
        "source_path": None,
        "alpha": PRIMARY_ALPHA,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(train_sequences, val_sequences, test_sequences, device):
    results = []

    for epochs in EPOCH_VALUES:
        for lr in LR_VALUES:
            print(f"\n--- SRN condition: epochs={epochs}, lr={lr} ---")
            set_seed(SEED)
            model = ElmanKeySRN(input_size=24, hidden_size=HIDDEN_SIZE, output_size=24).to(device)

            model, log = rc._run_capturing_stdout(
                train_srn,
                model,
                train_sequences,
                val_sequences=val_sequences,
                epochs=epochs,
                lr=lr,
                seed=SEED,
                device=device,
                verbose=True,
            )
            final = rc._parse_last_epoch_line(log)

            def predict_fn(seq, _model=model):
                from srn_model import predict_sequence_probs
                return predict_sequence_probs(_model, seq, device=device)

            test_metrics = rc.evaluate_model(test_sequences, predict_fn)
            print(
                f"  final train loss={final.get('train_loss')} val loss={final.get('val_loss')} "
                f"val acc={final.get('val_acc')} | test overall_accuracy={test_metrics['overall_accuracy']:.4f}"
            )

            results.append({
                "epochs": epochs,
                "lr": lr,
                "hidden_size": HIDDEN_SIZE,
                "seed": SEED,
                "training": {
                    "final_train_loss": final.get("train_loss"),
                    "final_train_acc": final.get("train_acc"),
                    "final_val_loss": final.get("val_loss"),
                    "final_val_acc": final.get("val_acc"),
                },
                "test_metrics": test_metrics,
            })

    return results


def pick_best_condition(sweep_results):
    """Best = highest test overall_accuracy; ties broken by lower mean modulation lag,
    then by lower switch_failure_rate."""
    def sort_key(r):
        m = r["test_metrics"]
        acc = m["overall_accuracy"] if m["overall_accuracy"] is not None else -1.0
        lag = m["modulation_lag"]["mean_lag"]
        lag_key = lag if lag is not None else float("inf")
        fail = m["modulation_lag"]["switch_failure_rate"]
        fail_key = fail if fail is not None else float("inf")
        return (-acc, lag_key, fail_key)

    return sorted(sweep_results, key=sort_key)[0]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt(x, digits=4):
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _sweep_row(r):
    tm = r["test_metrics"]
    lag = tm["modulation_lag"]
    tr = r["training"]
    return (
        f"| {r['epochs']} | {r['lr']:g} "
        f"| {_fmt(tr['final_train_loss'])} | {_fmt(tr['final_val_loss'])} | {_fmt(tr['final_val_acc'])} "
        f"| {_fmt(tm['overall_accuracy'])} | {_fmt(tm['masked_accuracy_excl_ambiguous'])} "
        f"| {_fmt(tm['modulation_accuracy'])} | {_fmt(tm['no_modulation_accuracy'])} "
        f"| {_fmt(lag['mean_lag'], 2)} | {_fmt(lag['switch_failure_rate'])} |"
    )


def build_report_md(results: dict) -> str:
    ds = results["dataset"]
    baseline = results["baseline_reference"]
    sweep = results["sweep"]
    best = results["best_condition"]

    lines = []
    lines.append("# SRN Training Diagnostic Sweep (Phase 1)")
    lines.append("")
    lines.append(
        "Focused diagnostic: does the Elman SRN's weaker showing in the first synthetic comparison "
        "(`EMA_vs_SRN_synthetic_comparison_report.md`, 10 epochs) reflect undertraining, or a more "
        "structural limitation? Sweeps epochs x learning rate only; `hidden_size` and the Elman "
        "architecture are held fixed. No plots, no MIDI/chroma, no raw-chroma SRN."
    )
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append(f"- Base seed: `{ds['seed_base']}`")
    lines.append(f"- Train sequences: {ds['train_size']} (rng_seed={ds['train_seed']})")
    lines.append(f"- Val sequences: {ds['val_size']} (rng_seed={ds['val_seed']})")
    lines.append(f"- Test sequences: {ds['test_size']} (rng_seed={ds['test_seed']})")
    lines.append(f"- Sequence length range (per segment): {tuple(ds['length_range'])}")
    lines.append(
        "- Identical construction to `run_comparison.py` (same seeds/sizes), so results here are "
        "directly comparable to the EMA+MLP baseline reference below."
    )
    lines.append("")

    lines.append("## EMA+MLP baseline reference (alpha=0.20)")
    lines.append("")
    lines.append(f"- Source: {baseline['source']}" + (f" (`{baseline['source_path']}`)" if baseline.get("source_path") else ""))
    bm = baseline["metrics"]
    blag = bm["modulation_lag"]
    lines.append(
        f"- overall_accuracy={_fmt(bm['overall_accuracy'])}, "
        f"masked_accuracy_excl_ambiguous={_fmt(bm['masked_accuracy_excl_ambiguous'])}, "
        f"modulation_accuracy={_fmt(bm['modulation_accuracy'])}, "
        f"no_modulation_accuracy={_fmt(bm['no_modulation_accuracy'])}, "
        f"mean_lag={_fmt(blag['mean_lag'], 2)}, switch_failure_rate={_fmt(blag['switch_failure_rate'])}"
    )
    lines.append("")

    lines.append("## SRN epoch x learning-rate sweep (hidden_size=48, architecture unchanged)")
    lines.append("")
    lines.append(
        "| epochs | lr | final_train_loss | final_val_loss | final_val_acc | test_overall_acc | "
        "test_masked_acc (excl. ambig.) | test_modulation_acc | test_no_mod_acc | test_mean_lag | test_switch_fail_rate |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sweep:
        lines.append(_sweep_row(r))
    lines.append("")

    lines.append("## Best SRN condition")
    lines.append("")
    lines.append(
        f"- epochs={best['epochs']}, lr={best['lr']:g} "
        f"(selected by highest test overall_accuracy, tie-broken by lower mean modulation lag, "
        f"then lower switch_failure_rate)"
    )
    btm = best["test_metrics"]
    blag2 = btm["modulation_lag"]
    lines.append(
        f"- test overall_accuracy={_fmt(btm['overall_accuracy'])}, "
        f"mean_lag={_fmt(blag2['mean_lag'], 2)}, switch_failure_rate={_fmt(blag2['switch_failure_rate'])}"
    )
    lines.append("")

    lines.append("## Answers")
    lines.append("")
    lines.append(build_answers(sweep, baseline["metrics"], best))
    lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This diagnostic only sweeps epochs and learning rate at a fixed `hidden_size=48` and "
        "unchanged Elman architecture. It does not sweep hidden size, does not add plots "
        "(`plotting_comparison.py`), and does not evaluate `Twinkle.mid`/`Twinkle 12.mid` (still "
        "synthetic-only, same as `run_comparison.py`)."
    )
    lines.append("")

    return "\n".join(lines)


def build_answers(sweep, baseline_metrics, best):
    parts = []

    # Q1: does performance improve meaningfully with more epochs?
    by_epochs = {}
    for r in sweep:
        by_epochs.setdefault(r["epochs"], []).append(r["test_metrics"]["overall_accuracy"])
    epoch_means = {e: (sum(v) / len(v)) for e, v in by_epochs.items() if all(x is not None for x in v)}
    sorted_epochs = sorted(epoch_means.keys())
    if len(sorted_epochs) >= 2:
        first_e, last_e = sorted_epochs[0], sorted_epochs[-1]
        delta = epoch_means[last_e] - epoch_means[first_e]
        trend = "improves" if delta > 0.02 else ("stays roughly flat" if abs(delta) <= 0.02 else "gets worse")
        parts.append(
            f"**(1) Epoch sensitivity:** across the swept learning rates, mean test overall_accuracy "
            f"{trend} going from {first_e} to {last_e} epochs "
            f"({_fmt(epoch_means[first_e])} -> {_fmt(epoch_means[last_e])}, delta={_fmt(delta)})."
        )

    # Q2: does any condition approach/beat baseline?
    b_acc = baseline_metrics["overall_accuracy"]
    best_acc = best["test_metrics"]["overall_accuracy"]
    if b_acc is not None and best_acc is not None:
        if best_acc > b_acc:
            rel = f"beats it by {_fmt(best_acc - b_acc)}"
        elif best_acc >= b_acc - 0.03:
            rel = f"approaches it (within {_fmt(b_acc - best_acc)})"
        else:
            rel = f"falls short by {_fmt(b_acc - best_acc)}"
        parts.append(
            f"**(2) Vs. baseline:** the best swept SRN condition (epochs={best['epochs']}, lr={best['lr']:g}) "
            f"reaches test overall_accuracy={_fmt(best_acc)} against the EMA+MLP baseline's {_fmt(b_acc)} -- {rel}."
        )

    # Q3: main weakness -- accuracy, lag, or switch failure?
    b_lag = baseline_metrics["modulation_lag"]["mean_lag"]
    b_fail = baseline_metrics["modulation_lag"]["switch_failure_rate"]
    best_lag = best["test_metrics"]["modulation_lag"]["mean_lag"]
    best_fail = best["test_metrics"]["modulation_lag"]["switch_failure_rate"]

    gaps = {}
    if b_acc is not None and best_acc is not None:
        gaps["accuracy"] = b_acc - best_acc  # positive = SRN worse
    if b_lag is not None and best_lag is not None:
        # normalize lag gap roughly by baseline lag scale to make it comparable to accuracy/rate gaps
        gaps["modulation_lag"] = (best_lag - b_lag) / max(b_lag, 1.0)
    if b_fail is not None and best_fail is not None:
        gaps["switch_failure_rate"] = best_fail - b_fail

    if gaps:
        worst_dim = max(gaps, key=lambda k: gaps[k])
        parts.append(
            f"**(3) Main weakness:** comparing the best SRN condition to the EMA+MLP baseline, the largest "
            f"relative gap is in **{worst_dim}** "
            f"(accuracy gap={_fmt(gaps.get('accuracy'))}, "
            f"normalized lag gap={_fmt(gaps.get('modulation_lag'))}, "
            f"switch_failure_rate gap={_fmt(gaps.get('switch_failure_rate'))})."
        )

    # Q4: recommendation
    if b_acc is not None and best_acc is not None and best_acc >= b_acc - 0.03:
        rec = (
            "the best SRN condition is close enough to the baseline that the next reasonable step is "
            "plotting the best SRN vs. EMA (probability tracking + Circle-of-Fifths trajectories) to "
            "inspect qualitative behavior, rather than further retuning training."
        )
    else:
        rec = (
            "the gap to the baseline is still large enough that revising training/data design "
            "(e.g. more epochs beyond this sweep's range, a learning-rate schedule, or reconsidering "
            "sequence length/composition) is likely more productive than moving to plots yet."
        )
    parts.append(f"**(4) Recommended next step:** {rec}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _scan_for_nan(obj, path="root"):
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

    nan_paths = _scan_for_nan(results)
    checks.append(("no NaNs anywhere in metrics", len(nan_paths) == 0))

    expected_conditions = {(e, lr) for e in EPOCH_VALUES for lr in LR_VALUES}
    found_conditions = {(r["epochs"], r["lr"]) for r in results["sweep"]}
    checks.append(("all sweep conditions represented", expected_conditions == found_conditions))

    checks.append(("results contains baseline_reference", "baseline_reference" in results))
    checks.append(("results contains best_condition", "best_condition" in results))

    with open(md_path) as f:
        md_text = f.read()
    checks.append(("markdown report present and non-empty", len(md_text) > 0))
    checks.append(("markdown mentions SRN", "SRN" in md_text))
    checks.append(("markdown mentions EMA", "EMA" in md_text))

    print()
    print("diagnose_srn_training.py verification")
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

    print("Building synthetic train/val/test datasets (same construction as run_comparison.py)...")
    train_sequences, val_sequences, test_sequences, dataset_info = rc.build_datasets()
    print(
        f"  train={dataset_info['train_size']} val={dataset_info['val_size']} "
        f"test={dataset_info['test_size']} length_range={dataset_info['length_range']}"
    )

    device = get_device()

    print("\nLoading/computing EMA+MLP baseline reference (alpha=0.20)...")
    baseline_reference = load_or_compute_baseline_reference(test_sequences, device)
    print(f"  baseline overall_accuracy={_fmt(baseline_reference['metrics']['overall_accuracy'])} (source: {baseline_reference['source']})")

    print(f"\nRunning SRN sweep: epochs={EPOCH_VALUES} x lr={LR_VALUES} (hidden_size={HIDDEN_SIZE} fixed)...")
    sweep_results = run_sweep(train_sequences, val_sequences, test_sequences, device)

    best_condition = pick_best_condition(sweep_results)

    results = {
        "phase": "phase_1_synthetic_only_srn_training_diagnostic",
        "seed": SEED,
        "dataset": dataset_info,
        "baseline_reference": baseline_reference,
        "sweep_settings": {
            "epoch_values": EPOCH_VALUES,
            "lr_values": LR_VALUES,
            "hidden_size": HIDDEN_SIZE,
        },
        "sweep": sweep_results,
        "best_condition": best_condition,
        "notes": (
            "Synthetic-only Phase 1 diagnostic. Sweeps SRN epochs x learning rate at fixed "
            "hidden_size=48 and unchanged Elman architecture; does not sweep hidden size. "
            "Does not evaluate Twinkle.mid or Twinkle 12.mid."
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
