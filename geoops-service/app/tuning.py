"""Automatic hyperparameter selection (Optuna) and multi-seed reporting.

Both are plain wrappers around the existing trainers, which stay the single
source of truth for how a model is trained:

* ``tune``     — a small in-memory Optuna study (TPE). Every trial runs the real
                 trainer with ``dry_run=True`` (no artifact written) and is scored
                 by the trainer's own best validation loss. The study is not
                 persisted here; the caller stores the returned result (the API
                 keeps it in Postgres so the same dataset + model is never tuned
                 twice). Nothing is scored on the test split.
* ``multirun`` — trains one model per seed with the chosen hyperparameters,
                 evaluates each on the held-out test split, and reports
                 mean / sample std (n-1) / min / max per metric. The model that is
                 kept is the seed with the best *validation* loss, so the test
                 split is never used to pick a winner.

``train_fn(overrides) -> train result`` and ``eval_fn(artifact_key) -> metrics``
are supplied by main.py (they bind the request), which keeps this module free
of torch / HTTP concerns and trivially testable with fakes.
"""
from __future__ import annotations

import logging
import math
import statistics
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

TrainFn = Callable[..., dict[str, Any]]  # (overrides, on_epoch=None) -> train result
EvalFn = Callable[[str], dict[str, Any]]

# Metrics aggregated across seeds (a metric the evaluator did not return is skipped).
METRICS: dict[str, tuple[str, ...]] = {
    "classify": ("accuracy", "precision", "recall", "f1"),
    "fcos": ("ap50", "precision", "recall", "mean_iou_tp"),
    "yolo": ("ap50", "map", "precision", "recall"),
    "segment": ("ap50", "precision", "recall", "mean_iou_tp", "pixel_iou", "dice"),
}


# A DINOv2-head search trial trains in steps of 30 epochs (30 | 60 | 90 | 120): at least MIN_EPOCHS, at most EPOCH_CAP, and it
# stops as soon as validation loss has not improved for EARLY_STOP_PATIENCE epochs. The epoch the search reports is where
# validation loss was lowest, so a high cap lets it find that epoch without paying for the epochs after it.
MIN_EPOCHS = 30
EPOCH_CAP = 120
EARLY_STOP_PATIENCE = 15
# Trials clearly behind the median of the earlier ones (compared at the same epoch, after the first MIN_EPOCHS) are abandoned.
PRUNER_STARTUP_TRIALS = 3


def _suggest(trial, kind: str, base: dict[str, Any]) -> dict[str, Any]:
    """The search space. Ranges always include the request's own values so the
    first (baseline) trial is exactly what an untuned run would have used."""
    if kind == "yolo":
        out = {}
        if int(base["stage1_epochs"]) > 0:
            lr1 = float(base["stage1_lr"])
            out["stage1_lr"] = trial.suggest_float("stage1_lr", min(5e-4, lr1), max(1e-2, lr1), log=True)
        if int(base["stage2_epochs"]) > 0:
            lr2 = float(base["stage2_lr"])
            out["stage2_lr"] = trial.suggest_float("stage2_lr", min(2e-5, lr2), max(5e-4, lr2), log=True)
        return out
    # Epochs are not sampled: every trial trains up to the same EPOCH_CAP and reports the epoch where its validation loss was
    # lowest. The best trial's epoch is the epochs value found by the search (see tune()). Sampling epochs directly would
    # always pick the longest run, because each run keeps its best-validation epoch anyway.
    lr, wd = float(base["lr"]), float(base["weight_decay"])
    return {
        "lr": trial.suggest_float("lr", min(1e-5, lr), max(3e-3, lr), log=True),
        "weight_decay": trial.suggest_float("weight_decay", min(1e-4, wd), max(1e-1, wd), log=True),
    }


def tune(kind: str, base: dict[str, Any], train_fn: TrainFn, n_trials: int, seed: int = 0) -> dict[str, Any]:
    """Run ``n_trials`` sequential trials and return the best params plus the full trial history.

    ``base`` holds the request's current values for every tunable key; they are
    enqueued as trial 0, so the result is never worse (on validation) than not tuning.
    """
    import optuna  # noqa: PLC0415 — keep the import off /health's path

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=PRUNER_STARTUP_TRIALS, n_warmup_steps=MIN_EPOCHS),
    )
    if kind == "yolo":  # both stages are trained in every trial; only a stage that runs has a rate to tune
        keys = tuple(k for k, n in (("stage1_lr", "stage1_epochs"), ("stage2_lr", "stage2_epochs")) if int(base[n]) > 0)
    else:
        keys = ("lr", "weight_decay")
    study.enqueue_trial({k: base[k] for k in keys})

    def objective(trial) -> float:
        overrides = _suggest(trial, kind, base)
        # Same seed and same epoch budget in every trial, so trials differ only by learning rate and weight decay.
        budget = {} if kind == "yolo" else {"epochs": EPOCH_CAP, "patience": EARLY_STOP_PATIENCE, "min_epochs": MIN_EPOCHS}

        def report(epoch: int, val_loss: float) -> None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                trial.set_user_attr("pruned_at", epoch)
                raise optuna.TrialPruned()

        result = train_fn({**overrides, **budget, "seed": seed, "dry_run": True}, on_epoch=None if kind == "yolo" else report)
        value = result.get("best_val_loss")
        if value is None or not math.isfinite(value):
            raise ValueError("Tuning needs a validation (eval) split with usable examples.")
        if kind != "yolo":
            trial.set_user_attr("best_epoch", int(result.get("best_epoch") or EPOCH_CAP))
            trial.set_user_attr("epochs_run", int(result.get("epochs_run") or result.get("best_epoch") or EPOCH_CAP))
        logger.info("tune kind=%s trial=%d value=%.5f best_epoch=%s params=%s", kind, trial.number, value, result.get("best_epoch"), overrides)
        return float(value)

    t0 = time.monotonic()
    study.optimize(objective, n_trials=n_trials)
    best_params = dict(study.best_params)
    if kind != "yolo":
        # The epochs value found by the search: where the best trial's validation loss bottomed out. Every final run uses it.
        best_params["epochs"] = int(study.best_trial.user_attrs.get("best_epoch", EPOCH_CAP))
    return {
        "metric": "best_val_loss",
        "direction": "minimize",
        "n_trials": len(study.trials),
        "seed": seed,
        # What every trial shared, so the report can say so.
        "fixed": {k: base[k] for k in ("stage1_epochs", "stage2_epochs")} if kind == "yolo" else {"max_epochs": EPOCH_CAP, "min_epochs": MIN_EPOCHS, "patience": EARLY_STOP_PATIENCE},
        "epoch_search": None if kind == "yolo" else "best_epoch",
        "best_params": best_params,
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "trials": [
            {
                "number": t.number, "params": t.params, "state": t.state.name,
                # A pruned trial's "value" in Optuna is just its last epoch's loss, which is not comparable: report none.
                "value": t.value if t.state.name == "COMPLETE" else None,
                "best_epoch": t.user_attrs.get("best_epoch"), "epochs_run": t.user_attrs.get("epochs_run"), "pruned_at": t.user_attrs.get("pruned_at"),
            }
            for t in study.trials
        ],
        "seconds": round(time.monotonic() - t0, 2),
    }


def summarize(values: list[float | None]) -> dict[str, Any]:
    """mean / sample std (n-1) / min / max over the non-missing values. A single run has std 0."""
    xs = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not xs:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 4),
        "std": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
        "min": round(min(xs), 4),
        "max": round(max(xs), 4),
    }


def multirun(
    kind: str, seeds: list[int], train_fn: TrainFn, eval_fn: EvalFn, artifact_prefix: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train + test-evaluate once per seed. Returns ``(report, winning train result)``.

    Each seed's model lands under ``{artifact_prefix}/seeds/seed-{n}/``; the one
    with the best validation loss is the model that gets registered.
    """
    names = METRICS[kind]
    runs: list[dict[str, Any]] = []
    results: dict[int, dict[str, Any]] = {}
    t0 = time.monotonic()
    for seed in seeds:
        result = train_fn({"seed": seed, "artifact_prefix": f"{artifact_prefix}/seeds/seed-{seed}"})
        metrics = eval_fn(result["artifact_key"])
        results[seed] = result
        runs.append({
            "seed": seed,
            "artifact_key": result["artifact_key"],
            "best_epoch": result.get("best_epoch"),
            "best_val_loss": result.get("best_val_loss"),
            "epoch_losses": result.get("epoch_losses"),
            "val_losses": result.get("val_losses"),
            "metrics": {m: metrics.get(m) for m in names},
        })
        logger.info("multirun kind=%s seed=%d val=%s metrics=%s", kind, seed, result.get("best_val_loss"), runs[-1]["metrics"])

    def val_key(r: dict[str, Any]) -> float:
        v = r["best_val_loss"]
        return v if v is not None and math.isfinite(v) else math.inf

    winner = min(runs, key=val_key)  # ties / no validation split -> first seed
    report = {
        "seeds": list(seeds),
        "n_runs": len(runs),
        "std": "sample (n-1)",
        "selected_seed": winner["seed"],
        "selected_by": "best_val_loss",
        "summary": {m: summarize([r["metrics"][m] for r in runs]) for m in names},
        "best_val_loss": summarize([r["best_val_loss"] for r in runs]),
        "runs": runs,
        "seconds": round(time.monotonic() - t0, 2),
    }
    return report, results[winner["seed"]]


def run_auto(
    kind: str, base: dict[str, Any], train_fn: TrainFn, eval_fn: EvalFn, artifact_prefix: str,
    tune_trials: int, seeds: list[int],
) -> dict[str, Any]:
    """Optionally tune, then train: once per seed (multirun) or once. Returns the
    winning train result plus ``tuning`` / ``multirun`` / ``applied_params``."""
    tuning = tune(kind, base, train_fn, tune_trials, seed=seeds[0] if seeds else 0) if tune_trials > 0 else None
    params = dict(tuning["best_params"]) if tuning else {}
    final = lambda extra: train_fn({**params, **extra})  # noqa: E731 — tuned params apply to every final run

    report = None
    if seeds:
        report, result = multirun(kind, seeds, final, eval_fn, artifact_prefix)
    else:
        result = final({"seed": 0, "artifact_prefix": artifact_prefix})
    return {**result, "tuning": tuning, "multirun": report, "applied_params": params}
