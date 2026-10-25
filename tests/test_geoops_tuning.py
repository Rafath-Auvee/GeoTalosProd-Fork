"""geoops-service/app/tuning.py: multi-seed aggregation and the tune -> train flow, with fake trainers.

The module is loaded by path: geoops-service has its own top-level ``app`` package, which clashes with this one.
"""
import importlib.util
import math
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "geoops-service" / "app" / "tuning.py"
_spec = importlib.util.spec_from_file_location("geoops_tuning", _PATH)
tuning = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tuning)

_stop_spec = importlib.util.spec_from_file_location("geoops_early_stop", _PATH.with_name("early_stop.py"))
early_stop = importlib.util.module_from_spec(_stop_spec)
_stop_spec.loader.exec_module(early_stop)

BASE = dict(lr=1e-4, weight_decay=0.01, epochs=30, stage1_lr=0.002, stage2_lr=1e-4, stage1_epochs=30, stage2_epochs=30)


def _train(overrides, on_epoch=None):
    seed = overrides.get("seed", 0)
    lr = overrides.get("lr", BASE["lr"])
    return {
        "best_val_loss": 0.1 * (math.log10(lr) + 3) ** 2 + 0.01 * (seed == 0) + 0.02 * (seed == 2),
        "best_epoch": 4,
        "artifact_key": None if overrides.get("dry_run") else f"{overrides['artifact_prefix']}/head_artifact.pt",
    }


def _eval(_key):
    return {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75}


def test_summarize_uses_sample_std_and_skips_missing():
    s = tuning.summarize([0.8, None, 0.9, 1.0])
    assert (s["n"], s["mean"], s["std"], s["min"], s["max"]) == (3, 0.9, 0.1, 0.8, 1.0)
    assert tuning.summarize([0.5])["std"] == 0.0
    assert tuning.summarize([None])["mean"] is None


def test_multirun_reports_mean_std_and_keeps_best_validation_seed():
    report, winner = tuning.multirun("classify", [0, 1, 2], _train, _eval, "dino-models/j")
    assert report["n_runs"] == 3 and report["selected_seed"] == 1  # seed 1 has the lowest val loss
    assert report["summary"]["accuracy"] == {"n": 3, "mean": 0.9, "std": 0.0, "min": 0.9, "max": 0.9}
    assert winner["artifact_key"] == "dino-models/j/seeds/seed-1/head_artifact.pt"
    assert [r["seed"] for r in report["runs"]] == [0, 1, 2]


def test_run_auto_without_tuning_or_seeds_is_one_plain_run():
    out = tuning.run_auto("classify", BASE, _train, _eval, "p", 0, [])
    assert out["tuning"] is None and out["multirun"] is None and out["applied_params"] == {}
    assert out["artifact_key"] == "p/head_artifact.pt"


def test_tune_starts_from_the_request_and_applies_best_params_to_every_seed():
    pytest.importorskip("optuna")
    seen = []

    def train(o, on_epoch=None):
        seen.append(o)
        return _train(o)

    out = tuning.run_auto("classify", BASE, train, _eval, "p", 4, [0, 1, 2])
    trials = out["tuning"]["trials"]
    assert len(trials) == 4 and trials[0]["params"]["lr"] == BASE["lr"]  # trial 0 = the request's own values
    assert out["tuning"]["best_value"] == min(t["value"] for t in trials)
    assert all(o["dry_run"] and o["seed"] == 0 for o in seen[:4])  # trials write no artifact, one seed
    finals = [o for o in seen[4:]]
    assert len(finals) == 3 and all(o["lr"] == out["applied_params"]["lr"] and not o.get("dry_run") for o in finals)


def test_tune_requires_a_validation_split():
    pytest.importorskip("optuna")
    with pytest.raises(ValueError, match="validation"):
        tuning.tune("classify", BASE, lambda o, on_epoch=None: {"best_val_loss": None}, 2)


def test_yolo_tunes_only_the_stages_that_run():
    pytest.importorskip("optuna")
    two = tuning.tune("yolo", BASE, lambda o, on_epoch=None: {"best_val_loss": o["stage1_lr"] + o["stage2_lr"]}, 3)
    assert set(two["best_params"]) == {"stage1_lr", "stage2_lr"}
    one = tuning.tune("yolo", {**BASE, "stage2_epochs": 0}, lambda o, on_epoch=None: {"best_val_loss": o["stage1_lr"]}, 3)
    assert set(one["best_params"]) == {"stage1_lr"}


def test_search_finds_the_epoch_from_the_best_trials_best_epoch():
    pytest.importorskip("optuna")
    seen = []

    def train(o, on_epoch=None):
        seen.append(o)
        lr = o.get("lr", BASE["lr"])
        # lower lr -> converges later: best epoch grows as lr shrinks; val loss is best near lr = 1e-3
        return {"best_val_loss": abs(math.log10(lr) + 3), "best_epoch": int(10 + 40 * (1e-2 / lr) ** 0.3) if lr else 10}

    out = tuning.tune("classify", BASE, train, 5)
    assert all(o["epochs"] == tuning.EPOCH_CAP and o["dry_run"] for o in seen), "every trial trains to the same cap"
    best = min(out["trials"], key=lambda t: t["value"])
    assert out["best_params"]["epochs"] == best["best_epoch"], "the epochs found = the best trial's best epoch"
    assert set(out["best_params"]) == {"lr", "weight_decay", "epochs"}
    assert out["epoch_search"] == "best_epoch"
    assert out["fixed"] == {"max_epochs": tuning.EPOCH_CAP, "min_epochs": tuning.MIN_EPOCHS, "patience": tuning.EARLY_STOP_PATIENCE}
    assert all(t["best_epoch"] is not None for t in out["trials"])


def test_final_runs_use_the_found_epochs_and_yolo_keeps_its_stages():
    pytest.importorskip("optuna")
    seen = []

    def train(o, on_epoch=None):
        seen.append(o)
        return {"best_val_loss": 0.1, "best_epoch": 17, "artifact_key": f"{o.get('artifact_prefix')}/a.pt"}

    out = tuning.run_auto("classify", BASE, train, _eval, "p", 2, [0, 1, 2])
    finals = [o for o in seen if not o.get("dry_run")]
    assert out["applied_params"]["epochs"] == 17
    assert len(finals) == 3 and all(o["epochs"] == 17 for o in finals)
    yolo = tuning.tune("yolo", BASE, lambda o, on_epoch=None: {"best_val_loss": o["stage1_lr"]}, 2)
    assert "epochs" not in yolo["best_params"] and yolo["epoch_search"] is None
    assert yolo["fixed"] == {"stage1_epochs": 30, "stage2_epochs": 30}


def test_early_stop_waits_for_min_epochs_then_stops_patience_after_the_best_epoch():
    m = early_stop.EpochMonitor(patience=15, min_epochs=30)
    # best epoch 20; worse ever since. It must not stop before epoch 30, and stops once 15 epochs have passed since the best (35).
    stops = [e for e in range(1, 61) if (m.update(e, 1.0, 20))]
    assert stops[0] == 35 and m.stopped_early
    quiet = early_stop.EpochMonitor(patience=None)
    assert not any(quiet.update(e, 1.0, 1) for e in range(1, 200)), "no patience means train every epoch"


def test_epoch_hook_can_abandon_a_run():
    def boom(epoch, val):
        raise RuntimeError("pruned")
    m = early_stop.EpochMonitor(on_epoch=boom)
    with pytest.raises(RuntimeError):
        m.update(1, 0.5, 1)


def test_search_trains_in_capped_steps_with_early_stop_and_prunes_trials_that_fall_behind():
    pytest.importorskip("optuna")
    seen = []

    def train(o, on_epoch=None):
        seen.append(o)
        lr = o.get("lr", BASE["lr"])
        loss = abs(math.log10(lr) + 3)  # the further lr is from 1e-3, the worse the whole curve
        for epoch in range(1, 41):
            if on_epoch:
                on_epoch(epoch, loss + 1.0 / epoch)
        return {"best_val_loss": loss, "best_epoch": 40, "epochs_run": 40}

    out = tuning.tune("classify", BASE, train, 8)
    assert all(o["epochs"] == tuning.EPOCH_CAP and o["patience"] == tuning.EARLY_STOP_PATIENCE and o["min_epochs"] == tuning.MIN_EPOCHS for o in seen)
    assert tuning.EPOCH_CAP == 120 and tuning.MIN_EPOCHS == 30
    states = {t["state"] for t in out["trials"]}
    assert "COMPLETE" in states and "PRUNED" in states, states
    pruned = [t for t in out["trials"] if t["state"] == "PRUNED"]
    assert all(t["value"] is None and t["pruned_at"] >= tuning.MIN_EPOCHS for t in pruned), "no pruning during the warm-up epochs"
    assert out["trials"][0]["state"] == "COMPLETE", "the starting values are never pruned"
    assert out["best_params"]["epochs"] == 40
