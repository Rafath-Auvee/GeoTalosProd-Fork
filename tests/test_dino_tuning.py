"""Auto-tune / multi-run: request validation and the tuned-config read model."""
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from geoops.dino_api import _tuned_read
from geoops.dino_schemas import DinoFinetuneRequest


def _req(**kw):
    return DinoFinetuneRequest(dataset_id=uuid4(), **kw)


def test_defaults_are_a_plain_single_run():
    r = _req()
    assert (r.auto_tune, r.force_retune, r.seeds, r.n_trials) == (False, False, [], 6)


@pytest.mark.parametrize("seeds", [[0], [0, 1], [0, 1, 2, 3, 4, 5], [0, 0, 0]])
def test_seeds_must_be_three_to_five_distinct(seeds):
    with pytest.raises(ValidationError):
        _req(seeds=seeds)


def test_valid_seeds_are_kept_in_order():
    assert _req(seeds=[3, 1, 2]).seeds == [3, 1, 2]
    assert len(_req(seeds=[0, 1, 2, 3, 4]).seeds) == 5


def test_negative_seed_rejected():
    with pytest.raises(ValidationError):
        _req(seeds=[-1, 1, 2])


def test_n_trials_bounds():
    with pytest.raises(ValidationError):
        _req(n_trials=0)
    with pytest.raises(ValidationError):
        _req(n_trials=21)


def _row(build_no):
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=uuid4(), dataset_id=uuid4(), dataset_build_no=build_no, model_key="dinov2-vits14",
        best_params={"lr": 1e-3}, tuning={"n_trials": 5}, multirun=None, job_id=None, created_at=now, updated_at=now,
    )


def test_tuned_config_is_stale_after_a_dataset_rebuild():
    assert _tuned_read(_row(1), SimpleNamespace(build_no=1)).is_stale is False
    assert _tuned_read(_row(1), SimpleNamespace(build_no=2)).is_stale is True
