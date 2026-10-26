"""One place to seed every RNG a training run touches, so a run is reproducible
from its ``seed`` (head init and shuffle order for the DINOv2 trainers; the
Ultralytics seed is passed separately by yolo_train)."""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int | None) -> None:
    """No-op when ``seed`` is None, so unseeded runs behave exactly as before."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
