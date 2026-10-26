"""Per-epoch hooks shared by the DINOv2-head trainers: early stopping on validation loss, and a callback the tuner uses to prune
a trial that is clearly behind. Both only act when the run has a validation split."""
from __future__ import annotations

from collections.abc import Callable

EpochCallback = Callable[[int, float], None]


class EpochMonitor:
    """Call ``update`` once per epoch, after the validation loss and the best epoch so far are known.

    * ``on_epoch(epoch, val_loss)``: the tuner's hook; it may raise (Optuna's TrialPruned) to abandon the run.
    * early stop: once ``min_epochs`` have run, stop when the best epoch is ``patience`` or more epochs ago. The trainer then
      keeps the best epoch's weights, so stopping early loses nothing.
    """

    def __init__(self, patience: int | None = None, min_epochs: int = 0, on_epoch: EpochCallback | None = None) -> None:
        self.patience = patience
        self.min_epochs = min_epochs
        self.on_epoch = on_epoch
        self.stopped_early = False

    def update(self, epoch: int, val_loss: float, best_epoch: int | None) -> bool:
        """``epoch`` is 1-based. Returns True when training should stop now."""
        if self.on_epoch is not None:
            self.on_epoch(epoch, val_loss)
        if self.patience and best_epoch is not None and epoch >= self.min_epochs and epoch - best_epoch >= self.patience:
            self.stopped_early = True
            return True
        return False
