"""Classifier training — same feature-caching principle as the detect
tier's train.py: DINOv2's CLS token for a given image is identical every
epoch (frozen), so it's extracted once per image up front, then every
epoch only re-runs the actual trainable part (one linear layer) against
the cached tokens. Even more valuable here than for detect: the classifier
forward pass on cached features is essentially free, so an entire training
run is dominated by the one-time caching pass.
"""
from __future__ import annotations

import copy
import io
import logging
import random

import torch

from app.classify_dataset import load_classify_split
from app.classify_model import build_classifier
from app.device import DEVICE
from app.early_stop import EpochCallback, EpochMonitor
from app.seeding import seed_everything
from app.storage import put_bytes

logger = logging.getLogger(__name__)

BATCH_SIZE = 8


def run_classify_training(
    *,
    org_id: str,
    backbone_key: str,
    s3_prefix: str,
    epochs: int,
    lr: float,
    artifact_prefix: str | None = None,
    seed: int | None = None,
    weight_decay: float = 0.01,
    dry_run: bool = False,  # tuning trials: score the config, don't write an artifact
    patience: int | None = None,  # stop when the best epoch is this many epochs ago (None: train all epochs)
    min_epochs: int = 0,
    on_epoch: EpochCallback | None = None,
) -> dict:
    seed_everything(seed)
    monitor = EpochMonitor(patience, min_epochs, on_epoch)
    train_examples = load_classify_split(org_id, s3_prefix, "train")
    eval_examples = load_classify_split(org_id, s3_prefix, "eval")
    if not train_examples:
        raise ValueError("No usable training examples (manifest empty).")
    if not any(e.label > 0.5 for e in train_examples) or not any(e.label < 0.5 for e in train_examples):
        raise ValueError("Training split needs both positive and negative examples.")

    model = build_classifier(backbone_key)
    model.train()
    optimizer = torch.optim.AdamW(model.linear.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    logger.info("dino_classify_train caching frozen features for %d examples", len(train_examples))
    cached: list[dict] = []
    with torch.no_grad():
        for ex in train_examples:
            # extract_cls moves its input to DEVICE internally; cache the
            # result on CPU (see train.py's _cache_examples for the same
            # bounded-GPU-memory reasoning) and move per-batch below.
            cls_token = model.extract_cls(ex.image.unsqueeze(0))[0].detach().cpu()
            cached.append({"cls": cls_token, "label": torch.tensor(ex.label, dtype=torch.float32)})

    # Eval split = validation set: loss tracked per epoch, best epoch saved.
    cached_eval: list[dict] = []
    with torch.no_grad():
        for ex in eval_examples:
            cls_token = model.extract_cls(ex.image.unsqueeze(0))[0].detach().cpu()
            cached_eval.append({"cls": cls_token, "label": torch.tensor(ex.label, dtype=torch.float32)})

    epoch_losses: list[float] = []
    val_losses: list[float] = []
    best_val, best_epoch, best_state = float("inf"), None, None
    for epoch in range(epochs):
        random.shuffle(cached)
        running_loss = 0.0
        num_batches = 0
        for i in range(0, len(cached), BATCH_SIZE):
            batch = cached[i : i + BATCH_SIZE]
            cls_batch = torch.stack([b["cls"] for b in batch]).to(DEVICE)
            labels_batch = torch.stack([b["label"] for b in batch]).to(DEVICE)

            optimizer.zero_grad()
            logits = model.linear(cls_batch).squeeze(-1)
            loss = loss_fn(logits, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        epoch_losses.append(avg_loss)
        if cached_eval:
            with torch.no_grad():
                logits = model.linear(torch.stack([b["cls"] for b in cached_eval]).to(DEVICE)).squeeze(-1)
                val_loss = float(loss_fn(logits, torch.stack([b["label"] for b in cached_eval]).to(DEVICE)).item())
            val_losses.append(val_loss)
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch + 1
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.linear.state_dict().items()})
            if monitor.update(epoch + 1, val_loss, best_epoch):
                logger.info("dino_classify_train early stop at epoch %d (best %d)", epoch + 1, best_epoch)
                break
        logger.info("dino_classify_train epoch=%d/%d avg_loss=%.4f", epoch + 1, epochs, avg_loss)

    if best_state is not None:
        model.linear.load_state_dict(best_state)

    # Artifact is just the linear layer — a handful of KB, not DINOv2's
    # frozen params. Even smaller than the detect tier's head.
    artifact = {
        "kind": "classify",
        "backbone_key": backbone_key,
        "linear_state_dict": model.linear.state_dict(),
    }
    artifact_key = None
    if not dry_run:
        buf = io.BytesIO()
        torch.save(artifact, buf)
        artifact_key = f"{artifact_prefix or s3_prefix}/head_artifact.pt"  # the model gets its own folder when one is given
        put_bytes(org_id, artifact_key, buf.getvalue(), "application/octet-stream")

    return {
        "artifact_key": artifact_key,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "epoch_losses": epoch_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch if best_epoch is not None else len(epoch_losses),
        "best_val_loss": best_val if best_epoch is not None else None,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "epochs_run": len(epoch_losses),
        "stopped_early": monitor.stopped_early,
    }
