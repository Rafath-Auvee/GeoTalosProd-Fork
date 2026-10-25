"""Train, load and evaluate the segmentation head (see ``segment_model``). Mirrors ``train.py``: DINOv2's output is
the same every epoch, so every example's frozen tokens are computed once and only the pyramid and the small
decoder are trained against them. The eval split is the validation set and the best epoch's weights are kept.
"""
from __future__ import annotations

import io
import logging
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from app.dataset import _load_image_tensor
from app.detect_model import INPUT_SIZE
from app.device import DEVICE
from app.segment_model import (
    OUT_SIZE, SegmentModel, build_segment, label_map, masks_to_instances, predict_probs, targets_from_labels,
)
from app.early_stop import EpochCallback, EpochMonitor
from app.seeding import seed_everything
from app.storage import get_json, put_bytes
from app.train import BATCH_SIZE, _snapshot

logger = logging.getLogger(__name__)


@dataclass
class SegExample:
    image: torch.Tensor          # (3, 518, 518), 0-1
    polygons: list[np.ndarray]   # one (N, 2) array per object, 0..1 coordinates


def load_segment_split(org_id: str, s3_prefix: str, split: str) -> list[SegExample]:
    manifest = get_json(org_id, f"{s3_prefix}/manifest.json")
    examples: list[SegExample] = []
    for item in manifest["items"]:
        if item["split"] != split:
            continue
        polys = []
        for t in get_json(org_id, item["targets"]).get("targets", []):
            flat = t.get("polygon")
            if flat and len(flat) >= 6:
                polys.append(np.asarray(flat, dtype=np.float32).reshape(-1, 2))
        # A tile with no objects stays in as a background example.
        examples.append(SegExample(image=_load_image_tensor(org_id, item["image"]), polygons=polys))
    return examples


def _cache(model: SegmentModel, examples: list[SegExample]) -> list[dict]:
    cached = []
    with torch.no_grad():
        for ex in examples:
            x = model.normalise(ex.image[None].to(DEVICE))
            tokens = model.backbone.extract_tokens(x)[0].cpu()
            small = model.small(x)[0].cpu()
            target = torch.from_numpy(targets_from_labels(label_map(ex.polygons, OUT_SIZE))).float()
            cached.append({"tokens": tokens, "small": small, "target": target})
    return cached


def _loss(model: SegmentModel, batch: list[dict]) -> torch.Tensor:
    tokens = torch.stack([b["tokens"] for b in batch]).to(DEVICE)
    small = torch.stack([b["small"] for b in batch]).to(DEVICE)
    target = torch.stack([b["target"] for b in batch]).to(DEVICE)
    feats = list(model.backbone.pyramid(tokens).values())
    logits = model.head(feats, small)
    fg_logit, edge_logit = logits[:, 0], logits[:, 1]
    fg_t, edge_t = target[:, 0], target[:, 1]
    p = torch.sigmoid(fg_logit)
    dice = 1 - (2 * (p * fg_t).sum() + 1.0) / (p.sum() + fg_t.sum() + 1.0)   # over the batch, so empty tiles don't blow up
    bce_fg = F.binary_cross_entropy_with_logits(fg_logit, fg_t)
    bce_edge = F.binary_cross_entropy_with_logits(edge_logit, edge_t, pos_weight=torch.tensor(4.0, device=DEVICE))
    return bce_fg + dice + bce_edge


def _mean_loss(model: SegmentModel, cached: list[dict]) -> float:
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(cached), BATCH_SIZE):
            total += float(_loss(model, cached[i : i + BATCH_SIZE]).item())
            n += 1
    return total / max(1, n)


def run_segment_training(
    *, org_id: str, backbone_key: str, s3_prefix: str, epochs: int, lr: float, artifact_prefix: str | None = None,
    seed: int | None = None, weight_decay: float = 0.01, dry_run: bool = False,  # dry_run: tuning trial, no artifact
    patience: int | None = None, min_epochs: int = 0, on_epoch: EpochCallback | None = None,  # early stop / tuner hook
) -> dict:
    seed_everything(seed)
    monitor = EpochMonitor(patience, min_epochs, on_epoch)
    train_examples = load_segment_split(org_id, s3_prefix, "train")
    eval_examples = load_segment_split(org_id, s3_prefix, "eval")
    if not any(ex.polygons for ex in train_examples):
        raise ValueError("No mask in the training tiles. Choose annotation sets that have masks.")

    model = build_segment(backbone_key)
    model.train()
    optimizer = torch.optim.AdamW(list(model.backbone.trainable_parameters()) + list(model.head.parameters()), lr=lr, weight_decay=weight_decay)
    cached = _cache(model, train_examples)
    cached_eval = _cache(model, eval_examples) if eval_examples else []

    epoch_losses: list[float] = []
    val_losses: list[float] = []
    best_val, best_epoch, best_state = float("inf"), None, None
    for epoch in range(epochs):
        random.shuffle(cached)
        running, batches = 0.0, 0
        for i in range(0, len(cached), BATCH_SIZE):
            optimizer.zero_grad()
            loss = _loss(model, cached[i : i + BATCH_SIZE])
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            batches += 1
        epoch_losses.append(running / max(1, batches))
        if cached_eval:
            val = _mean_loss(model, cached_eval)
            val_losses.append(val)
            if val < best_val:
                best_val, best_epoch, best_state = val, epoch + 1, _snapshot(model)
            if monitor.update(epoch + 1, val, best_epoch):
                logger.info("dino_segment_train early stop at epoch %d (best %d)", epoch + 1, best_epoch)
                break
        logger.info("dino_segment_train epoch=%d/%d loss=%.4f", epoch + 1, epochs, epoch_losses[-1])

    if best_state is not None:
        model.backbone.pyramid.load_state_dict(best_state["pyramid"])
        model.head.load_state_dict(best_state["head"])
    artifact = {
        "kind": "segment", "backbone_key": backbone_key,
        "pyramid_state_dict": model.backbone.pyramid.state_dict(), "head_state_dict": model.head.state_dict(),
    }
    artifact_key = None
    if not dry_run:
        buf = io.BytesIO()
        torch.save(artifact, buf)
        artifact_key = f"{artifact_prefix or s3_prefix}/head_artifact.pt"
        put_bytes(org_id, artifact_key, buf.getvalue(), "application/octet-stream")
    return {
        "artifact_key": artifact_key, "train_examples": len(train_examples), "eval_examples": len(eval_examples),
        "epoch_losses": epoch_losses, "val_losses": val_losses,
        "best_epoch": best_epoch if best_epoch is not None else len(epoch_losses),
        "best_val_loss": best_val if best_epoch is not None else None,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "epochs_run": len(epoch_losses),
        "stopped_early": monitor.stopped_early,
    }


def evaluate_segment(model: SegmentModel, examples: list[SegExample], iou_thresh: float = 0.5, score_thresh: float = 0.5) -> dict:
    """Instance metrics (AP50, precision, recall, mean IoU of matched objects), plus pixel IoU/Dice, the count
    error per tile and the area error of matched objects."""
    preds: list[tuple[int, int, float]] = []   # (image, instance id, score)
    gt_total, inter_px, union_px, pred_px, gt_px = 0, 0, 0, 0, 0
    count_err: list[int] = []
    per_image: list[tuple[np.ndarray, np.ndarray]] = []
    for ex in examples:
        inst, scores = masks_to_instances(predict_probs(model, ex.image), score_thresh)
        gt = label_map(ex.polygons, INPUT_SIZE).astype(np.int32)
        per_image.append((inst, gt))
        gt_total += len(ex.polygons)
        count_err.append(abs(len(scores) - len(ex.polygons)))
        pf, gf = inst > 0, gt > 0
        inter_px += int((pf & gf).sum()); union_px += int((pf | gf).sum()); pred_px += int(pf.sum()); gt_px += int(gf.sum())
        preds += [(len(per_image) - 1, i + 1, s) for i, s in enumerate(scores)]

    base = {"num_images": len(examples), "num_ground_truth": gt_total, "num_predictions": len(preds), "iou_thresh": iou_thresh, "score_thresh": score_thresh}
    if gt_total == 0:
        return {**base, "ap50": None, "note": "No ground-truth masks in this split."}
    pixel = {
        "pixel_iou": round(inter_px / union_px, 4) if union_px else None,
        "dice": round(2 * inter_px / (pred_px + gt_px), 4) if pred_px + gt_px else None,
        "count_mae": round(float(np.mean(count_err)), 3),
    }
    if not preds:
        return {**base, **pixel, "ap50": 0.0, "precision": 0.0, "recall": 0.0, "mean_iou_tp": None, "area_err_pct": None}

    taken: set[tuple[int, int]] = set()
    tp, matched_iou, area_err = [], [], []
    for img, pid, _score in sorted(preds, key=lambda p: -p[2]):
        inst, gt = per_image[img]
        pm = inst == pid
        best, best_g = 0.0, 0
        for g in np.unique(gt[pm]):
            if g == 0 or (img, int(g)) in taken:
                continue
            gm = gt == g
            iou = (pm & gm).sum() / max(1, (pm | gm).sum())
            if iou > best:
                best, best_g = float(iou), int(g)
        if best >= iou_thresh:
            taken.add((img, best_g))
            tp.append(1.0)
            matched_iou.append(best)
            area_err.append(abs(int(pm.sum()) - int((gt == best_g).sum())) / max(1, int((gt == best_g).sum())))
        else:
            tp.append(0.0)
    tp_a = np.array(tp)
    tp_cum, fp_cum = np.cumsum(tp_a), np.cumsum(1 - tp_a)
    recall = tp_cum / gt_total
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = sum(float(precision[recall >= t].max()) if (recall >= t).any() else 0.0 for t in np.linspace(0, 1, 101)) / 101
    return {
        **base, **pixel, "ap50": round(ap, 4), "precision": round(float(precision[-1]), 4), "recall": round(float(recall[-1]), 4),
        "mean_iou_tp": round(float(np.mean(matched_iou)), 4) if matched_iou else None,
        "area_err_pct": round(100 * float(np.mean(area_err)), 2) if area_err else None,
    }
