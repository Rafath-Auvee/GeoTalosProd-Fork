"""Segmentation head on the frozen DINOv2 feature pyramid.

Same backbone and pyramid as the detector (``Dinov2FeaturePyramidBackbone``); the FCOS head is replaced by a
small decoder that fuses the four pyramid levels, upsamples, and sharpens with a shallow RGB path (DINOv2's
14 px patches alone would blur the outlines). It predicts two maps at ``OUT_SIZE`` px:

  * foreground  — is this pixel part of an object of the class
  * boundary    — is it on an object's edge (also between touching objects)

Instances are the connected pieces of foreground minus boundary (``masks_to_instances``), so touching objects
stay separate and each one has its own mask, count and area.

``SegmentModel`` exposes ``.backbone`` and ``.head`` like the detector, so the trainer's weight snapshot and the
loader treat both alike.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from app.detect_model import INPUT_SIZE
from app.device import DEVICE
from app.dinov2_backbone import Dinov2FeaturePyramidBackbone

OUT_SIZE = 148  # INPUT_SIZE / 3.5: about 7 cm per output pixel on a 10 m tile
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _conv(cin: int, cout: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2), nn.GroupNorm(8, cout), nn.ReLU(inplace=True))


class SegHead(nn.Module):
    def __init__(self, in_channels: int = 256, mid: int = 64):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(in_channels, mid, 1) for _ in range(4)])
        self.fuse = _conv(mid, mid)
        self.rgb = _conv(3, 32)
        self.out = nn.Sequential(_conv(mid + 32, 64), nn.Conv2d(64, 2, 1))

    def forward(self, feats: list[torch.Tensor], image_small: torch.Tensor) -> torch.Tensor:
        size = feats[0].shape[-2:]
        x = sum(F.interpolate(p(f), size=size, mode="bilinear", align_corners=False) for p, f in zip(self.proj, feats, strict=True))
        x = F.interpolate(self.fuse(x), size=image_small.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat([x, self.rgb(image_small)], dim=1))  # (B, 2, OUT, OUT): foreground, boundary logits


class SegmentModel(nn.Module):
    def __init__(self, backbone_key: str):
        super().__init__()
        self.backbone = Dinov2FeaturePyramidBackbone(backbone_key)
        self.head = SegHead(self.backbone.out_channels)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def normalise(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std

    def small(self, x: torch.Tensor) -> torch.Tensor:
        # 2x2 average, then a bilinear resize: close to area resampling, and exportable to ONNX (adaptive pooling is not)
        return F.interpolate(F.avg_pool2d(x, 2), size=(OUT_SIZE, OUT_SIZE), mode="bilinear", align_corners=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, 3, 518, 518), 0-1 RGB -> (B, 2, OUT, OUT) logits."""
        x = self.normalise(image)
        feats = list(self.backbone.pyramid(self.backbone.extract_tokens(x)).values())
        return self.head(feats, self.small(x))


def build_segment(backbone_key: str) -> SegmentModel:
    return SegmentModel(backbone_key).to(DEVICE)


def label_map(polygons: list[np.ndarray], size: int) -> np.ndarray:
    """Instance-id map (uint16, 0 = background) from polygons given as (N, 2) arrays in 0..1 coordinates."""
    lab = np.zeros((size, size), dtype=np.uint16)
    for i, poly in enumerate(polygons, start=1):
        pts = np.round(np.asarray(poly, dtype=np.float32) * size).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(lab, [pts], i)
    return lab


def targets_from_labels(lab: np.ndarray) -> np.ndarray:
    """(2, H, W) uint8 training target: foreground, and boundary = where the instance id changes."""
    kernel = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(lab, kernel) != cv2.erode(lab, kernel)
    return np.stack([lab > 0, edge]).astype(np.uint8)


def masks_to_instances(
    probs: np.ndarray, score_thresh: float = 0.5, min_area_px: int = 40
) -> tuple[np.ndarray, list[float]]:
    """probs: (2, H, W) foreground and boundary probabilities at full size.
    Returns (instance map with ids 1..N, one score per id: the mean foreground probability)."""
    fg_p, edge_p = probs[0], probs[1]
    fg = fg_p >= score_thresh
    core = (fg & (edge_p < 0.5)).astype(np.uint8)
    n, comp = cv2.connectedComponents(core)
    markers = np.zeros(fg.shape, np.int32)
    markers[~fg] = 1                      # background marker
    keep = [i for i in range(1, n) if (comp == i).sum() >= min_area_px // 2]
    for new_id, i in enumerate(keep, start=2):
        markers[comp == i] = new_id
    img = cv2.cvtColor((255 * (1 - fg_p)).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(img, markers)           # grows each core into its boundary pixels, without crossing into a neighbour
    inst = np.zeros(fg.shape, np.int32)
    scores: list[float] = []
    for new_id in range(2, 2 + len(keep)):
        m = markers == new_id
        if m.sum() < min_area_px:
            continue
        scores.append(float(fg_p[m].mean()))
        inst[m] = len(scores)
    return inst, scores


def predict_probs(model: SegmentModel, image: torch.Tensor) -> np.ndarray:
    """image (3, 518, 518) 0-1 -> (2, 518, 518) probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(image[None].to(DEVICE))
        up = F.interpolate(logits, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        return torch.sigmoid(up)[0].cpu().numpy()
