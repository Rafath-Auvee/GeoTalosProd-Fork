"""Export a trained head (plus its frozen DINOv2 backbone) to a single ONNX file.

Detector: input ``image`` float32 [1, 3, 518, 518], RGB scaled to 0-1. Outputs ``boxes`` [N, 4]
(x1, y1, x2, y2 in those 518 px) and ``scores`` [N] for every anchor position. Decoding is baked in,
so the caller only applies a score threshold and NMS (the same two steps the platform applies).

Classifier: input the same; output ``score`` [1], the probability that the class is present.

Preprocessing matches what training and the platform's inference use: ImageNet normalisation is part
of the detector graph, so the caller only divides pixels by 255.
"""
from __future__ import annotations

import io
import math

import torch
from torch import nn
from torchvision.models.detection.image_list import ImageList

from app.detect_model import INPUT_SIZE
from app.device import DEVICE

OPSET = 17
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _DetectWrapper(nn.Module):
    def __init__(self, fcos: nn.Module):
        super().__init__()
        self.fcos = fcos
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        with torch.no_grad():
            probe = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)
            feats = list(fcos.backbone((probe - self.mean.to(DEVICE)) / self.std.to(DEVICE)).values())
            anchors = fcos.anchor_generator(ImageList(probe, [(INPUT_SIZE, INPUT_SIZE)]), feats)[0]
        self.register_buffer("anchors", anchors)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = (image - self.mean) / self.std
        feats = list(self.fcos.backbone(x).values())
        out = self.fcos.head(feats)
        boxes = self.fcos.box_coder.decode(out["bbox_regression"][0], self.anchors)
        boxes = boxes.clamp(0, INPUT_SIZE)
        scores = torch.sqrt(torch.sigmoid(out["cls_logits"][0, :, 0]) * torch.sigmoid(out["bbox_ctrness"][0, :, 0]))
        return boxes, scores


class _ClassifyWrapper(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(image))


class _SegmentWrapper(nn.Module):
    """Image in, foreground and boundary probabilities out (1, 2, 518, 518); the caller turns them into objects."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model(image)
        return torch.sigmoid(torch.nn.functional.interpolate(logits, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False))


def export_onnx(model: nn.Module, kind: str) -> bytes:
    """ONNX bytes for a loaded (eval-mode) trained model."""
    wrapper = {"fcos": _DetectWrapper, "segment": _SegmentWrapper}.get(kind, _ClassifyWrapper)(model)
    wrapper.to(DEVICE).eval()
    dummy = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)
    outputs = {"fcos": ["boxes", "scores"], "segment": ["probs"]}.get(kind, ["score"])
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, buf, opset_version=OPSET, input_names=["image"], output_names=outputs,
            do_constant_folding=True, dynamo=False,
        )
    return buf.getvalue()


def n_candidates() -> int:
    """Anchor count for the 518 px input (levels 74², 37², 19², 10²) — informational."""
    return sum(math.ceil(INPUT_SIZE / 14 * f) ** 2 for f in (2, 1, 0.5, 0.25))
