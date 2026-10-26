"""Segmentation inference for the platform's scan: a patch image in, one polygon per object out, in the same
{"label", "confidence", "geometry"} shape the detector returns, so ModelManager needs no change."""
from __future__ import annotations

import base64
import io
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from app.detect_model import INPUT_SIZE
from app.segment_model import SegmentModel, masks_to_instances, predict_probs

_SIMPLIFY_PX = 1.0


def _ring_to_geo(ring: np.ndarray, bbox: list[float]) -> list[list[float]]:
    ix0, iy0, ix1, iy1 = bbox
    sx, sy = ix1 - ix0, iy1 - iy0
    pts = [[ix0 + x / INPUT_SIZE * sx, iy1 - y / INPUT_SIZE * sy] for x, y in ring]
    return pts + [pts[0]]


def run_segment_inference(
    model: SegmentModel, image_base64: str, georef_metadata: dict[str, Any], *, class_label: str, score_thresh: float = 0.5,
) -> list[dict[str, Any]]:
    bbox = georef_metadata.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("georef_metadata.bbox [minx, miny, maxx, maxy] is required")
    img = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    tensor = torch.from_numpy(np.array(img)).float().div(255.0).permute(2, 0, 1)
    inst, scores = masks_to_instances(predict_probs(model, tensor), score_thresh)
    out: list[dict[str, Any]] = []
    for i, score in enumerate(scores, start=1):
        contours, _ = cv2.findContours((inst == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        ring = cv2.approxPolyDP(max(contours, key=cv2.contourArea), _SIMPLIFY_PX, True).reshape(-1, 2)
        if len(ring) < 3:
            continue
        out.append({"label": class_label, "confidence": float(score), "geometry": {"type": "Polygon", "coordinates": [_ring_to_geo(ring, bbox)]}})
    return out
