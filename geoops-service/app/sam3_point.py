"""SAM3 interactive point-prompt segmentation ("click an object, get its mask").

Uses ultralytics' ``SAM3Predictor`` (the SAM 2-style interactive head), separate
from the concept head in ``sam3_infer`` (text / box exemplars -> all
instances). Positive points (label 1) say "this is part of the object";
negative points (label 0) say "not this" — clicks are combined for ONE object,
so adding points refines the same mask instead of finding new objects.

The predictor is loaded once (a 3 GB+ checkpoint) and guarded by a lock: it is
stateful (``set_image`` caches the image embedding). Refinement clicks reuse
the same image, so the embedding is kept between calls when the image bytes are
unchanged — the second and later clicks are fast.
"""
from __future__ import annotations

import hashlib
import io
import logging
import threading

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.device import DEVICE

logger = logging.getLogger(__name__)

_predictor = None
_lock = threading.Lock()
_image_key: str | None = None

# Polygon simplification tolerance in pixels: removes staircase noise, keeps shape.
_SIMPLIFY_PX = 0.75


def _get_predictor():
    global _predictor
    if _predictor is None:
        from ultralytics.models.sam import SAM3Predictor  # noqa: PLC0415 — slow import, GPU-only path

        overrides = {
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "model": settings.SAM3_MODEL_PATH,
            "device": DEVICE,
            "save": False,
            "verbose": False,
        }
        logger.info("Loading SAM3 interactive predictor from %s (device=%s)", settings.SAM3_MODEL_PATH, DEVICE)
        _predictor = SAM3Predictor(overrides=overrides)
    return _predictor


def _rings_for_point(mask: np.ndarray, px: tuple[float, float]) -> list[list[list[float]]]:
    """Outer ring (+ its holes) of the mask component containing ``px``; the
    largest component if the point sits on none (e.g. a negative-only edge case)."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return []
    hier = hierarchy[0]
    outers = [i for i, h in enumerate(hier) if h[3] == -1]
    if not outers:
        return []

    def contains(i: int) -> bool:
        return cv2.pointPolygonTest(contours[i], px, False) >= 0

    chosen = next((i for i in outers if contains(i)), None)
    if chosen is None:
        chosen = max(outers, key=lambda i: cv2.contourArea(contours[i]))

    def ring(i: int) -> list[list[float]]:
        approx = cv2.approxPolyDP(contours[i], _SIMPLIFY_PX, True).reshape(-1, 2)
        return [[float(x), float(y)] for x, y in approx]

    rings = [ring(chosen)]
    rings += [ring(i) for i, h in enumerate(hier) if h[3] == chosen and cv2.contourArea(contours[i]) >= 4]
    return [r for r in rings if len(r) >= 3]


def run_sam3_points(image_bytes: bytes, points: list[list[float]], labels: list[int]) -> dict:
    """One object's mask from positive/negative click points in ``image_bytes``' pixel space.

    Returns ``{"rings": [...], "area_px", "score", "image_px": [w, h]}``; ``rings``
    is empty when SAM finds no object for these clicks.
    """
    global _image_key
    if not points or len(points) != len(labels):
        raise ValueError("points and labels must be the same non-zero length")
    if not any(lab == 1 for lab in labels):
        raise ValueError("at least one positive point is required")

    image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    h, w = image.shape[:2]
    key = hashlib.sha1(image_bytes).hexdigest()  # noqa: S324 — cache key, not security

    with _lock:
        predictor = _get_predictor()
        if key != _image_key:
            predictor.set_image(image)
            _image_key = key
        # Nested lists = several points prompting ONE object (refinement), not one object per point.
        results = predictor(points=[points], labels=[labels])

    result = results[0]
    if result.masks is None or len(result.masks.data) == 0:
        return {"rings": [], "area_px": 0.0, "score": None, "image_px": [w, h]}

    mask = result.masks.data[0].cpu().numpy()
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    mask_u8 = (mask > 0.5).astype(np.uint8)

    first_pos = next(p for p, lab in zip(points, labels, strict=True) if lab == 1)
    rings = _rings_for_point(mask_u8, (float(first_pos[0]), float(first_pos[1])))
    score = None
    if result.boxes is not None and getattr(result.boxes, "conf", None) is not None and len(result.boxes.conf):
        score = float(result.boxes.conf[0])
    return {"rings": rings, "area_px": float(mask_u8.sum()), "score": score, "image_px": [w, h]}


def _largest_rings(mask_u8: np.ndarray) -> list[list[list[float]]]:
    """Outer ring (+ holes) of a mask's largest connected component."""
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return []
    hier = hierarchy[0]
    outers = [i for i, h in enumerate(hier) if h[3] == -1]
    if not outers:
        return []
    chosen = max(outers, key=lambda i: cv2.contourArea(contours[i]))

    def ring(i: int) -> list[list[float]]:
        approx = cv2.approxPolyDP(contours[i], _SIMPLIFY_PX, True).reshape(-1, 2)
        return [[float(x), float(y)] for x, y in approx]

    rings = [ring(chosen)]
    rings += [ring(i) for i, h in enumerate(hier) if h[3] == chosen and cv2.contourArea(contours[i]) >= 4]
    return [r for r in rings if len(r) >= 3]


def run_sam3_tile(
    image_bytes: bytes,
    points_per_side: int = 16,
    min_score: float = 0.5,
    nms_iou: float = 0.7,
    max_area_frac: float = 0.6,
    min_area_px: int = 24,
) -> dict:
    """"Segment everything" in one image: prompt a regular grid of points, keep
    the confident masks, and collapse near-duplicates (every point inside one
    object yields the same mask) with a raster mask-IoU NMS on the GPU.

    Returns ``{"masks": [{"rings", "score", "bbox", "area_px"}], "image_px": [w, h]}``
    with ``bbox`` = [x0, y0, x1, y1] in pixels, so the caller can discard masks
    cut by the image border. Cross-tile / cross-scale cleaning is the caller's job.
    """
    import torch  # noqa: PLC0415 — keep torch off the import path of /health

    global _image_key
    n = max(4, min(int(points_per_side), 40))
    image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    h, w = image.shape[:2]
    key = hashlib.sha1(image_bytes).hexdigest()  # noqa: S324 — cache key, not security
    xs = np.linspace(w / (2 * n), w - w / (2 * n), n)
    ys = np.linspace(h / (2 * n), h - h / (2 * n), n)
    points = [[float(x), float(y)] for y in ys for x in xs]

    with _lock:
        predictor = _get_predictor()
        if key != _image_key:
            predictor.set_image(image)
            _image_key = key
        results = predictor(points=points, labels=[1] * len(points))
        result = results[0]
        if result.masks is None or len(result.masks.data) == 0 or result.boxes is None:
            return {"masks": [], "image_px": [w, h]}

        masks = result.masks.data > 0.5  # (K, H, W) bool, on the predictor's device
        scores = result.boxes.conf.to(masks.device).float()
        areas = masks.flatten(1).sum(1).float()
        keep = (scores >= min_score) & (areas >= min_area_px) & (areas <= max_area_frac * h * w)
        idx = torch.nonzero(keep).flatten()
        if idx.numel() == 0:
            return {"masks": [], "image_px": [w, h]}
        order = idx[torch.argsort(scores[idx], descending=True)]

        flat = masks[order].flatten(1).half()  # (K, HW)
        inter = flat @ flat.T
        area = flat.sum(1)
        iou = inter / (area[:, None] + area[None, :] - inter).clamp(min=1)
        iou_cpu = iou.float().cpu().numpy()
        kept: list[int] = []
        suppressed = np.zeros(len(order), dtype=bool)
        for i in range(len(order)):
            if suppressed[i]:
                continue
            kept.append(i)
            suppressed |= iou_cpu[i] > nms_iou
        final = order[torch.as_tensor(kept, device=order.device)]
        final_masks = masks[final].cpu().numpy().astype(np.uint8)
        final_scores = scores[final].cpu().numpy()

    out = []
    for m, sc in zip(final_masks, final_scores, strict=True):
        rings = _largest_rings(m)
        if not rings:
            continue
        ys_, xs_ = np.nonzero(m)
        out.append(
            {
                "rings": rings,
                "score": float(sc),
                "bbox": [int(xs_.min()), int(ys_.min()), int(xs_.max()), int(ys_.max())],
                "area_px": int(m.sum()),
            }
        )
    return {"masks": out, "image_px": [w, h]}


def run_sam3_boxes_masks(image_bytes: bytes, boxes: list[list[float]]) -> dict:
    """One mask per box prompt (each box = one object), in ``image_bytes``' pixel space.

    ``boxes`` are [x0, y0, x1, y1] pixels. Returns ``{"results": [...], "image_px": [w, h]}`` where
    ``results[i]`` belongs to ``boxes[i]``: ``{"rings", "score", "area_px", "bbox"}``, or ``None`` when
    SAM finds nothing for that box. The image embedding is computed once for all boxes.
    """
    global _image_key
    if not boxes:
        raise ValueError("at least one box is required")
    image = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    h, w = image.shape[:2]
    key = hashlib.sha1(image_bytes).hexdigest()  # noqa: S324 — cache key, not security

    with _lock:
        predictor = _get_predictor()
        if key != _image_key:
            predictor.set_image(image)
            _image_key = key
        results = predictor(bboxes=[[float(v) for v in b] for b in boxes])
        result = results[0]
        masks = None if result.masks is None else result.masks.data.cpu().numpy()
        scores = None
        if result.boxes is not None and getattr(result.boxes, "conf", None) is not None:
            scores = result.boxes.conf.cpu().numpy()

    out: list[dict | None] = []
    for i, b in enumerate(boxes):
        if masks is None or i >= len(masks):
            out.append(None)
            continue
        m = masks[i]
        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        m_u8 = (m > 0.5).astype(np.uint8)
        rings = _rings_for_point(m_u8, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        if not rings:
            out.append(None)
            continue
        ys_, xs_ = np.nonzero(m_u8)
        out.append(
            {
                "rings": rings,
                "score": float(scores[i]) if scores is not None and i < len(scores) else None,
                "area_px": int(m_u8.sum()),
                "bbox": [int(xs_.min()), int(ys_.min()), int(xs_.max()), int(ys_.max())],
            }
        )
    return {"results": out, "image_px": [w, h]}


__all__ = ["run_sam3_boxes_masks", "run_sam3_points", "run_sam3_tile"]
