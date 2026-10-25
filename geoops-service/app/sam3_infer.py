"""SAM3 box+text ("semantic") inference — the concrete model call
``geoops.sam3_tasks.run_sam3_auto_annotate`` hits via ``POST /autoannotate``.

Unlike the DINOv2 tier's per-request model rebuild, SAM3 is loaded once from
a local checkpoint (``settings.SAM3_MODEL_PATH`` — mounted from
``/home/prass25/projects/GreenMark/models/sam3.pt`` in docker-compose.yml,
not a training artifact this service produces itself) and cached in process
memory: it's a single fixed 3B+ param model, not one-per-(org, artifact) like
``load_model.py``'s trained heads.

``SAM3SemanticPredictor``'s box-prompt API is single-image only — a box is
always in the *same* image's pixel space it's paired with (see
``ultralytics.models.sam.predict.SAM3SemanticPredictor._prepare_geometric_prompts``,
which normalizes the box against ``self.src_shape``, the image just set via
``set_image``). There is no cross-image "reference crop" concept at the
model level, which is why the caller (``geoops.sam3_tasks``) always supplies
both the image *and* the box exemplar from the same source annotation/tile.
"""
from __future__ import annotations

import io
import logging
import threading

import numpy as np
from PIL import Image

from app.config import settings
from app.device import DEVICE

logger = logging.getLogger(__name__)

_predictor = None
_predictor_lock = threading.Lock()


def _get_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor
    with _predictor_lock:
        if _predictor is None:
            from ultralytics.models.sam import SAM3SemanticPredictor  # noqa: PLC0415 — slow import, GPU-only path

            overrides = {
                "conf": 0.3,
                "task": "segment",
                "mode": "predict",
                "model": settings.SAM3_MODEL_PATH,
                "device": DEVICE,
            }
            logger.info("Loading SAM3 model from %s (device=%s)", settings.SAM3_MODEL_PATH, DEVICE)
            _predictor = SAM3SemanticPredictor(overrides=overrides)
    return _predictor


def run_sam3_boxes(
    image_bytes: bytes, bboxes: list[list[float]], text: str, conf: float
) -> list[dict]:
    """Run one box+text (or text-only, if ``bboxes`` is empty) SAM3 prompt
    against a single image. Returns bounding boxes only — ``result.boxes``,
    never ``result.masks`` — per this feature's "boxes, not masks" scope.

    ``bboxes`` are xyxy pixel coordinates within ``image_bytes`` itself (see
    module docstring — never a different image's coordinates).
    """
    predictor = _get_predictor()
    predictor.args.conf = conf

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    predictor.set_image(np.asarray(image))

    kwargs: dict = {"text": [text]}
    if bboxes:
        kwargs["bboxes"] = bboxes

    results = predictor(**kwargs)
    predictor.reset_image()

    boxes: list[dict] = []
    for result in results:
        if result.boxes is None:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for (x0, y0, x1, y1), score in zip(xyxy, scores):
            boxes.append({"bbox": [float(x0), float(y0), float(x1), float(y1)], "score": float(score)})
    return boxes


__all__ = ["run_sam3_boxes"]
