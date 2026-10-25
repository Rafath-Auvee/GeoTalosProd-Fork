"""geoops-service — GPU model-training/serving microservice.

Serves as the single generic backend for every backbone+head combination
AwakeForestProd's geoops.* CPU orchestration calls out to (DINOv2 today,
YOLO/others later) — one set of routes, config-driven per request, rather
than a service per backbone. /train (detect/FCOS only so far) is real;
/infer and /eval are still stubs.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Literal

# Nothing else in this service configures logging, so without this,
# logging.info/.warning calls — including app.device's device-selection log,
# which fires at import time below — are silently dropped by the root
# logger's default WARNING level. Must run before importing app.device (the
# one thing you'd actually want visible in container logs to confirm GPU vs.
# CPU at a glance).
logging.basicConfig(level=logging.INFO)

import torch
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import require_bearer_token
from app.backbones import BACKBONES
from app.device import DEVICE

# PyTorch defaults to intra-op parallelism across every available core. That's
# right for one big batch on a beefy machine, wrong for this tier's actual
# workload: tiny per-class heads, batches of 4, on hosts that can have dozens
# of cores. Coordinating that many threads per matmul costs more than the
# matmul itself — observed directly: a training run pegged 51 of 64 cores
# (5148% CPU) while making barely any progress. Cap it once, at process
# startup, before any tensor op runs. Still relevant on GPU: CPU-side work
# (image decode, tensor collation, the Celery-called HTTP handling itself)
# runs on these same threads regardless of where the model executes.
torch.set_num_threads(min(8, os.cpu_count() or 8))

logger = logging.getLogger(__name__)
logger.info("geoops-service starting on device=%s", DEVICE)

app = FastAPI(title="geoops-service")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/yolo/bases", dependencies=[Depends(require_bearer_token)])
async def yolo_bases() -> list[dict]:
    """Pretrained YOLO weights files available to start a training from."""
    from app.yolo_train import list_bases  # noqa: PLC0415

    return list_bases()


@app.get("/backbones", dependencies=[Depends(require_bearer_token)])
async def list_backbones() -> dict[str, dict]:
    return BACKBONES


class TrainRequest(BaseModel):
    org_id: str
    backbone_key: str = "dinov2-vits14"
    kind: Literal["fcos", "classify", "segment", "yolo"] = "fcos"
    s3_prefix: str  # e.g. "training-runs/{job_id}" — same layout dino_tasks.py exports to
    epochs: int = 60
    lr: float = 1e-3
    artifact_prefix: str | None = None  # where the trained model is saved; defaults to s3_prefix (the data folder)
    base_weights: str | None = None  # yolo: a name in models/yolo, or an earlier model's storage key
    stage1_epochs: int = 40
    stage2_epochs: int = 40
    stage1_lr: float = 0.002
    stage2_lr: float = 0.0001
    imgsz: int = 640  # yolo: the dataset's tile size, so tiles are not resized
    seed: int | None = None  # None = unseeded (the previous behaviour)
    weight_decay: float = 0.01  # AdamW, fcos / classify / segment
    dry_run: bool = False  # score the config only: no artifact is written (used by tuning trials)
    patience: int | None = Field(default=None, ge=1)  # DINOv2 heads: stop when the best epoch is this many epochs ago
    min_epochs: int = Field(default=0, ge=0)  # ... but never before this many epochs
    tune_trials: int = Field(default=0, ge=0, le=50)  # >0: Optuna search over lr / epochs / weight decay (yolo: the two stage lrs)
    seeds: list[int] = Field(default_factory=list, max_length=10)  # non-empty: train once per seed, evaluate each on the test split


class InferRequest(BaseModel):
    """Matches app.services.model_manager.ModelManager._call_model's generic
    body exactly (dataset_item_id, georef_metadata, patch_image_base64, ...)
    plus fields injected via this head's ai_models.request_config.payload
    (org_id, kind, backbone_key, artifact_key, class_label, score_thresh) —
    that's how a per-class head's identity reaches this endpoint without
    ModelManager needing any DINOv2-specific code.
    """

    model_config = {"extra": "allow"}  # ModelManager sends more fields than we read

    georef_metadata: dict[str, Any]
    patch_image_base64: str
    org_id: str
    artifact_key: str
    class_label: str
    kind: Literal["fcos", "classify", "segment", "yolo"] = "fcos"
    backbone_key: str = "dinov2-vits14"
    score_thresh: float = 0.3


class EmbedRequest(BaseModel):
    """Matches geoops.tasks._call_embed_model / geoops.service.EmbeddingService's
    generic embed-model body (dataset_item_id, stac_item_id, bbox,
    patch_image_base64, ...) plus fields injected via this model's
    ai_models.request_config.payload (backbone_key) — same pattern as
    InferRequest above.
    """

    model_config = {"extra": "allow"}

    patch_image_base64: str
    backbone_key: str = "dinov2-vits14"


class EmbedBatchRequest(BaseModel):
    """Many small crops -> DINOv2 CLS embeddings, batched (Interactive annotation grouping)."""

    backbone_key: str = "dinov2-vits14"
    images_base64: list[str]
    size: int = 224


class PointPromptRequest(BaseModel):
    """One object from click points. ``points`` are [x, y] pixels within
    ``patch_image_base64`` itself; ``labels`` are 1 (positive) / 0 (negative)."""

    patch_image_base64: str
    points: list[list[float]]
    labels: list[int]


class TileSegmentRequest(BaseModel):
    """"Segment everything" in one image: a grid of point prompts, confident
    masks kept, near-duplicates collapsed. See app.sam3_point.run_sam3_tile."""

    patch_image_base64: str
    points_per_side: int = 16
    min_score: float = 0.5
    nms_iou: float = 0.7


class AutoAnnotateRequest(BaseModel):
    """See geoops.sam3_tasks.run_sam3_auto_annotate — one box(+text) SAM3
    prompt against a single titiler crop. ``bboxes`` are xyxy pixel
    coordinates within ``patch_image_base64`` itself (never another image's
    coordinates — SAM3's box prompt is single-image only, see
    app/sam3_infer.py's docstring). ``text`` is hardcoded to "crop" by the
    caller for now; kept as a field (not literal-typed) so that hardcoding
    can move to the caller side without a contract change here.
    """

    patch_image_base64: str
    bboxes: list[list[float]] = []
    text: str = "crop"
    conf: float = 0.3


class EvalRequest(BaseModel):
    org_id: str
    s3_prefix: str  # manifest.json lives here — same layout /train reads from
    artifact_key: str | None = None  # defaults to f"{s3_prefix}/head_artifact.pt"
    kind: Literal["fcos", "classify", "segment", "yolo"] = "fcos"
    split: Literal["train", "eval", "test"] = "test"
    iou_thresh: float = 0.5
    score_thresh: float = 0.5


class ExportOnnxRequest(BaseModel):
    org_id: str
    artifact_key: str
    kind: Literal["fcos", "classify", "segment", "yolo"] = "fcos"
    out_key: str


@app.post("/export-onnx", dependencies=[Depends(require_bearer_token)])
async def export_onnx_endpoint(payload: ExportOnnxRequest) -> dict:
    """Export a trained head and its frozen backbone as one ONNX file, stored at ``out_key``."""
    try:
        from app.load_model import load_trained_classify_model, load_trained_detect_model  # noqa: PLC0415
        from app.onnx_export import export_onnx  # noqa: PLC0415
        from app.storage import put_bytes  # noqa: PLC0415

        from app.load_model import load_trained_segment_model  # noqa: PLC0415

        if payload.kind == "yolo":
            import asyncio  # noqa: PLC0415

            from app.yolo_train import export_yolo_onnx  # noqa: PLC0415

            data = await asyncio.to_thread(export_yolo_onnx, payload.org_id, payload.artifact_key)
        else:
            loader = {"classify": load_trained_classify_model, "segment": load_trained_segment_model}.get(payload.kind, load_trained_detect_model)
            data = export_onnx(loader(payload.org_id, payload.artifact_key), payload.kind)
        put_bytes(payload.org_id, payload.out_key, data, "application/octet-stream")
        return {"out_key": payload.out_key, "size_bytes": len(data)}
    except Exception:
        logger.exception("onnx_export_failed org_id=%s artifact=%s", payload.org_id, payload.artifact_key)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ONNX export failed") from None


def _train_once(payload: TrainRequest, on_epoch=None) -> dict:
    """One plain training run of ``payload.kind``. Sync — callers run it in a worker thread.

    ``on_epoch(epoch, val_loss)`` is the tuner's per-epoch hook (DINOv2 heads; YOLO trains through Ultralytics and ignores it)."""
    if payload.kind == "yolo":
        from app.yolo_train import run_yolo_training  # noqa: PLC0415

        return run_yolo_training(
            org_id=payload.org_id, s3_prefix=payload.s3_prefix, base_weights=payload.base_weights,
            stage1_epochs=payload.stage1_epochs, stage2_epochs=payload.stage2_epochs,
            lr1=payload.stage1_lr, lr2=payload.stage2_lr, imgsz=payload.imgsz,
            artifact_prefix=payload.artifact_prefix or payload.s3_prefix, seed=payload.seed, dry_run=payload.dry_run,
        )
    common = dict(
        org_id=payload.org_id, backbone_key=payload.backbone_key, s3_prefix=payload.s3_prefix, epochs=payload.epochs,
        lr=payload.lr, artifact_prefix=payload.artifact_prefix, seed=payload.seed,
        weight_decay=payload.weight_decay, dry_run=payload.dry_run,
        patience=payload.patience, min_epochs=payload.min_epochs, on_epoch=on_epoch,
    )
    if payload.kind == "segment":
        from app.segment_train import run_segment_training  # noqa: PLC0415

        return run_segment_training(**common)
    if payload.kind == "classify":
        from app.classify_train import run_classify_training  # noqa: PLC0415

        return run_classify_training(**common)
    from app.train import run_detect_training  # noqa: PLC0415 — lazy: keeps torch import off /health's path

    return run_detect_training(**common)


def _train_auto(payload: TrainRequest) -> dict:
    """Optuna tuning and / or one run per seed around ``_train_once`` (see app.tuning)."""
    from app.tuning import run_auto  # noqa: PLC0415

    def train_fn(overrides: dict, on_epoch=None) -> dict:
        return _train_once(payload.model_copy(update=overrides), on_epoch)

    def eval_fn(artifact_key: str) -> dict:
        return _eval_once(EvalRequest(
            org_id=payload.org_id, s3_prefix=payload.s3_prefix, artifact_key=artifact_key, kind=payload.kind, split="test",
        ))

    return run_auto(
        payload.kind, payload.model_dump(), train_fn, eval_fn, payload.artifact_prefix or payload.s3_prefix,
        payload.tune_trials, payload.seeds,
    )


@app.post("/train", dependencies=[Depends(require_bearer_token)])
async def train(payload: TrainRequest) -> dict:
    import asyncio  # noqa: PLC0415

    try:
        # In a thread so a long run (a tune or multi-seed run can take a while) doesn't block /health.
        fn = _train_auto if (payload.tune_trials > 0 or payload.seeds) else _train_once
        return await asyncio.to_thread(fn, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("train_failed org_id=%s s3_prefix=%s", payload.org_id, payload.s3_prefix)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Training failed") from None


@app.post("/infer", dependencies=[Depends(require_bearer_token)])
async def infer(payload: InferRequest) -> dict:
    try:
        if payload.kind == "yolo":
            from app.yolo_train import load_trained_yolo_model, run_yolo_inference  # noqa: PLC0415

            return {"predictions": run_yolo_inference(
                load_trained_yolo_model(payload.org_id, payload.artifact_key), payload.patch_image_base64, payload.georef_metadata,
                score_thresh=payload.score_thresh,
            )}
        if payload.kind == "segment":
            from app.load_model import load_trained_segment_model  # noqa: PLC0415
            from app.segment_infer import run_segment_inference  # noqa: PLC0415

            model = load_trained_segment_model(payload.org_id, payload.artifact_key)
            return {"predictions": run_segment_inference(
                model, payload.patch_image_base64, payload.georef_metadata,
                class_label=payload.class_label, score_thresh=payload.score_thresh,
            )}
        if payload.kind == "classify":
            from app.classify_infer import run_classify_inference  # noqa: PLC0415
            from app.load_model import load_trained_classify_model  # noqa: PLC0415

            model = load_trained_classify_model(payload.org_id, payload.artifact_key)
            predictions = run_classify_inference(
                model,
                payload.patch_image_base64,
                payload.georef_metadata,
                class_label=payload.class_label,
                score_thresh=payload.score_thresh,
            )
        else:
            from app.infer_model import run_detect_inference  # noqa: PLC0415
            from app.load_model import load_trained_detect_model  # noqa: PLC0415

            model = load_trained_detect_model(payload.org_id, payload.artifact_key)
            predictions = run_detect_inference(
                model,
                payload.patch_image_base64,
                payload.georef_metadata,
                class_label=payload.class_label,
                score_thresh=payload.score_thresh,
            )
        return {"predictions": predictions}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("infer_failed org_id=%s artifact_key=%s", payload.org_id, payload.artifact_key)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference failed") from None


@app.post("/embed", dependencies=[Depends(require_bearer_token)])
async def embed(payload: EmbedRequest) -> dict:
    try:
        from app.dinov2_backbone import extract_cls_embedding  # noqa: PLC0415 — lazy: keeps torch off /health's path

        vector = extract_cls_embedding(payload.backbone_key, payload.patch_image_base64)
        return {"embedding": vector, "embedding_dim": len(vector), "model_name": payload.backbone_key}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("embed_failed backbone_key=%s", payload.backbone_key)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Embedding failed") from None


@app.post("/embed/batch", dependencies=[Depends(require_bearer_token)])
async def embed_batch(payload: EmbedBatchRequest) -> dict:
    if not payload.images_base64:
        return {"embeddings": [], "embedding_dim": 0, "model_name": payload.backbone_key}
    if payload.size % 14 != 0 or not (56 <= payload.size <= 518):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="size must be a multiple of 14 between 56 and 518")
    try:
        from app.dinov2_backbone import extract_cls_embeddings_batch  # noqa: PLC0415 — keeps torch off /health's path

        vectors = extract_cls_embeddings_batch(payload.backbone_key, payload.images_base64, payload.size)
        return {"embeddings": vectors, "embedding_dim": len(vectors[0]), "model_name": payload.backbone_key}
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("embed_batch_failed n=%d", len(payload.images_base64))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Batch embedding failed") from None


@app.post("/autoannotate", dependencies=[Depends(require_bearer_token)])
async def autoannotate(payload: AutoAnnotateRequest) -> dict:
    try:
        import base64  # noqa: PLC0415 — lazy, keeps torch/ultralytics off /health's path

        from app.sam3_infer import run_sam3_boxes  # noqa: PLC0415

        image_bytes = base64.b64decode(payload.patch_image_base64)
        boxes = run_sam3_boxes(image_bytes, payload.bboxes, payload.text, payload.conf)
        return {"boxes": boxes}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("autoannotate_failed text=%s conf=%s", payload.text, payload.conf)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Auto-annotate failed") from None


class BoxesMasksRequest(BaseModel):
    patch_image_base64: str
    boxes: list[list[float]]  # xyxy pixels in the image


@app.post("/sam3/boxes-masks", dependencies=[Depends(require_bearer_token)])
async def sam3_boxes_masks(payload: BoxesMasksRequest) -> dict:
    """A mask for each box prompt in one image (SAM 3 interactive head)."""
    try:
        import base64  # noqa: PLC0415

        from app.sam3_point import run_sam3_boxes_masks  # noqa: PLC0415

        return run_sam3_boxes_masks(base64.b64decode(payload.patch_image_base64), payload.boxes)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("sam3_boxes_masks_failed boxes=%d", len(payload.boxes))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Box segmentation failed") from None


@app.post("/sam3/point", dependencies=[Depends(require_bearer_token)])
async def sam3_point(payload: PointPromptRequest) -> dict:
    try:
        import base64  # noqa: PLC0415 — lazy, keeps torch/ultralytics off /health's path

        from app.sam3_point import run_sam3_points  # noqa: PLC0415

        return run_sam3_points(base64.b64decode(payload.patch_image_base64), payload.points, payload.labels)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("sam3_point_failed n_points=%d", len(payload.points))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Point segmentation failed") from None


@app.post("/sam3/segment-tile", dependencies=[Depends(require_bearer_token)])
async def sam3_segment_tile(payload: TileSegmentRequest) -> dict:
    try:
        import base64  # noqa: PLC0415

        from app.sam3_point import run_sam3_tile  # noqa: PLC0415

        return run_sam3_tile(
            base64.b64decode(payload.patch_image_base64),
            points_per_side=payload.points_per_side,
            min_score=payload.min_score,
            nms_iou=payload.nms_iou,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception("sam3_segment_tile_failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Tile segmentation failed") from None


def _eval_once(payload: EvalRequest) -> dict:
    artifact_key = payload.artifact_key or f"{payload.s3_prefix}/head_artifact.pt"
    if payload.kind == "yolo":
        from app.yolo_train import evaluate_yolo  # noqa: PLC0415

        return evaluate_yolo(payload.org_id, artifact_key, payload.s3_prefix, payload.split, payload.iou_thresh)
    if payload.kind == "segment":
        from app.load_model import load_trained_segment_model  # noqa: PLC0415
        from app.segment_train import evaluate_segment, load_segment_split  # noqa: PLC0415

        model = load_trained_segment_model(payload.org_id, artifact_key)
        examples = load_segment_split(payload.org_id, payload.s3_prefix, payload.split)
        return evaluate_segment(model, examples, iou_thresh=payload.iou_thresh, score_thresh=payload.score_thresh)
    if payload.kind == "classify":
        from app.classify_dataset import load_classify_split  # noqa: PLC0415
        from app.classify_eval import evaluate_classify  # noqa: PLC0415
        from app.load_model import load_trained_classify_model  # noqa: PLC0415

        model = load_trained_classify_model(payload.org_id, artifact_key)
        examples = load_classify_split(payload.org_id, payload.s3_prefix, payload.split)
        return evaluate_classify(model, examples, score_thresh=payload.score_thresh)

    from app.dataset import load_split  # noqa: PLC0415
    from app.eval_model import evaluate_detect  # noqa: PLC0415
    from app.load_model import load_trained_detect_model  # noqa: PLC0415

    model = load_trained_detect_model(payload.org_id, artifact_key)
    examples = load_split(payload.org_id, payload.s3_prefix, payload.split)
    return evaluate_detect(model, examples, iou_thresh=payload.iou_thresh)


@app.post("/eval", dependencies=[Depends(require_bearer_token)])
async def eval_head(payload: EvalRequest) -> dict:
    artifact_key = payload.artifact_key or f"{payload.s3_prefix}/head_artifact.pt"
    try:
        import asyncio  # noqa: PLC0415

        return await asyncio.to_thread(_eval_once, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        logger.exception(
            "eval_failed org_id=%s s3_prefix=%s artifact_key=%s", payload.org_id, payload.s3_prefix, artifact_key
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Eval failed") from None
