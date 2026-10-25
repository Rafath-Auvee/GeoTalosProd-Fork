"""Celery task: "Segment all" for Interactive annotation mode.

Tiles a small AOI at each requested real-world scale, asks geoops-service's
SAM 3 point-grid endpoint (``/sam3/segment-tile``) for every tile's masks, and
returns one cleaned set of georeferenced polygons.

Per tile, the rendered window is the tile plus a 25% margin on each side, so an
object near the tile edge is whole inside the window; only masks whose centre
is in the *core* tile are kept (neighbouring tiles keep their own), and masks
touching the window border are dropped. Across tiles and scales, duplicates are
removed with the same greedy IoU NMS the inference pipeline uses
(``ModelManager._merge_overlapping_predictions``), with the user's reference
objects in the pool at top priority so auto masks that re-find them are dropped.

Results are written to object storage after every scale (``masks.json``), so the
UI can show a finished scale while the next one runs. Sync (WorkerSession).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np
from PIL import Image

from shapely.geometry import Polygon, box, mapping, shape
from sqlalchemy import select

from app.core.enums import JobStatus
from app.models.dataset_item import DatasetItem
from app.models.job import Job
from app.services import storage_service
from app.workers.celery_app import celery_app
from app.workers.db import WorkerSession
from app.workers.queues import TRAINING

from geoops.dino_tasks import _call_geoops_service, _now
from geoops.export_common import resolve_rescale_params
from geoops.scale import SIZE_FRAC_RANGE, bbox_coords_span_m, canonical_tiles, meters_to_degrees
from geoops.yolo_tasks import _crop_dims, _fetch_crop_png

logger = logging.getLogger(__name__)

# Interactive mode is for small AOIs; both limits are enforced at the API and here.
MAX_AOI_HECTARES = 50.0
MAX_TILES = 2500
WINDOW_MARGIN = 0.25  # of the tile edge, on each side
RENDER_MAX_PX = 1024
REF_PRIORITY = 10.0  # confidence given to reference objects in the NMS pool: they always win


def masks_key(job_id: uuid.UUID | str) -> str:
    return f"interactive/segment-all/{job_id}/masks.json"


def plan_tiles(item_bbox: list[float], aoi_bbox: list[float], aoi_poly, scales: list[float]) -> dict[float, list[tuple[int, int, list[float]]]]:
    """Tiles per scale: the canonical grid over AOI ∩ item, kept where it touches the AOI shape."""
    clip = [max(item_bbox[0], aoi_bbox[0]), max(item_bbox[1], aoi_bbox[1]), min(item_bbox[2], aoi_bbox[2]), min(item_bbox[3], aoi_bbox[3])]
    if clip[0] >= clip[2] or clip[1] >= clip[3]:
        return {s: [] for s in scales}
    plan: dict[float, list[tuple[int, int, list[float]]]] = {}
    for s in scales:
        grid = canonical_tiles(item_bbox, 0, clip_bbox=clip, tiers=[s])
        plan[s] = [t for t in grid if box(*t[2]).intersects(aoi_poly)]
    return plan


def _merge(masks: list[dict[str, Any]], refs: list[dict[str, Any]], iou: float) -> list[dict[str, Any]]:
    """Existing NMS over (references + masks). References outrank everything and are removed afterwards."""
    from app.services.model_manager import ModelManager  # noqa: PLC0415 — heavy import, worker only

    pool = [{"class_id": None, "geom_4326": r["geom"], "confidence": REF_PRIORITY, "properties": {"ref": True}} for r in refs]
    pool += [
        {"class_id": None, "geom_4326": m["geom"], "confidence": m["score"], "properties": {"scale_m": m["scale_m"]}}
        for m in masks
    ]
    kept = ModelManager._merge_overlapping_predictions(pool, iou)
    return [
        {"geom": k["geom_4326"], "score": k["confidence"], "scale_m": k["properties"]["scale_m"]}
        for k in kept
        if not k["properties"].get("ref")
    ]


def _write(org_id: uuid.UUID, job_id: uuid.UUID, masks: list[dict[str, Any]], scales_done: list[float]) -> None:
    payload = {
        "scales_done": scales_done,
        "masks": [{"scale_m": m["scale_m"], "geometry": m["geom"], "score": round(float(m["score"]), 4)} for m in masks],
    }
    storage_service.upload_bytes(org_id, masks_key(job_id), json.dumps(payload).encode(), "application/json")


@celery_app.task(bind=True, queue=TRAINING, max_retries=0)
def run_interactive_segment_all(self, job_id: str) -> None:
    with WorkerSession() as session:
        job = session.get(Job, uuid.UUID(job_id))
        if job is None:
            logger.warning("run_interactive_segment_all: job %s not found", job_id)
            return
        req = (job.config or {}).get("request") or {}
        job.status = JobStatus.RUNNING
        job.started_at = _now()
        session.commit()

        def fail(message: str) -> None:
            job.status = JobStatus.FAILED
            job.logs = message[:2000]
            job.finished_at = _now()
            session.commit()

        try:
            item = session.scalar(
                select(DatasetItem).where(
                    DatasetItem.dataset_id == uuid.UUID(req["dataset_id"]),
                    DatasetItem.stac_item_id == req["stac_item_id"],
                    DatasetItem.organization_id == job.organization_id,
                    DatasetItem.is_active.is_(True),
                )
            )
            if item is None or not item.geometry:
                return fail("Imagery item not found.")

            aoi_bbox = [float(v) for v in req["aoi_bbox"]]
            aoi_poly = shape(req["aoi_geometry"]) if req.get("aoi_geometry") else box(*aoi_bbox)
            item_bbox = list(shape(item.geometry).bounds)
            scales = sorted({float(s) for s in req["scales_m"]})
            plan = plan_tiles(item_bbox, aoi_bbox, aoi_poly, scales)
            total = sum(len(v) for v in plan.values())
            if total == 0:
                return fail("The AOI doesn't overlap this imagery.")
            if total > MAX_TILES:
                return fail(f"{total} tiles is over the {MAX_TILES} limit. Use fewer or larger scales.")

            job.total_items = total
            session.commit()

            refs = [{"geom": g} for g in (req.get("reference_geometries") or [])]
            rescale = resolve_rescale_params(item.stac_collection_id, item.stac_item_id).get("rescale")
            lo, hi = SIZE_FRAC_RANGE
            iou = float(req.get("nms_iou", 0.5))
            all_masks: list[dict[str, Any]] = []
            done_scales: list[float] = []
            processed = failed = 0

            for scale in scales:
                for _col, _row, tile in plan[scale]:
                    session.refresh(job)
                    if job.status == JobStatus.CANCELLED:
                        _write(job.organization_id, job.id, _merge(all_masks, refs, iou), done_scales)
                        return
                    mid_lat = (tile[1] + tile[3]) / 2
                    m_lon, m_lat = meters_to_degrees(scale * WINDOW_MARGIN, mid_lat)
                    window = [tile[0] - m_lon, tile[1] - m_lat, tile[2] + m_lon, tile[3] + m_lat]
                    try:
                        w_px, h_px = _crop_dims(window, RENDER_MAX_PX)
                        png = _fetch_crop_png(item, window, w_px, h_px, rescale=rescale)
                        res = _call_geoops_service(
                            "/sam3/segment-tile",
                            {
                                "patch_image_base64": base64.b64encode(png).decode("ascii"),
                                "points_per_side": int(req.get("points_per_side", 16)),
                                "min_score": float(req.get("min_score", 0.5)),
                            },
                            timeout=240.0,
                        )
                        img_w, img_h = res["image_px"]
                        wd, hd = window[2] - window[0], window[3] - window[1]
                        core = box(*tile)
                        for m in res["masks"]:
                            x0, y0, x1, y1 = m["bbox"]
                            if x0 <= 1 or y0 <= 1 or x1 >= img_w - 2 or y1 >= img_h - 2:
                                continue  # cut by the window border
                            ring = lambda r: [(window[0] + x / img_w * wd, window[3] - y / img_h * hd) for x, y in r]  # noqa: E731
                            poly = Polygon(ring(m["rings"][0]), [ring(r) for r in m["rings"][1:]])
                            if not poly.is_valid:
                                poly = poly.buffer(0)
                            if poly.is_empty or poly.geom_type != "Polygon":
                                continue
                            c = poly.centroid
                            if not core.contains(c) or not aoi_poly.contains(c):
                                continue  # a neighbouring tile owns it / outside the AOI
                            frac = bbox_coords_span_m(list(poly.bounds)) / scale
                            if (frac < lo and scale != scales[0]) or (frac > hi and scale != scales[-1]):
                                continue  # a neighbouring scale owns it
                            all_masks.append({"geom": mapping(poly), "score": float(m["score"]), "scale_m": scale})
                    except Exception as exc:  # noqa: BLE001
                        failed += 1
                        logger.warning("segment_all_tile_failed job=%s scale=%s error=%s", job.id, scale, exc)
                    processed += 1
                    job.processed_items, job.failed_items = processed, failed
                    job.progress = processed / total
                    session.commit()

                all_masks = _merge(all_masks, refs, iou)
                done_scales.append(scale)
                _write(job.organization_id, job.id, all_masks, done_scales)
                job.config = {**(job.config or {}), "result": {"scales_done": done_scales, "mask_count": len(all_masks), "masks_key": masks_key(job.id)}}
                session.commit()

            if failed == total:
                return fail("Every tile failed — is the SAM 3 service running?")
            job.status = JobStatus.COMPLETED
            job.logs = f"{len(all_masks)} masks from {total - failed}/{total} tiles."
            job.finished_at = _now()
            session.commit()
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            logger.exception("run_interactive_segment_all_failed job=%s", job_id)
            job2 = session.get(Job, uuid.UUID(job_id))
            if job2 is not None:
                job2.status = JobStatus.FAILED
                job2.logs = str(exc)[:2000]
                job2.finished_at = _now()
                session.commit()
            raise


# ── Mask embeddings (the "Group" stage) ──────────────────────────────────────

EMBED_TIER = "interactive_mask_embeddings"
EMBED_BACKBONE = "dinov2-vits14"
CROP_PX = 256          # rendered crop (mask outline is drawn at this size)
EMBED_PX = 224         # model input (multiple of the 14 px patch)
THUMB_PX = 112
EMBED_BATCH = 64
CROP_CONTEXT = {"mask": 1.15, "box": 1.7}  # window side as a multiple of the object's longer side
MAX_EMBED_MASKS = 6000


def embeddings_prefix(job_id: uuid.UUID | str) -> str:
    return f"interactive/embeddings/{job_id}"


def _crop_for_mask(item: DatasetItem, geometry: dict[str, Any], mode: str, rescale: str | None) -> tuple[str, str]:
    """(embedding image b64 PNG, thumbnail b64 JPEG) for one mask.

    ``mask`` mode embeds only the object (background greyed out): "what does this
    thing look like". ``box`` mode embeds its box with surrounding context: "what
    is it and where is it", e.g. a crop among crops vs a tree beside a road."""
    poly = shape(geometry)
    minx, miny, maxx, maxy = poly.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    w_m = bbox_coords_span_m([minx, cy, maxx, cy])
    h_m = bbox_coords_span_m([cx, miny, cx, maxy])
    side_m = max(1.0, max(w_m, h_m) * CROP_CONTEXT[mode])
    d_lon, d_lat = meters_to_degrees(side_m / 2, cy)
    window = [cx - d_lon, cy - d_lat, cx + d_lon, cy + d_lat]
    png = _fetch_crop_png(item, window, CROP_PX, CROP_PX, rescale=rescale)
    rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    def to_px(ring) -> np.ndarray:
        return np.array(
            [[(x - window[0]) / (2 * d_lon) * CROP_PX, (window[3] - y) / (2 * d_lat) * CROP_PX] for x, y in ring.coords],
            dtype=np.int32,
        )

    if mode == "mask":
        m = np.zeros((CROP_PX, CROP_PX), dtype=np.uint8)
        cv2.fillPoly(m, [to_px(poly.exterior)], 255)
        for hole in poly.interiors:
            cv2.fillPoly(m, [to_px(hole)], 0)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8))
        emb_rgb = np.where(m[..., None] > 0, rgb, 127).astype(np.uint8)
    else:
        emb_rgb = rgb

    def enc(arr: np.ndarray, fmt: str, size: int | None = None, **kw) -> str:
        im = Image.fromarray(arr)
        if size:
            im = im.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format=fmt, **kw)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return enc(emb_rgb, "PNG", EMBED_PX), enc(rgb, "JPEG", THUMB_PX, quality=72)


def render_thumb(item: DatasetItem, geometry: dict[str, Any], rescale: str | None, size: int = 128) -> str:
    """A JPEG (base64) of the imagery around a mask, with the mask outlined — for looking at a
    mask in the review card and the gallery without needing it to have been embedded."""
    poly = shape(geometry)
    minx, miny, maxx, maxy = poly.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    side_m = max(1.0, max(bbox_coords_span_m([minx, cy, maxx, cy]), bbox_coords_span_m([cx, miny, cx, maxy])) * CROP_CONTEXT["box"])
    d_lon, d_lat = meters_to_degrees(side_m / 2, cy)
    window = [cx - d_lon, cy - d_lat, cx + d_lon, cy + d_lat]
    png = _fetch_crop_png(item, window, CROP_PX, CROP_PX, rescale=rescale)
    rgb = np.array(Image.open(io.BytesIO(png)).convert("RGB"))  # writable copy
    ring = np.array(
        [[(x - window[0]) / (2 * d_lon) * CROP_PX, (window[3] - y) / (2 * d_lat) * CROP_PX] for x, y in poly.exterior.coords],
        dtype=np.int32,
    )
    cv2.polylines(rgb, [ring], True, (255, 209, 64), 3, cv2.LINE_AA)  # gold outline: which object this is
    buf = io.BytesIO()
    Image.fromarray(rgb).resize((size, size), Image.LANCZOS).save(buf, format="JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@celery_app.task(bind=True, queue=TRAINING, max_retries=0)
def run_mask_embeddings(self, job_id: str) -> None:
    """Embed every proposed mask with DINOv2 (batched) and keep a thumbnail of each."""
    with WorkerSession() as session:
        job = session.get(Job, uuid.UUID(job_id))
        if job is None:
            return
        req = (job.config or {}).get("request") or {}
        job.status = JobStatus.RUNNING
        job.started_at = _now()
        session.commit()

        def fail(message: str) -> None:
            job.status = JobStatus.FAILED
            job.logs = message[:2000]
            job.finished_at = _now()
            session.commit()

        try:
            item = session.scalar(
                select(DatasetItem).where(
                    DatasetItem.dataset_id == uuid.UUID(req["dataset_id"]),
                    DatasetItem.stac_item_id == req["stac_item_id"],
                    DatasetItem.organization_id == job.organization_id,
                    DatasetItem.is_active.is_(True),
                )
            )
            if item is None:
                return fail("Imagery item not found.")
            masks = req["masks"]
            mode = req.get("mode", "mask")
            job.total_items = len(masks)
            session.commit()
            rescale = resolve_rescale_params(item.stac_collection_id, item.stac_item_id).get("rescale")

            def crop(m: dict[str, Any]) -> tuple[str, str] | None:
                try:
                    return _crop_for_mask(item, m["geometry"], mode, rescale)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("mask_crop_failed job=%s mask=%s error=%s", job_id, m.get("id"), exc)
                    return None

            vectors: list[list[float] | None] = [None] * len(masks)
            thumbs: dict[str, str] = {}
            processed = 0
            with ThreadPoolExecutor(max_workers=8) as pool:
                for start in range(0, len(masks), EMBED_BATCH):
                    session.refresh(job)
                    if job.status == JobStatus.CANCELLED:
                        return
                    chunk = masks[start : start + EMBED_BATCH]
                    crops = list(pool.map(crop, chunk))
                    ok = [(i, c) for i, c in enumerate(crops) if c is not None]
                    if ok:
                        res = _call_geoops_service(
                            "/embed/batch",
                            {"backbone_key": EMBED_BACKBONE, "images_base64": [c[0] for _, c in ok], "size": EMBED_PX},
                            timeout=300.0,
                        )
                        for (i, c), vec in zip(ok, res["embeddings"], strict=True):
                            vectors[start + i] = vec
                            thumbs[chunk[i]["id"]] = c[1]
                    processed += len(chunk)
                    job.processed_items, job.progress = processed, processed / len(masks)
                    session.commit()

            good = [i for i, v in enumerate(vectors) if v is not None]
            if not good:
                return fail("No mask could be embedded — is the imagery readable and the model service running?")
            ids = [masks[i]["id"] for i in good]
            arr = np.asarray([vectors[i] for i in good], dtype=np.float32)
            buf = io.BytesIO()
            np.save(buf, arr)
            prefix = embeddings_prefix(job.id)
            storage_service.upload_bytes(job.organization_id, f"{prefix}/embeddings.npy", buf.getvalue(), "application/octet-stream")
            storage_service.upload_bytes(job.organization_id, f"{prefix}/ids.json", json.dumps(ids).encode(), "application/json")
            storage_service.upload_bytes(job.organization_id, f"{prefix}/thumbs.json", json.dumps(thumbs).encode(), "application/json")
            job.config = {**(job.config or {}), "result": {"n": len(ids), "dim": int(arr.shape[1]), "mode": mode, "skipped": len(masks) - len(ids)}}
            job.status = JobStatus.COMPLETED
            job.logs = f"Embedded {len(ids)} of {len(masks)} masks ({mode} crops)."
            job.finished_at = _now()
            session.commit()
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            logger.exception("run_mask_embeddings_failed job=%s", job_id)
            job2 = session.get(Job, uuid.UUID(job_id))
            if job2 is not None:
                job2.status = JobStatus.FAILED
                job2.logs = str(exc)[:2000]
                job2.finished_at = _now()
                session.commit()
            raise
