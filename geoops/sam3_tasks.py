"""Celery task: direct SAM3 box+text auto-annotate over an AOI ("similar to
Train Head" for the reference set, but scanning an explicit AOI and calling
geoops-service's /autoannotate directly instead of the embedding-similarity
pipeline in geoops.exemplar_tasks).

Takes the same "verified collection + class" input Train Head
(``geoops.dino_tasks``) does — see ``_gather_rows_for_class`` — to build a
reference set, then tiles an explicit ``dataset_item_ids`` AOI SAHI-style
(overlapping tiles, reusing ``geoops.exemplar_tasks.overlapping_tiles``) at a
working resolution auto-inferred from the references' own real-world size
(``geoops.scale.working_resolution_from_refs``).

SAM3's box-prompt API is single-image only (see geoops-service's
``app/sam3_infer.py`` docstring: a box is normalized against the same
image's own shape) — a reference's bbox can only be used as a box exemplar
on a tile cropped from *that reference's own dataset item*, and only when
the reference geometrically falls inside that tile. Every other AOI tile
(including every tile of every *other* dataset item) is scanned with SAM3's
text-only prompt instead ("crop", hardcoded on the geoops-service side for
now) — text prompting needs no image-local coordinates, so unlike the box
exemplar it works across the whole AOI regardless of source image. This is
what actually makes this a whole-AOI scan rather than "re-detect near each
reference's own tile only" (the module's first cut).

Duplicate detections from overlapping tiles are merged via NMS
(``supervision``), same as ``geoops.exemplar_tasks._run_similarity_scan_core``.

Sync (WorkerSession/psycopg2) + urllib HTTP to geoops-service, same pattern
as ``geoops.dino_tasks._call_geoops_service`` and ``geoops.tasks``'
``_fetch_patch_png_b64``/``_persist_results_annotation_set`` — reused
directly rather than duplicated.
"""
from __future__ import annotations

import json
import logging
import uuid
from urllib import request as urlrequest

import numpy as np
import supervision as sv
from shapely.geometry import box, shape
from sqlalchemy import select

from app.config import settings
from app.core.enums import JobStatus
from app.models.dataset_item import DatasetItem
from app.models.job import Job
from app.workers.celery_app import celery_app
from app.workers.db import WorkerSession
from app.workers.queues import TRAINING

from geoops.exemplar_tasks import MAX_OVERLAP_RATIO, MIN_OVERLAP_RATIO, overlapping_tiles
from geoops.scale import meters_to_degrees, working_resolution_from_refs
from geoops.tasks import _fetch_patch_png_b64, _intersect, _now, _persist_results_annotation_set

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
MAX_ERROR_SAMPLE = 50
# Hardcoded per this feature's current scope (see module + schema docstrings)
# — a later iteration can expose it once there's more than one class to try.
SAM3_TEXT_PROMPT = "crop"


def _call_sam3_autoannotate(patch_b64: str, bboxes: list[list[float]], conf: float) -> list[dict]:
    url = f"{settings.GEOOPS_SERVICE_URL.rstrip('/')}/autoannotate"
    body = {
        "patch_image_base64": patch_b64,
        "bboxes": bboxes,
        "text": SAM3_TEXT_PROMPT,
        "conf": conf,
    }
    req = urlrequest.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.GEOOPS_SERVICE_API_TOKEN}",
        },
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=120.0) as resp:  # nosec B310
        data = json.loads(resp.read().decode("utf-8"))
    boxes = data.get("boxes")
    if not isinstance(boxes, list):
        raise ValueError("geoops-service /autoannotate response missing a 'boxes' list")
    return boxes


def _geo_bbox_to_pixel(geo_bbox: list[float], crop_bbox: list[float], crop_size_px: int) -> list[float]:
    """Project an EPSG:4326 bbox onto the pixel grid of a ``crop_size_px``
    square crop of ``crop_bbox`` — the inverse of ``_pixel_bbox_to_geo``.
    Y is flipped (image row 0 = north/top, same as the PNG TiTiler returns).
    """
    minx, miny, maxx, maxy = geo_bbox
    cminx, cminy, cmaxx, cmaxy = crop_bbox
    w, h = (cmaxx - cminx) or 1e-12, (cmaxy - cminy) or 1e-12
    x0 = (minx - cminx) / w * crop_size_px
    x1 = (maxx - cminx) / w * crop_size_px
    y0 = (cmaxy - maxy) / h * crop_size_px
    y1 = (cmaxy - miny) / h * crop_size_px
    return [x0, y0, x1, y1]


def _pixel_bbox_to_geo(px_bbox: list[float], crop_bbox: list[float], crop_size_px: int) -> list[float]:
    """Inverse of ``_geo_bbox_to_pixel`` — maps a SAM3 output box back onto
    EPSG:4326 within the same crop it was detected in.
    """
    x0, y0, x1, y1 = px_bbox
    cminx, cminy, cmaxx, cmaxy = crop_bbox
    w, h = cmaxx - cminx, cmaxy - cminy
    minx = cminx + (x0 / crop_size_px) * w
    maxx = cminx + (x1 / crop_size_px) * w
    maxy = cmaxy - (y0 / crop_size_px) * h
    miny = cmaxy - (y1 / crop_size_px) * h
    return [minx, miny, maxx, maxy]


def _bbox_contains(outer: list[float], inner: list[float]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


@celery_app.task(bind=True, queue=TRAINING, max_retries=2, default_retry_delay=60)
def run_sam3_auto_annotate(self, job_id: str) -> None:
    """Reads ``Sam3AutoAnnotateRequest`` (see geoops.schemas) from
    ``job.config["request"]``. Results land in ``job.config["result"]`` and
    as a new reviewable ``AnnotationSet`` (bounding boxes only).
    """
    from geoops.dino_tasks import _gather_rows_for_class  # noqa: PLC0415 — avoids a hard import cycle at module load

    job_uuid = uuid.UUID(job_id)

    with WorkerSession() as session:
        job = session.get(Job, job_uuid)
        if job is None:
            logger.error("run_sam3_auto_annotate: job %s not found", job_id)
            return

        cfg = dict(job.config or {})
        req = cfg.get("request") or {}
        try:
            collection_id = uuid.UUID(req["annotation_set_collection_id"])
            class_id = uuid.UUID(req["class_id"])
            max_references = int(req.get("max_references", 5))
            dataset_item_ids = [uuid.UUID(i) for i in req["dataset_item_ids"]]
            aoi_bbox = req.get("aoi_bbox")
            crop_size_px = int(req.get("crop_size_px", 256))
            requested_overlap_ratio = req.get("overlap_ratio")
            conf = float(req.get("conf", 0.3))
            iou_thresh = float(req.get("iou_thresh", 0.3))
            top_k = int(req.get("top_k", 50))
            max_patches = int(req.get("max_patches", 200))
            output_class_id = uuid.UUID(req["output_class_id"]) if req.get("output_class_id") else class_id
        except (KeyError, ValueError, TypeError) as exc:
            job.status = JobStatus.FAILED
            job.logs = f"Invalid job config: {exc}"
            job.finished_at = _now()
            session.commit()
            return

        job.status = JobStatus.RUNNING
        job.started_at = _now()
        session.commit()

        try:
            rows, _skipped, _collection, _class_name = _gather_rows_for_class(
                session, job.organization_id, collection_id, class_id
            )
            if not rows:
                raise ValueError(
                    "No eligible (verified, resolvable-item) annotations of this class found in this collection."
                )
            references = rows[:max_references]
            ref_geometries = [r["geometry"] for r in references]

            # ── Auto-infer working resolution + tile overlap from the references ──
            meters_per_pixel, object_px = working_resolution_from_refs(ref_geometries)
            overlap_ratio = requested_overlap_ratio
            if overlap_ratio is None:
                overlap_ratio = max(MIN_OVERLAP_RATIO, min(MAX_OVERLAP_RATIO, object_px / crop_size_px))

            # (dataset_item_id, geo_bbox) for every reference — used below to decide
            # which AOI tiles get a box exemplar instead of text-only.
            ref_bboxes_by_item: dict[uuid.UUID, list[list[float]]] = {}
            for r in references:
                ref_bboxes_by_item.setdefault(r["item"].id, []).append(list(shape(r["geometry"]).bounds))

            items = session.execute(
                select(DatasetItem).where(
                    DatasetItem.id.in_(dataset_item_ids),
                    DatasetItem.organization_id == job.organization_id,
                    DatasetItem.is_active.is_(True),
                )
            ).scalars().all()
            if len(items) != len(set(dataset_item_ids)):
                raise ValueError("One or more dataset items not found")

            # ── Build the full overlapping AOI tile list up front ──
            all_tiles: list[tuple[DatasetItem, list[float]]] = []
            for item in items:
                if not item.geometry:
                    continue
                item_bbox = list(shape(item.geometry).bounds)
                mid_lat = (item_bbox[1] + item_bbox[3]) / 2.0
                tile_span_m = crop_size_px * meters_per_pixel
                tile_w_deg, tile_h_deg = meters_to_degrees(tile_span_m, mid_lat)
                clip_bbox = _intersect(item_bbox, aoi_bbox) if aoi_bbox else item_bbox
                if clip_bbox is None:
                    continue
                for _col, _row, tile_bbox in overlapping_tiles(
                    item_bbox, clip_bbox, tile_w_deg, tile_h_deg, overlap_ratio
                ):
                    all_tiles.append((item, tile_bbox))

            patches_truncated = False
            if len(all_tiles) > max_patches:
                all_tiles = all_tiles[:max_patches]
                patches_truncated = True

            job.total_items = len(all_tiles)
            job.processed_items = 0
            job.failed_items = 0
            job.progress = 1.0 if not all_tiles else 0.0
            session.commit()

            errors: list[dict] = []
            error_total = 0
            candidates: list[dict] = []

            for idx, (item, tile_bbox) in enumerate(all_tiles):
                try:
                    patch_b64 = _fetch_patch_png_b64(item, tile_bbox, crop_size_px)

                    bboxes_px: list[list[float]] = []
                    used_box_exemplar = False
                    for ref_geo_bbox in ref_bboxes_by_item.get(item.id, []):
                        if _bbox_contains(tile_bbox, ref_geo_bbox):
                            bboxes_px = [_geo_bbox_to_pixel(ref_geo_bbox, tile_bbox, crop_size_px)]
                            used_box_exemplar = True
                            break

                    boxes = _call_sam3_autoannotate(patch_b64, bboxes_px, conf)
                    for b in boxes:
                        geo_bbox = _pixel_bbox_to_geo(b["bbox"], tile_bbox, crop_size_px)
                        candidates.append(
                            {
                                "dataset_item_id": item.id,
                                "bbox": geo_bbox,
                                "score": b["score"],
                                "used_box_exemplar": used_box_exemplar,
                            }
                        )
                except Exception as exc:  # noqa: BLE001 — one bad tile shouldn't fail the whole scan
                    error_total += 1
                    if len(errors) < MAX_ERROR_SAMPLE:
                        errors.append({"dataset_item_id": str(item.id), "bbox": tile_bbox, "error": str(exc)})

                if (idx + 1) % BATCH_SIZE == 0 or idx == len(all_tiles) - 1:
                    job.processed_items = idx + 1
                    job.progress = (idx + 1) / len(all_tiles)
                    session.commit()

            # ── Merge duplicates across overlapping tiles (supervision NMS) ──
            candidates_before_nms = len(candidates)
            matches = candidates
            if matches:
                xyxy = np.array([m["bbox"] for m in matches], dtype=np.float32)
                confidence = np.array([m["score"] for m in matches], dtype=np.float32)
                detections = sv.Detections(
                    xyxy=xyxy, confidence=confidence, class_id=np.arange(len(matches), dtype=int)
                )
                keep = detections.with_nms(threshold=iou_thresh, class_agnostic=True)
                kept_indices = set(keep.class_id.tolist())
                matches = [m for i, m in enumerate(matches) if i in kept_indices]

            matches.sort(key=lambda m: m["score"], reverse=True)
            top_matches = matches[:top_k]

            # ── Persist top_k matches as a new annotation_set/layer ──
            result_dataset_id = items[0].dataset_id if items else None
            annotation_set_id = _persist_results_annotation_set(
                session, job, output_class_id, None, result_dataset_id,
                name=f"SAM3 Auto Annotate — {len(top_matches)} matches",
                rows=[
                    {
                        "geometry": box(*m["bbox"]),
                        "confidence": max(0.0, min(1.0, m["score"])),
                        "properties": {
                            "dataset_item_id": str(m["dataset_item_id"]),
                            "score": m["score"],
                            "used_box_exemplar": m["used_box_exemplar"],
                        },
                    }
                    for m in top_matches
                ],
            )

            cfg["result"] = {
                "references_used": len(references),
                "meters_per_pixel": meters_per_pixel,
                "object_px": object_px,
                "overlap_ratio": overlap_ratio,
                "patches_scanned": len(all_tiles),
                "patches_truncated": patches_truncated,
                "candidates_before_nms": candidates_before_nms,
                "errors_total": error_total,
                "errors_sample": errors,
                "matches": [
                    {
                        "dataset_item_id": str(m["dataset_item_id"]),
                        "bbox": m["bbox"],
                        "score": m["score"],
                        "used_box_exemplar": m["used_box_exemplar"],
                    }
                    for m in top_matches
                ],
                "annotation_set_id": str(annotation_set_id) if annotation_set_id else None,
            }
            job.config = cfg
            job.processed_items = len(all_tiles)
            job.failed_items = error_total
            job.progress = 1.0
            job.status = JobStatus.COMPLETED
            job.finished_at = _now()
            session.commit()

            logger.info(
                "run_sam3_auto_annotate done job=%s tiles=%d errors=%d matches=%d",
                job.id, len(all_tiles), error_total, len(top_matches),
            )

        except Exception as exc:
            session.rollback()
            logger.exception("run_sam3_auto_annotate failed job=%s", job_id)
            try:
                job = session.get(Job, job_uuid)
                if job is not None:
                    job.status = JobStatus.FAILED
                    job.logs = str(exc)[:4000]
                    job.finished_at = _now()
                    session.commit()
            except Exception:
                logger.exception("could not mark job %s as failed", job_id)
            raise


__all__ = ["run_sam3_auto_annotate"]
