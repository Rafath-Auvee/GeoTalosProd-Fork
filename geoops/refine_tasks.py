"""Celery task: turn an annotation set of boxes (e.g. a detector's output) into masks with SAM 3.

Each box is a prompt; SAM returns the object's mask. Boxes are grouped into tiles of the
platform's own scale grid so one image embedding serves many boxes. The result is a NEW
annotation set laid out like the interactive ones: the geometry is the mask polygon and
``properties.bbox`` its bounding box. The source set is left as it was, and the new set stays
unreviewed ("raw"): SAM sharpens outlines but does not decide whether a detection was real.

Sync (WorkerSession / psycopg2), like the other geoops tasks.
"""
from __future__ import annotations

import base64
import logging
import uuid
from collections import defaultdict
from statistics import median
from typing import Any

from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Polygon, box, shape
from sqlalchemy import select

from app.core.enums import JobStatus
from app.models.annotation import Annotation
from app.models.annotation_set import AnnotationSet
from app.models.dataset_item import DatasetItem
from app.models.job import Job
from app.models.map import Map
from app.models.map_annotation_set import MapAnnotationSet
from app.models.map_aoi import MapAOI
from app.models.project_annotation_set import ProjectAnnotationSet
from app.workers.celery_app import celery_app
from app.workers.db import WorkerSession
from app.workers.queues import TRAINING

from geoops.dino_tasks import _call_geoops_service, _now
from geoops.export_common import resolve_rescale_params
from geoops.scale import canonical_tiles, meters_to_degrees
from geoops.yolo_tasks import _crop_dims, _fetch_crop_png

logger = logging.getLogger(__name__)

RENDER_MAX_PX = 1024
WINDOW_MARGIN = 0.25      # of the tile edge, each side — room for boxes that straddle the tile border
MIN_MASK_SCORE = 0.4      # SAM's own confidence in the mask
AREA_RATIO = (0.2, 1.1)   # mask area / box area: far outside this, SAM latched onto something else
MIN_BOX_IOU = 0.5         # the mask's bounds must still agree with the box that prompted it


def _px_ring_to_geo(ring: list[list[float]], window: list[float], img_w: int, img_h: int) -> list[tuple[float, float]]:
    wd, hd = window[2] - window[0], window[3] - window[1]
    return [(window[0] + x / img_w * wd, window[3] - y / img_h * hd) for x, y in ring]


@celery_app.task(bind=True, queue=TRAINING, max_retries=0)
def run_refine_boxes_to_masks(self, job_id: str) -> None:
    with WorkerSession() as session:
        job = session.get(Job, uuid.UUID(job_id))
        if job is None:
            logger.warning("run_refine_boxes_to_masks: job %s not found", job_id)
            return
        req: dict[str, Any] = (job.config or {}).get("request") or {}
        job.status = JobStatus.RUNNING
        job.started_at = _now()
        session.commit()

        def fail(message: str, extra: dict | None = None) -> None:
            job.status = JobStatus.FAILED
            job.logs = message[:2000]
            job.config = {**(job.config or {}), "result": extra or {}}
            job.finished_at = _now()
            session.commit()

        try:
            src = session.get(AnnotationSet, uuid.UUID(req["annotation_set_id"]))
            if src is None or src.organization_id != job.organization_id or src.deleted_at is not None:
                return fail("The annotation set no longer exists.")
            min_score = float(req.get("min_score", 0.5))
            scale = float(req.get("scale_m", 10.0))

            rows = session.scalars(
                select(Annotation).where(Annotation.annotation_set_id == src.id, Annotation.deleted_at.is_(None))
            ).all()
            total_boxes = len(rows)
            todo = [r for r in rows if r.class_id is not None and (r.confidence is None or r.confidence >= min_score)]
            below = total_boxes - len(todo)
            if not todo:
                return fail("No annotations to refine at this score.", {"boxes_total": total_boxes, "below_score": below})

            item_id = src.dataset_item_id or todo[0].dataset_item_id
            item = session.get(DatasetItem, item_id) if item_id else None
            if item is None or not item.geometry or not item.is_active:
                return fail("The imagery for this set is missing.")
            item_bbox = list(shape(item.geometry).bounds)
            rescale = resolve_rescale_params(item.stac_collection_id, item.stac_item_id).get("rescale")

            geoms = {r.id: to_shape(r.geometry) for r in todo}
            xs0, ys0, xs1, ys1 = zip(*(g.bounds for g in geoms.values()), strict=True)
            extent = [min(xs0), min(ys0), max(xs1), max(ys1)]
            grid = canonical_tiles(item_bbox, 0, clip_bbox=extent, tiers=[scale])
            by_tile: dict[tuple[int, int], list[Annotation]] = defaultdict(list)
            tile_boxes = {(c, r): b for c, r, b in grid}
            for r in todo:
                cen = geoms[r.id].centroid
                key = next((k for k, b in tile_boxes.items() if box(*b).contains(cen)), None)
                if key is not None:
                    by_tile[key].append(r)

            job.total_items = len(by_tile)
            session.commit()

            aois: list[tuple[MapAOI, Any]] = []
            map_row = session.get(Map, uuid.UUID(req["map_id"])) if req.get("map_id") else None
            if map_row is not None:
                for a in session.scalars(select(MapAOI).where(MapAOI.map_id == map_row.id, MapAOI.deleted_at.is_(None))).all():
                    aois.append((a, shape(a.geometry) if a.geometry else box(*a.bbox_4326)))

            kept: list[dict[str, Any]] = []
            dropped = {"no_mask": 0, "low_mask_score": 0, "odd_area": 0, "off_box": 0}
            processed = failed = 0
            for key, anns in by_tile.items():
                tile = tile_boxes[key]
                mid_lat = (tile[1] + tile[3]) / 2
                m_lon, m_lat = meters_to_degrees(scale * WINDOW_MARGIN, mid_lat)
                window = [tile[0] - m_lon, tile[1] - m_lat, tile[2] + m_lon, tile[3] + m_lat]
                for r in anns:  # a box poking out of the margin widens the window instead of being cut
                    b = geoms[r.id].bounds
                    window = [min(window[0], b[0]), min(window[1], b[1]), max(window[2], b[2]), max(window[3], b[3])]
                try:
                    w_px, h_px = _crop_dims(window, RENDER_MAX_PX)
                    png = _fetch_crop_png(item, window, w_px, h_px, rescale=rescale)
                    wd, hd = window[2] - window[0], window[3] - window[1]
                    px_boxes = []
                    for r in anns:
                        x0, y0, x1, y1 = geoms[r.id].bounds
                        px_boxes.append([(x0 - window[0]) / wd * w_px, (window[3] - y1) / hd * h_px,
                                         (x1 - window[0]) / wd * w_px, (window[3] - y0) / hd * h_px])
                    res = _call_geoops_service(
                        "/sam3/boxes-masks",
                        {"patch_image_base64": base64.b64encode(png).decode("ascii"), "boxes": px_boxes},
                        timeout=300.0,
                    )
                    img_w, img_h = res["image_px"]
                    for r, out in zip(anns, res["results"], strict=True):
                        if not out or not out.get("rings"):
                            dropped["no_mask"] += 1
                            continue
                        if out.get("score") is not None and out["score"] < MIN_MASK_SCORE:
                            dropped["low_mask_score"] += 1
                            continue
                        poly = Polygon(
                            _px_ring_to_geo(out["rings"][0], window, img_w, img_h),
                            [_px_ring_to_geo(h, window, img_w, img_h) for h in out["rings"][1:]],
                        )
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        if poly.is_empty or poly.geom_type != "Polygon":
                            dropped["no_mask"] += 1
                            continue
                        src_box = geoms[r.id]
                        ratio = poly.area / src_box.area if src_box.area else 0
                        if not (AREA_RATIO[0] <= ratio <= AREA_RATIO[1]):
                            dropped["odd_area"] += 1
                            continue
                        mb = box(*poly.bounds)
                        if mb.intersection(src_box).area / (mb.union(src_box).area or 1) < MIN_BOX_IOU:
                            dropped["off_box"] += 1
                            continue
                        kept.append({"ann": r, "poly": poly, "mask_score": out.get("score")})
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    logger.warning("refine_tile_failed job=%s tile=%s error=%s", job.id, key, exc)
                processed += 1
                job.processed_items, job.failed_items = processed, failed
                job.progress = processed / len(by_tile)
                session.commit()

            if failed == len(by_tile):
                return fail("Every tile failed. Is the SAM 3 service running?")
            if not kept:
                return fail("No usable masks came out of SAM.", {"boxes_total": total_boxes, "dropped": dropped})

            new_set = AnnotationSet(
                organization_id=job.organization_id, schema_id=src.schema_id, dataset_id=src.dataset_id, dataset_item_id=item.id,
                source_type="model", model_id=src.model_id, job_id=job.id,
                name=(req.get("name") or f"{src.name} · SAM masks")[:255],
                description=f"Masks for “{src.name}”: SAM 3 prompted with each box scoring {min_score:g} or more.",
                review_status="raw", created_by_user_id=job.created_by_user_id,
            )
            session.add(new_set)
            session.flush()
            for k in kept:
                r, poly = k["ann"], k["poly"]
                props: dict[str, Any] = {
                    "bbox": [round(v, 9) for v in poly.bounds], "source": "sam3_refined", "scale_m": scale,
                    "detection_score": r.confidence, "mask_score": k["mask_score"],
                    "refined_from_set_id": str(src.id), "refined_from_annotation_id": str(r.id), "review_status": "raw",
                }
                cen = poly.centroid
                for aoi, aoi_poly in aois:
                    if aoi_poly.contains(cen):
                        props.update(aoi_id=str(aoi.id), aoi_name=aoi.name)
                        break
                session.add(Annotation(
                    annotation_set_id=new_set.id, class_id=r.class_id, dataset_item_id=item.id, confidence=r.confidence,
                    geometry=from_shape(poly, srid=4326), properties=props, created_by_user_id=job.created_by_user_id,
                ))
            if map_row is not None:
                session.add(ProjectAnnotationSet(project_id=map_row.project_id, annotation_set_id=new_set.id, linked_by=job.created_by_user_id))
                session.add(MapAnnotationSet(map_id=map_row.id, annotation_set_id=new_set.id, visible=True, opacity=1.0, z_index=0))
            from app.services.annotation_set_grouping import ensure_schema_collection_sync  # noqa: PLC0415

            ensure_schema_collection_sync(session, new_set)

            scores = [k["mask_score"] for k in kept if k["mask_score"] is not None]
            job.config = {
                **(job.config or {}),
                "result": {
                    "annotation_set_id": str(new_set.id), "name": new_set.name, "masks": len(kept), "boxes_total": total_boxes,
                    "below_score": below, "dropped": dropped, "failed_tiles": failed,
                    "median_mask_score": round(median(scores), 3) if scores else None,
                },
            }
            job.status = JobStatus.COMPLETED
            job.logs = f"{len(kept)} masks from {len(todo)} boxes ({sum(dropped.values())} dropped, {below} below the score)."
            job.finished_at = _now()
            session.commit()
        except Exception as exc:  # noqa: BLE001
            session.rollback()
            logger.exception("run_refine_boxes_to_masks_failed job=%s", job_id)
            job2 = session.get(Job, uuid.UUID(job_id))
            if job2 is not None:
                job2.status = JobStatus.FAILED
                job2.logs = str(exc)[:2000]
                job2.finished_at = _now()
                session.commit()
