"""Celery task: build (or rebuild) one DinoDataset — tile an exhaustively-
labeled AOI, split it into train/eval/test by spatial region, and write the
tile images + labels + manifest.json to object storage.

This is the *only* place tiles are generated. Training
(geoops.dino_tasks.run_dino_train) reads the finished dataset and never
rebuilds it, so retraining with other hyperparameters is cheap and the data
stays reviewable between runs.

Storage layout (``build_no`` makes a rebuild write a fresh folder, so a
training run already reading the previous build is never disturbed):

    dino-datasets/{dataset_id}/b{build_no}/manifest.json
    dino-datasets/{dataset_id}/b{build_no}/{split}/images/{tile}.png
    dino-datasets/{dataset_id}/b{build_no}/{split}/targets/{tile}.json   (detect)

A classify dataset cloned from a detect one stores only a manifest whose
``image`` keys point into the source dataset's folder.

Sync (WorkerSession / psycopg2), same split as the other geoops tasks.
"""
from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

from shapely.geometry import box as shapely_box, shape

from app.core.enums import JobStatus
from app.models.job import Job
from app.models.map_aoi import MapAOI
from app.services import storage_service
from app.workers.celery_app import celery_app
from app.workers.db import WorkerSession
from app.workers.queues import TRAINING

from geoops.dino_dataset import (
    BuildParams,
    assign_splits,
    build_aoi_tiles,
    compute_data_fingerprint,
    seed_from,
    split_stats,
)
from geoops.dino_tasks import _filter_rows_by_aoi, _gather_rows_for_classes, explicit_set_ids
from geoops.export_common import resolve_rescale_params
from geoops.models import DinoDataset
from geoops.scale import SIZE_FRAC_RANGE, bbox_coords_span_m, home_scale, suggest_scales
from geoops.yolo_service import geometry_to_yolo_bbox, geometry_to_yolo_polygon
from geoops.yolo_tasks import _crop_dims, _fetch_crop_png

logger = logging.getLogger(__name__)

FETCH_WORKERS = 4
MAX_ERRORS_KEPT = 50


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _is_mask(geometry: dict) -> bool:
    """More than a box's four corners (an axis-aligned rectangle has five coordinates, closing point included)."""
    g = shape(geometry)
    if g.geom_type == "MultiPolygon":
        return True
    return g.geom_type == "Polygon" and len(g.exterior.coords) > 6


def _row_span_m(row: dict) -> float:
    return bbox_coords_span_m(list(shape(row["geometry"]).bounds))


def _params_from(raw: dict, rows: list[dict] | None = None) -> BuildParams:
    scales = raw.get("scales_m") or ([raw["tile_scale_m"]] if raw.get("tile_scale_m") else None)
    if not scales:
        scales = suggest_scales([_row_span_m(r) for r in (rows or [])])["scales_m"]
    return BuildParams(
        scales_m=tuple(sorted(float(v) for v in scales)),
        crop_px=int(raw.get("crop_px", 1280)),
        eval_split=float(raw.get("eval_split", 0.15)),
        test_split=float(raw.get("test_split", 0.15)),
        neg_ratio=float(raw.get("neg_ratio", 2.0)),
        region_tiles=int(raw.get("region_tiles", 3)),
    )


def _fail(session, job: Job, ds: DinoDataset, message: str, result: dict | None = None) -> None:
    ds.status = "failed"
    ds.error = message[:2000]
    job.status = JobStatus.FAILED
    job.logs = message[:2000]
    job.config = {**(job.config or {}), "result": result or {}}
    job.finished_at = _now()
    session.commit()


def _progress(session, job: Job, processed: int, failed: int) -> None:
    job.processed_items = processed
    job.failed_items = failed
    job.progress = processed / job.total_items if job.total_items else 1.0
    session.commit()


def _build_from_annotations(session, job: Job, ds: DinoDataset, build_no: int) -> bool:
    org_id = job.organization_id
    # Original behaviour: one AOI, the collection's verified sets, annotations stamped with the AOI.
    # With explicit annotation sets: any number of AOIs, annotations matched to them by position.
    chosen = [uuid.UUID(i) for i in (ds.annotation_set_ids or [])]
    class_ids = [uuid.UUID(c) for c in (ds.class_ids or [str(ds.class_id)])]
    rows, skipped, collection, class_names = _gather_rows_for_classes(
        session, org_id, ds.annotation_set_collection_id, class_ids, chosen or None
    )
    class_name = " + ".join(class_names)
    class_index = {str(c): i for i, c in enumerate(class_ids)}  # the order chosen is the YOLO class index
    for r in rows:
        r["class_index"] = class_index[r["class_id"]]
    boxes_skipped = 0
    if ds.task == "segment":
        # A segmenter learns from masks. A box (four corners) says nothing about the outline.
        masks_only = [r for r in rows if _is_mask(r["geometry"])]
        boxes_skipped, rows = len(rows) - len(masks_only), masks_only
        if not rows:
            _fail(session, job, ds, "None of the chosen annotations is a mask: they are boxes. Choose sets that have masks.")
            return False
    aois: list[MapAOI] = []
    for aid in (ds.aoi_ids or [str(ds.aoi_id)]):
        a = session.get(MapAOI, uuid.UUID(aid))
        if a is None or a.organization_id != org_id or a.deleted_at is not None:
            _fail(session, job, ds, "An AOI this dataset was built from no longer exists.")
            return False
        aois.append(a)
    aoi = aois[0]
    if chosen:
        polys = [shape(a.geometry) if a.geometry else shapely_box(*a.bbox_4326) for a in aois]
        inside: list[dict] = []
        for r in rows:
            g = shape(r["geometry"])
            for a, poly in zip(aois, polys, strict=True):
                if g.intersects(poly):
                    r["aoi_id"] = str(a.id)
                    inside.append(r)
                    break
        skipped_other_aoi = len(rows) - len(inside)
        rows = inside
    else:
        rows, skipped_other_aoi = _filter_rows_by_aoi(rows, str(ds.aoi_id))
    if not rows:
        _fail(
            session, job, ds,
            "No eligible annotations of this class in the chosen AOIs (need an active image "
            "and annotations drawn inside the AOI)." if chosen else
            "No eligible annotations of this class tagged to this AOI (need verified sets, "
            "an active image, and annotations drawn inside the AOI).",
            {"sets_skipped": skipped},
        )
        return False

    params = _params_from(ds.params, rows)
    roles = (ds.params or {}).get("set_roles") or {}
    for r in rows:
        r["role"] = roles.get(r.get("set_id"), "both")
    spans = [_row_span_m(r) for r in rows]
    sizing = suggest_scales(spans)
    # Every object's home scale: where it is closest to ~30% of the tile edge.
    for r, span in zip(rows, spans, strict=True):
        r["span_m"] = span
        r["home_scale"] = home_scale(span, params.scales_m)
    lo_frac, hi_frac = SIZE_FRAC_RANGE
    # Objects the ladder can't fit well (beyond its coarsest / finest scale).
    out_of_range = sum(
        1 for r in rows
        if not (lo_frac <= r["span_m"] / r["home_scale"] <= hi_frac)
    )

    fingerprint = compute_data_fingerprint(
        annotation_versions=[
            (str(r["annotation_id"]), r["updated_at"].astimezone(UTC).isoformat()) for r in rows
        ],
        set_ids=[str(sid) for sid in (explicit_set_ids(session, org_id, collection, chosen) if chosen else _included_set_ids(session, collection))],
        aoi_bbox=[list(a.bbox_4326) for a in aois] if chosen else list(aoi.bbox_4326),
        aoi_geometry=[a.geometry for a in aois] if chosen else aoi.geometry,  # type: ignore[arg-type]
        task=ds.task,
    )

    by_item: dict[uuid.UUID, list[dict]] = {}
    for r in rows:
        by_item.setdefault(r["item"].id, []).append(r)

    tiles_by_key: dict[str, dict] = {}
    ambiguous_dropped = 0
    for item_rows in by_item.values():
        item = item_rows[0]["item"]
        if not item.geometry:
            continue
        item_bbox = list(shape(item.geometry).bounds)
        for a in aois:
            a_rows = [r for r in item_rows if r.get("aoi_id") == str(a.id)] if chosen else item_rows
            if not a_rows:
                continue
            for scale in params.scales_m:
                got, amb = build_aoi_tiles(
                    item_id=str(item.id),
                    item_bbox=item_bbox,
                    aoi_bbox=list(a.bbox_4326),
                    aoi_geometry=a.geometry,
                    rows=[r for r in a_rows if r["home_scale"] == scale],
                    all_rows=a_rows,
                    tile_scale_m=scale,
                )
                ambiguous_dropped += amb
                # Empty tiles only count as background where the labels are confirmed complete: an AOI whose
                # annotations are all train-only (e.g. a detector's output) says nothing about what is empty.
                if not any(r["role"] != "train" for r in a_rows):
                    got = [t for t in got if t["rows"]]
                for t in got:
                    t["role"] = "train" if any(r["role"] == "train" for r in t["rows"]) else (
                        "test" if any(r["role"] == "test" for r in t["rows"]) else "both"
                    )
                    # Neighbouring AOIs can share an edge tile of the item's grid: keep one, with both AOIs' rows.
                    prev = tiles_by_key.get(t["tile_key"])
                    if prev is None:
                        tiles_by_key[t["tile_key"]] = {**t, "item": item}
                    else:
                        prev["rows"] = prev["rows"] + t["rows"]
                        prev["role"] = "train" if "train" in (prev["role"], t["role"]) else (
                            "test" if "test" in (prev["role"], t["role"]) else "both"
                        )
    tiles = list(tiles_by_key.values())

    kept = assign_splits(tiles, params, seed_from(fingerprint))
    stats = split_stats(kept, params)
    if stats["splits"]["train"]["positives"] == 0:
        _fail(
            session, job, ds,
            "The train split would contain no positive tiles — label more instances in this AOI "
            "or lower the eval/test split.",
            {"stats": stats},
        )
        return False

    prefix = f"dino-datasets/{ds.id}/b{build_no}"
    job.total_items = len(kept)
    session.commit()

    def process(tile: dict) -> dict:
        item = tile["item"]
        width_px, height_px = _crop_dims(tile["crop_bbox"], params.crop_px)
        rescale = resolve_rescale_params(item.stac_collection_id, item.stac_item_id)
        png = _fetch_crop_png(item, tile["crop_bbox"], width_px, height_px, rescale=rescale.get("rescale"))
        file_key = tile["tile_key"].replace(":", "_")
        base = f"{prefix}/{tile['split']}"
        image_key = f"{base}/images/{file_key}.png"
        storage_service.upload_bytes(org_id, image_key, png, "image/png")
        entry = {
            "item_id": str(item.id),
            "split": tile["split"],
            "image": image_key,
            "label": 1 if tile["rows"] else 0,
            "tile": [tile["col"], tile["row"]],
            "scale_m": tile["scale_m"],
        }
        if ds.task in ("detect", "segment", "yolo"):
            targets = [
                {
                    "bbox": list(geometry_to_yolo_bbox(r["geometry"], tile["crop_bbox"])),
                    **({"class": r["class_index"]} if ds.task == "yolo" else {}),
                    **({"polygon": geometry_to_yolo_polygon(r["geometry"], tile["crop_bbox"])} if ds.task == "segment" else {}),
                }
                for r in tile["rows"]
            ]
            targets_key = f"{base}/targets/{file_key}.json"
            storage_service.upload_bytes(
                org_id, targets_key, json.dumps({"targets": targets}).encode(), "application/json"
            )
            entry["targets"] = targets_key
        return entry

    manifest_items: list[dict] = []
    errors: list[str] = []
    processed = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(process, t): t for t in kept}
        for fut in as_completed(futures):
            tile = futures[fut]
            try:
                manifest_items.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                logger.exception("dino_dataset_tile_failed dataset=%s tile=%s", ds.id, tile["tile_key"])
                errors.append(f"{tile['tile_key']}: {exc}")
            processed += 1
            _progress(session, job, processed, len(errors))

    manifest_items.sort(key=lambda e: (e["split"], e["image"]))
    # Stats reflect what was actually written, not what was planned.
    written = {e["image"] for e in manifest_items}
    kept_ok = [
        t for t in kept
        if f"{prefix}/{t['split']}/images/{t['tile_key'].replace(':', '_')}.png" in written
    ]
    stats = split_stats(kept_ok, params)
    if stats["splits"]["train"]["positives"] == 0:
        _fail(session, job, ds, "Every positive training tile failed to export.", {"errors": errors[:MAX_ERRORS_KEPT]})
        return False

    positives_by_scale = {
        s_m: sum(len(t["rows"]) for t in kept_ok if t["scale_m"] == s_m) for s_m in params.scales_m
    }
    primary = max(params.scales_m, key=lambda s_m: positives_by_scale[s_m])
    ds.params = {**ds.params, "scales_m": list(params.scales_m), "tile_scale_m": primary}

    manifest = {
        "dataset_id": str(ds.id),
        "build_no": build_no,
        "class_id": str(ds.class_id),
        "class_name": class_name,
        "classes": [{"id": str(c), "name": n, "index": i} for i, (c, n) in enumerate(zip(class_ids, class_names, strict=True))],
        "task": ds.task,
        "params": ds.params,
        "items": manifest_items,
    }
    storage_service.upload_bytes(org_id, f"{prefix}/manifest.json", json.dumps(manifest).encode(), "application/json")

    warnings = list(stats["warnings"])
    if out_of_range:
        warnings.append(
            f"{out_of_range} annotation(s) are too large or too small for the scales used and may be "
            "clipped or tiny — add a larger/smaller scale."
        )
    stats = {**stats, "warnings": warnings}
    ds.stats = {
        **stats,
        "annotations": len(rows),
        "objects_out_of_range": out_of_range,
        "object_size_p10_m": sizing["p10_m"],
        "object_size_p90_m": sizing["p90_m"],
        "size_spread": sizing["spread"],
        "scales_m": list(params.scales_m),
        "primary_scale_m": primary,
        "items": len(by_item),
        "sets_skipped": skipped,
        "annotations_skipped_other_aoi": skipped_other_aoi,
        "annotations_skipped_boxes": boxes_skipped,
        "by_class": {n: sum(1 for r in rows if r["class_index"] == i) for i, n in enumerate(class_names)},
        "tiles_skipped_ambiguous": ambiguous_dropped,
        "tiles_failed": len(errors),
        "errors": errors[:MAX_ERRORS_KEPT],
        "tile_scale_m": primary,
        "crop_px": params.crop_px,
    }
    ds.data_fingerprint = fingerprint
    ds.s3_prefix = prefix
    ds.source_build_no = None
    return True


def _included_set_ids(session, collection) -> list[uuid.UUID]:
    from geoops.export_common import partition_member_sets_sync  # noqa: PLC0415

    ids, _skipped = partition_member_sets_sync(session, collection)
    return ids


def _build_from_source(session, job: Job, ds: DinoDataset, build_no: int) -> bool:
    """Classify dataset cloned from a detect one: same tiles + splits, labels
    derived from the detect targets (1 if the tile has any box)."""
    org_id = job.organization_id
    src = session.get(DinoDataset, ds.source_dataset_id) if ds.source_dataset_id else None
    if src is None or src.status != "ready" or not src.s3_prefix or src.task != "detect":
        _fail(session, job, ds, "The source detect dataset is missing or not ready.")
        return False

    manifest = storage_service.get_object_json(org_id, f"{src.s3_prefix}/manifest.json")
    src_items = manifest.get("items", [])
    job.total_items = len(src_items)
    session.commit()

    def label_of(it: dict) -> dict:
        targets = storage_service.get_object_json(org_id, it["targets"]).get("targets", [])
        return {
            "item_id": it["item_id"],
            "split": it["split"],
            "image": it["image"],  # shared with the source — not copied
            "label": 1 if targets else 0,
            "n_boxes": len(targets),
            "tile": it.get("tile"),
            "scale_m": it.get("scale_m"),
        }

    items: list[dict] = []
    errors: list[str] = []
    processed = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(label_of, it): it for it in src_items}
        for fut in as_completed(futures):
            try:
                items.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{futures[fut].get('image')}: {exc}")
            processed += 1
            _progress(session, job, processed, len(errors))

    counts = {s: {"tiles": 0, "positives": 0, "negatives": 0, "annotations": 0} for s in ("train", "eval", "test")}
    for it in items:
        c = counts[it["split"]]
        c["tiles"] += 1
        c["positives" if it["label"] else "negatives"] += 1
        c["annotations"] += it.pop("n_boxes", 0)
    if counts["train"]["positives"] == 0 or counts["train"]["negatives"] == 0:
        _fail(session, job, ds, "A classifier needs both positive and negative tiles in the train split.")
        return False

    prefix = f"dino-datasets/{ds.id}/b{build_no}"
    items.sort(key=lambda e: (e["split"], e["image"]))
    storage_service.upload_bytes(
        org_id,
        f"{prefix}/manifest.json",
        json.dumps(
            {
                "dataset_id": str(ds.id),
                "build_no": build_no,
                "source_dataset_id": str(src.id),
                "class_id": str(ds.class_id),
                "class_name": manifest.get("class_name"),
                "task": "classify",
                "params": src.params,
                "items": items,
            }
        ).encode(),
        "application/json",
    )
    warnings = [w for w in (src.stats or {}).get("warnings", [])]
    ds.params = dict(src.params)
    ds.stats = {
        **{k: v for k, v in (src.stats or {}).items() if k not in ("splits", "warnings", "errors", "tiles_failed")},
        "splits": counts,
        "warnings": warnings,
        "tiles_failed": len(errors),
        "errors": errors[:MAX_ERRORS_KEPT],
    }
    ds.data_fingerprint = src.data_fingerprint
    ds.s3_prefix = prefix
    ds.source_build_no = src.build_no
    return True


@celery_app.task(bind=True, queue=TRAINING, max_retries=0)
def run_dino_dataset_build(self, job_id: str) -> None:
    with WorkerSession() as session:
        job = session.get(Job, uuid.UUID(job_id))
        if job is None:
            logger.warning("run_dino_dataset_build: job %s not found", job_id)
            return
        ds = session.get(DinoDataset, uuid.UUID(job.config["dataset_id"]))
        if ds is None:
            job.status = JobStatus.FAILED
            job.logs = "Dataset no longer exists."
            job.finished_at = _now()
            session.commit()
            return

        job.status = JobStatus.RUNNING
        job.started_at = _now()
        ds.status = "building"
        ds.error = None
        session.commit()

        build_no = ds.build_no + 1
        try:
            ok = (
                _build_from_source(session, job, ds, build_no)
                if ds.source_dataset_id
                else _build_from_annotations(session, job, ds, build_no)
            )
            if not ok:
                return
            ds.build_no = build_no
            ds.status = "ready"
            job.status = JobStatus.COMPLETED
            sp = ds.stats["splits"]
            job.logs = (
                f"Built dataset b{build_no}: {sp['train']['tiles']} train / {sp['eval']['tiles']} eval / "
                f"{sp['test']['tiles']} test tiles."
            )
            job.config = {**(job.config or {}), "result": {"build_no": build_no, "stats": ds.stats}}
            job.finished_at = _now()
            session.commit()
        except Exception as exc:
            session.rollback()
            ds2 = session.get(DinoDataset, uuid.UUID(job.config["dataset_id"]))
            job2 = session.get(Job, uuid.UUID(job_id))
            if ds2 is not None:
                ds2.status = "failed"
                ds2.error = str(exc)[:2000]
            if job2 is not None:
                job2.status = JobStatus.FAILED
                job2.logs = str(exc)[:2000]
                job2.finished_at = _now()
            session.commit()
            raise
