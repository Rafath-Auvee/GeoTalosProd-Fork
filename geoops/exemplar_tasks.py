"""Exemplar-based detection-candidate generation ("similarity scan v2").

Generalizes ``geoops.tasks.run_aoi_scan`` for the annotation-prep workflow:
multiple reference exemplars (not one), an auto-inferred working resolution
and SAHI-style *overlapping* tile grid (not the fixed non-overlapping
``canonical_tiles`` grid), duplicate-suppression across overlapping tiles via
``supervision``'s NMS, optional box tightening through the already-registered
SAM3 model (reusing ``ModelManager``/``sam3_adapter`` — no new SAM-calling
code), and hard-negative suppression from previously-rejected candidates.

Two entry points share one scan core (``_run_similarity_scan_core``):

- ``run_exemplar_scan`` — references are embeddings the caller already
  created (manually drawn boxes, or picked annotations) — see
  ``POST /embeddings/exemplar-scan``.
- ``run_auto_annotate`` — references are auto-gathered from a verified
  ``AnnotationSetCollection`` + class (the same "collection + class" input
  Train Head takes — see ``geoops.dino_tasks._gather_rows_for_class``), so the
  whole pipeline is "pick a class, click run" — see
  ``POST /embeddings/auto-annotate``.

Left deliberately separate from ``run_aoi_scan``/``run_anomaly_detection``
rather than folded in — those stay simple single-reference / anomaly-centroid
tools; this is the heavier multi-exemplar loop.
"""
from __future__ import annotations

import logging
import uuid

import numpy as np
import supervision as sv
from shapely.geometry import box, shape
from sqlalchemy import insert, select

from app.core.enums import JobStatus
from app.models.ai_model import AIModel
from app.models.dataset_item import DatasetItem
from app.models.job import Job
from app.services.model_manager import ModelManager
from app.workers.celery_app import celery_app
from app.workers.db import WorkerSession
from app.workers.queues import EMBEDDING

from geoops.models import Embedding, EmbeddingTile
from geoops.scale import (
    bbox_span_m,
    cosine_distance,
    meters_to_degrees,
    nearest_tier,
    working_resolution_from_refs,
)
from geoops.tasks import (
    _bbox_to_polygon,
    _call_embed_model,
    _fetch_patch_png_b64,
    _intersect,
    _now,
    _persist_results_annotation_set,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 25
MAX_ERROR_SAMPLE = 50
# "At least one object width" overlap floor from the reference workflow, and a
# ceiling so a pathologically large object doesn't force near-total overlap
# (every tile re-embedding almost the same area).
MIN_OVERLAP_RATIO = 0.2
MAX_OVERLAP_RATIO = 0.6
# Refined boxes further than this multiple of the mean reference span (in
# either direction) are dropped as implausible SAM3 output, not a real
# instance of the same object.
SIZE_TOLERANCE_LOW = 0.4
SIZE_TOLERANCE_HIGH = 2.5
# Auto-annotate: how many of a class's verified annotations to use as
# references by default — few-shot, not "embed the whole collection".
DEFAULT_MAX_REFERENCES = 5


def overlapping_tiles(
    item_bbox: list[float],
    clip_bbox: list[float] | None,
    tile_w_deg: float,
    tile_h_deg: float,
    overlap_ratio: float,
) -> list[tuple[int, int, list[float]]]:
    """SAHI-style overlapping tile grid over ``item_bbox`` (clipped to
    ``clip_bbox``), stepping by ``tile_*_deg * (1 - overlap_ratio)`` instead of
    the full tile size — consecutive tiles overlap by ``overlap_ratio`` of
    their width/height, so an object near a tile boundary is still fully
    inside at least one neighboring tile.

    Unlike ``geoops.scale.canonical_tiles``, tiles are anchored to the clipped
    scan region's own origin, not the dataset item's — overlap ratio (and
    therefore the grid) can differ per scan, so there's no fixed grid to keep
    stable across requests the way the non-overlapping cache grid has. Within
    one scan, re-embedding the same tile twice still can't happen (each
    (col, row) is visited once); across separate scans of an overlapping AOI,
    ``embedding_tiles`` may hold near-duplicate tiles at slightly different
    offsets rather than an exact cache hit — an acceptable cost for this
    heavier, less frequently re-run scan.
    """
    if not (0.0 < overlap_ratio < 1.0):
        raise ValueError("overlap_ratio must be strictly between 0 and 1")

    minx, miny, maxx, maxy = item_bbox
    scan_minx, scan_miny, scan_maxx, scan_maxy = minx, miny, maxx, maxy
    if clip_bbox is not None:
        scan_minx = max(scan_minx, clip_bbox[0])
        scan_miny = max(scan_miny, clip_bbox[1])
        scan_maxx = min(scan_maxx, clip_bbox[2])
        scan_maxy = min(scan_maxy, clip_bbox[3])
        if scan_minx >= scan_maxx or scan_miny >= scan_maxy:
            return []

    step_w = tile_w_deg * (1.0 - overlap_ratio)
    step_h = tile_h_deg * (1.0 - overlap_ratio)

    tiles: list[tuple[int, int, list[float]]] = []
    row = 0
    y = scan_miny
    while y < scan_maxy:
        col = 0
        x = scan_minx
        while x < scan_maxx:
            tiles.append((col, row, [x, y, min(x + tile_w_deg, scan_maxx), min(y + tile_h_deg, scan_maxy)]))
            x += step_w
            col += 1
        y += step_h
        row += 1
    return tiles


def _refine_with_sam3(
    session, sam_model: AIModel, item: DatasetItem, candidate_bbox: list[float], crop_size_px: int
) -> dict | None:
    """Tighten one candidate box via the already-registered SAM3 model, reusing
    ModelManager._call_model + the sam3_to_platform adapter's bbox-prompt path
    (app.automation.adapters.sam3_adapter) exactly as the interactive
    bbox-exemplar flow does — no new SAM-calling code.

    Returns the highest-confidence prediction dict ({"label", "confidence",
    "geometry", "properties"}) or None if SAM3 returned nothing usable.
    """
    from app.automation.adapters import get_adapter  # noqa: PLC0415

    output_cfg = dict(sam_model.output_config or {})
    adapter_name = output_cfg.get("adapter", "sam3_to_platform")
    adapter = get_adapter(adapter_name)
    adapter_config = output_cfg.get("adapter_config") or {}

    patch_context = {"bbox": candidate_bbox, "width": crop_size_px, "height": crop_size_px}
    prompt_spec = dict(output_cfg.get("prompt_payload") or {})
    prompt_spec["bbox_prompts_4326"] = [candidate_bbox]

    if adapter.prompt_resolver is not None:
        prompt_payload = adapter.prompt_resolver(prompt_spec, patch_context, adapter_config)
        if prompt_payload is None:
            return None
    else:
        prompt_payload = prompt_spec

    patch_png_b64 = _fetch_patch_png_b64(item, candidate_bbox, crop_size_px)
    manager = ModelManager(session)
    raw_output = manager._call_model(  # noqa: SLF001 — reusing the generic call path deliberately
        model=sam_model,
        item=item,
        georef_context=patch_context,
        patch={"bbox": candidate_bbox, "width_px": crop_size_px, "height_px": crop_size_px},
        patch_image_base64=patch_png_b64,
        adapter=adapter,
        adapter_config=adapter_config,
        prompt_payload=prompt_payload,
    )
    normalized = adapter.convert_fn(raw_output, adapter_config, patch_context)
    predictions = normalized.get("predictions") or []
    if not predictions:
        return None
    return max(predictions, key=lambda p: p.get("confidence", 0.0))


def _gather_reference_embeddings_sync(
    session,
    org_id: uuid.UUID,
    model: AIModel,
    collection_id: uuid.UUID,
    class_id: uuid.UUID,
    max_references: int,
    crop_size_px: int,
) -> tuple[list[Embedding], str]:
    """Auto-gather up to ``max_references`` reference embeddings from a
    verified ``AnnotationSetCollection``'s existing annotations of one class —
    the same "collection + class" input Train Head takes
    (``geoops.dino_tasks._gather_rows_for_class``), so auto-annotate needs no
    manual box-drawing step. Embeds any not already cached (one embedding per
    (annotation, model) — ``uq_embeddings_annotation_model``), then returns
    the strongest-agreeing tile-tier group (annotations of wildly different
    real-world size can't all be used as one scan's references — see
    geoops/scale.py). Returns (references, class_name).
    """
    from geoops.dino_tasks import _gather_rows_for_class  # noqa: PLC0415 — avoids a hard import cycle at module load

    rows, _skipped, _collection, class_name = _gather_rows_for_class(session, org_id, collection_id, class_id)
    if not rows:
        raise ValueError(
            "No eligible (verified, resolvable-item) annotations of this class found in this collection."
        )
    selected = rows[:max_references]
    annotation_ids = [r["annotation_id"] for r in selected]

    existing_by_annotation = {
        e.annotation_id: e
        for e in session.execute(
            select(Embedding).where(Embedding.annotation_id.in_(annotation_ids), Embedding.model_id == model.id)
        ).scalars().all()
    }
    to_embed = [r for r in selected if r["annotation_id"] not in existing_by_annotation]

    new_rows: list[dict] = []
    for r in to_embed:
        bbox = list(shape(r["geometry"]).bounds)
        patch_b64 = _fetch_patch_png_b64(r["item"], bbox, crop_size_px)
        embed_result = _call_embed_model(model, r["item"], patch_b64, bbox)
        new_rows.append(
            {
                "organization_id": org_id,
                "model_id": model.id,
                "model_name": embed_result["model_name"],
                "embedding_dim": embed_result["embedding_dim"],
                "dataset_item_id": r["item"].id,
                "class_id": class_id,
                "annotation_id": r["annotation_id"],
                "source_geometry": r["geometry"],
                "tile_tier": nearest_tier(bbox_span_m(r["geometry"])),
                "embedding": embed_result["embedding"],
            }
        )
    if new_rows:
        session.execute(insert(Embedding.__table__), new_rows)
        session.flush()

    references = session.execute(
        select(Embedding).where(Embedding.annotation_id.in_(annotation_ids), Embedding.model_id == model.id)
    ).scalars().all()
    if not references:
        raise ValueError("Failed to embed any reference annotations for this class.")

    # Annotations of this class can legitimately span more than one real-world
    # scale tier (e.g. a partly-labeled AOI with both close-up and wide shots)
    # — a scan needs one consistent tile size, so keep only the majority tier
    # rather than erroring out the way the manual exemplar-scan path does
    # (there, mismatched tiers mean the user picked the wrong boxes).
    tier_counts: dict[int, int] = {}
    for r in references:
        tier_counts[r.tile_tier] = tier_counts.get(r.tile_tier, 0) + 1
    majority_tier = max(tier_counts, key=lambda t: tier_counts[t])
    references = [r for r in references if r.tile_tier == majority_tier]
    return references, class_name


def _run_similarity_scan_core(
    session,
    job: Job,
    model: AIModel,
    tier: int,
    references: list[Embedding],
    negatives: list[Embedding],
    sam_model: AIModel | None,
    items: list[DatasetItem],
    aoi_bbox: list[float] | None,
    crop_size_px: int,
    top_k: int,
    min_similarity: float | None,
    max_patches: int,
    iou_thresh: float,
    output_class_id: uuid.UUID,
    name_prefix: str,
) -> None:
    """Shared body of the similarity-scan job, used by both ``run_exemplar_scan``
    (manually-picked references) and ``run_auto_annotate`` (references sourced
    automatically from a verified annotation collection + class). Auto-infers
    working resolution from the references, tiles ``items`` SAHI-style with
    overlap, embeds/caches tiles, scores against the references (minus any
    negatives), merges duplicates via NMS, optionally refines via SAM3, and
    persists the result. Manages ``job.status``/progress/commit throughout;
    raises on failure for the caller's job-failure wrapper to catch.
    """
    model_names = {r.model_name for r in references}
    if len(model_names) != 1:
        raise ValueError("All reference embeddings must share the same model_name")

    # ── Auto-infer working resolution + slice geometry from the references ──
    ref_spans_m = [bbox_span_m(r.source_geometry) for r in references]
    mean_span_m = sum(ref_spans_m) / len(ref_spans_m)
    meters_per_pixel, object_px = working_resolution_from_refs([r.source_geometry for r in references])
    overlap_ratio = max(MIN_OVERLAP_RATIO, min(MAX_OVERLAP_RATIO, object_px / crop_size_px))

    # ── Build the full overlapping candidate tile list up front ──
    all_tiles: list[tuple[DatasetItem, int, int, list[float]]] = []
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
        for col, row, tile_bbox in overlapping_tiles(item_bbox, clip_bbox, tile_w_deg, tile_h_deg, overlap_ratio):
            all_tiles.append((item, col, row, tile_bbox))

    all_keys = {(item.id, col, row) for item, col, row, _ in all_tiles}
    bbox_by_key = {(item.id, col, row): tile_bbox for item, col, row, tile_bbox in all_tiles}

    existing_keys: set[tuple[uuid.UUID, int, int]] = set()
    if all_tiles:
        cached = session.execute(
            select(EmbeddingTile.dataset_item_id, EmbeddingTile.tile_col, EmbeddingTile.tile_row).where(
                EmbeddingTile.dataset_item_id.in_({k[0] for k in all_keys}),
                EmbeddingTile.model_id == model.id,
                EmbeddingTile.tile_tier == tier,
            )
        ).all()
        existing_keys = {(r[0], r[1], r[2]) for r in cached}

    missing = [t for t in all_tiles if (t[0].id, t[1], t[2]) not in existing_keys]
    patches_truncated = False
    if len(missing) > max_patches:
        missing = missing[:max_patches]
        patches_truncated = True

    job.total_items = len(missing)
    job.processed_items = 0
    job.failed_items = 0
    job.progress = 1.0 if not missing else 0.0
    session.commit()

    errors: list[dict] = []
    error_total = 0
    embedded = 0
    batch: list[dict] = []

    for idx, (item, col, row, tile_bbox) in enumerate(missing):
        try:
            patch_b64 = _fetch_patch_png_b64(item, tile_bbox, crop_size_px)
            embed_result = _call_embed_model(model, item, patch_b64, tile_bbox)
            if embed_result["model_name"] not in model_names:
                raise ValueError(
                    f"Model returned embeddings for {embed_result['model_name']!r}, "
                    f"expected one of {model_names!r}"
                )
            batch.append(
                {
                    "organization_id": job.organization_id,
                    "model_id": model.id,
                    "model_name": embed_result["model_name"],
                    "embedding_dim": embed_result["embedding_dim"],
                    "dataset_item_id": item.id,
                    "tile_tier": tier,
                    "tile_col": col,
                    "tile_row": row,
                    "bbox": _bbox_to_polygon(tile_bbox),
                    "embedding": embed_result["embedding"],
                    "created_by_job_id": job.id,
                }
            )
            embedded += 1
        except Exception as exc:  # noqa: BLE001 — one bad tile shouldn't fail the whole scan
            error_total += 1
            if len(errors) < MAX_ERROR_SAMPLE:
                errors.append({"dataset_item_id": str(item.id), "bbox": tile_bbox, "error": str(exc)})

        if len(batch) >= BATCH_SIZE or idx == len(missing) - 1:
            if batch:
                session.execute(insert(EmbeddingTile.__table__), batch)
                batch = []
            job.processed_items = idx + 1
            job.progress = (idx + 1) / len(missing)
            session.commit()

    # ── Score: max similarity to any positive reference, gated by negatives ──
    ref_vectors = [list(r.embedding) for r in references]
    neg_vectors = [list(n.embedding) for n in negatives]
    matches: list[dict] = []
    if all_keys:
        candidate_rows = session.execute(
            select(
                EmbeddingTile.dataset_item_id, EmbeddingTile.tile_col, EmbeddingTile.tile_row,
                EmbeddingTile.embedding,
            ).where(
                EmbeddingTile.dataset_item_id.in_({k[0] for k in all_keys}),
                EmbeddingTile.model_id == model.id,
                EmbeddingTile.tile_tier == tier,
            )
        ).all()
        for dataset_item_id, col, row, embedding in candidate_rows:
            key = (dataset_item_id, col, row)
            if key not in all_keys:
                continue
            vec = list(embedding)
            similarity_pos = max(1.0 - cosine_distance(vec, r) for r in ref_vectors)
            if min_similarity is not None and similarity_pos < min_similarity:
                continue
            if neg_vectors:
                similarity_neg = max(1.0 - cosine_distance(vec, n) for n in neg_vectors)
                if similarity_neg >= similarity_pos:
                    continue
            matches.append(
                {"dataset_item_id": dataset_item_id, "bbox": bbox_by_key[key], "similarity": similarity_pos}
            )

    # ── Merge duplicates across overlapping tiles (supervision NMS) ──
    # Geo-coordinate bboxes lose precision when cast to xyxy's required
    # float32 (e.g. -70.252 -> -70.25199890136719) — comparing NMS survivors
    # back to the original float64 boxes by value would silently match
    # nothing. Track survivors by index instead (stashed in class_id, which
    # NMS filters element-wise alongside xyxy/confidence regardless of
    # class_agnostic — that flag only affects whether suppression considers
    # class equality, not whether class_id itself is preserved).
    candidates_before_nms = len(matches)
    if matches:
        xyxy = np.array([m["bbox"] for m in matches], dtype=np.float32)
        confidence = np.array([m["similarity"] for m in matches], dtype=np.float32)
        detections = sv.Detections(
            xyxy=xyxy, confidence=confidence, class_id=np.arange(len(matches), dtype=int)
        )
        keep = detections.with_nms(threshold=iou_thresh, class_agnostic=True)
        kept_indices = set(keep.class_id.tolist())
        matches = [m for i, m in enumerate(matches) if i in kept_indices]

    matches.sort(key=lambda m: m["similarity"], reverse=True)
    top_matches = matches[:top_k]

    # ── Optional SAM3 box refinement ──
    items_by_id = {item.id: item for item in items}
    refine_errors = 0
    if sam_model is not None:
        for m in top_matches:
            try:
                item = items_by_id[m["dataset_item_id"]]
                refined = _refine_with_sam3(session, sam_model, item, m["bbox"], crop_size_px)
                if refined is None:
                    continue
                refined_span_m = bbox_span_m(refined["geometry"])
                if not (SIZE_TOLERANCE_LOW * mean_span_m <= refined_span_m <= SIZE_TOLERANCE_HIGH * mean_span_m):
                    continue
                m["geometry"] = refined["geometry"]
                m["refined"] = True
            except Exception as exc:  # noqa: BLE001 — refinement is best-effort
                refine_errors += 1
                logger.warning("similarity_scan_refine_failed job=%s bbox=%s error=%s", job.id, m["bbox"], exc)

    # ── Persist top_k matches as a new annotation_set/layer ──
    result_dataset_id = items[0].dataset_id if items else None
    annotation_set_id = _persist_results_annotation_set(
        session, job, output_class_id, model.id, result_dataset_id,
        name=f"{name_prefix} — {len(top_matches)} matches",
        rows=[
            {
                "geometry": shape(m["geometry"]) if "geometry" in m else box(*m["bbox"]),
                "confidence": max(0.0, min(1.0, (m["similarity"] + 1.0) / 2.0)),
                "properties": {
                    "dataset_item_id": str(m["dataset_item_id"]),
                    "similarity": m["similarity"],
                    "refined": m.get("refined", False),
                },
            }
            for m in top_matches
        ],
    )

    cfg = dict(job.config or {})
    cfg["result"] = {
        "model_name": model.name,
        "meters_per_pixel": meters_per_pixel,
        "object_px": object_px,
        "overlap_ratio": overlap_ratio,
        "patches_scanned": len(all_tiles),
        "patches_cached": len(all_tiles) - len(missing),
        "patches_embedded": embedded,
        "patches_truncated": patches_truncated,
        "candidates_before_nms": candidates_before_nms,
        "errors_total": error_total,
        "errors_sample": errors,
        "refine_errors": refine_errors,
        "matches": [
            {
                "dataset_item_id": str(m["dataset_item_id"]),
                "bbox": m["bbox"],
                "similarity": m["similarity"],
                "refined": m.get("refined", False),
            }
            for m in top_matches
        ],
        "annotation_set_id": str(annotation_set_id) if annotation_set_id else None,
    }
    job.config = cfg
    job.processed_items = len(missing)
    job.failed_items = error_total
    job.progress = 1.0
    job.status = JobStatus.COMPLETED
    job.finished_at = _now()
    session.commit()

    logger.info(
        "similarity_scan done job=%s scanned=%d cached=%d embedded=%d matches=%d refined_errors=%d",
        job.id, len(all_tiles), len(all_tiles) - len(missing), embedded, len(top_matches), refine_errors,
    )


@celery_app.task(bind=True, queue=EMBEDDING, max_retries=2, default_retry_delay=60)
def run_exemplar_scan(self, job_id: str) -> None:
    """Multi-exemplar similarity scan with SAHI-style overlapping slices, NMS
    merge, and optional SAM3 box refinement. Reads ``ExemplarScanRequest``
    (see geoops.schemas) from ``job.config["request"]``; results land in
    ``job.config["result"]`` and as a new reviewable ``AnnotationSet``.
    """
    job_uuid = uuid.UUID(job_id)

    with WorkerSession() as session:
        job = session.get(Job, job_uuid)
        if job is None:
            logger.error("run_exemplar_scan: job %s not found", job_id)
            return

        cfg = dict(job.config or {})
        req = cfg.get("request") or {}
        try:
            reference_ids = [uuid.UUID(i) for i in req["reference_embedding_ids"]]
            negative_ids = [uuid.UUID(i) for i in req.get("negative_embedding_ids", [])]
            dataset_item_ids = [uuid.UUID(i) for i in req["dataset_item_ids"]]
            aoi_bbox = req.get("aoi_bbox")
            crop_size_px = int(req.get("crop_size_px", 128))
            top_k = int(req.get("top_k", 20))
            min_similarity = req.get("min_similarity")
            max_patches = int(req.get("max_patches", 200))
            iou_thresh = float(req.get("iou_thresh", 0.3))
            output_class_id = uuid.UUID(req["output_class_id"])
            refine_with_sam = bool(req.get("refine_with_sam", False))
            sam_model_id = uuid.UUID(req["sam_model_id"]) if req.get("sam_model_id") else None
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
            references = session.execute(
                select(Embedding).where(
                    Embedding.id.in_(reference_ids), Embedding.organization_id == job.organization_id
                )
            ).scalars().all()
            if len(references) != len(set(reference_ids)):
                raise ValueError("One or more reference embeddings not found")
            tiers = {r.tile_tier for r in references}
            if len(tiers) != 1:
                raise ValueError("All reference embeddings must share the same tile_tier")
            tier = tiers.pop()

            negatives: list[Embedding] = []
            if negative_ids:
                negatives = session.execute(
                    select(Embedding).where(
                        Embedding.id.in_(negative_ids), Embedding.organization_id == job.organization_id
                    )
                ).scalars().all()

            model = session.get(AIModel, references[0].model_id)
            if model is None or not model.endpoint_url:
                raise ValueError("Reference embeddings' model no longer exists or has no endpoint_url")

            sam_model: AIModel | None = None
            if refine_with_sam:
                if sam_model_id is None:
                    raise ValueError("refine_with_sam requires sam_model_id")
                sam_model = session.get(AIModel, sam_model_id)
                if sam_model is None or sam_model.organization_id != job.organization_id or not sam_model.endpoint_url:
                    raise ValueError("SAM refinement model not found or has no endpoint_url")

            items = session.execute(
                select(DatasetItem).where(
                    DatasetItem.id.in_(dataset_item_ids),
                    DatasetItem.organization_id == job.organization_id,
                    DatasetItem.is_active.is_(True),
                )
            ).scalars().all()

            _run_similarity_scan_core(
                session, job, model, tier, references, negatives, sam_model, items, aoi_bbox,
                crop_size_px, top_k, min_similarity, max_patches, iou_thresh, output_class_id,
                name_prefix="Exemplar Scan",
            )

        except Exception as exc:
            session.rollback()
            logger.exception("run_exemplar_scan failed job=%s", job_id)
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


@celery_app.task(bind=True, queue=EMBEDDING, max_retries=2, default_retry_delay=60)
def run_auto_annotate(self, job_id: str) -> None:
    """"Pick a class, click run" version of the exemplar scan — references are
    sourced automatically from a verified ``AnnotationSetCollection``'s
    existing annotations of one class (same inputs as Train Head: collection +
    class), instead of drawing reference boxes by hand. Reads
    ``AutoAnnotateRequest`` (see geoops.schemas) from ``job.config["request"]``.
    """
    job_uuid = uuid.UUID(job_id)

    with WorkerSession() as session:
        job = session.get(Job, job_uuid)
        if job is None:
            logger.error("run_auto_annotate: job %s not found", job_id)
            return

        cfg = dict(job.config or {})
        req = cfg.get("request") or {}
        try:
            collection_id = uuid.UUID(req["annotation_set_collection_id"])
            class_id = uuid.UUID(req["class_id"])
            embedding_model_id = uuid.UUID(req["embedding_model_id"])
            dataset_item_ids = [uuid.UUID(i) for i in req["dataset_item_ids"]]
            aoi_bbox = req.get("aoi_bbox")
            max_references = int(req.get("max_references", DEFAULT_MAX_REFERENCES))
            crop_size_px = int(req.get("crop_size_px", 128))
            top_k = int(req.get("top_k", 20))
            min_similarity = req.get("min_similarity")
            max_patches = int(req.get("max_patches", 200))
            iou_thresh = float(req.get("iou_thresh", 0.3))
            refine_with_sam = bool(req.get("refine_with_sam", False))
            sam_model_id = uuid.UUID(req["sam_model_id"]) if req.get("sam_model_id") else None
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
            model = session.get(AIModel, embedding_model_id)
            if model is None or model.organization_id != job.organization_id or not model.endpoint_url:
                raise ValueError("Embedding model not found or has no endpoint_url")

            sam_model: AIModel | None = None
            if refine_with_sam:
                if sam_model_id is None:
                    raise ValueError("refine_with_sam requires sam_model_id")
                sam_model = session.get(AIModel, sam_model_id)
                if sam_model is None or sam_model.organization_id != job.organization_id or not sam_model.endpoint_url:
                    raise ValueError("SAM refinement model not found or has no endpoint_url")

            references, class_name = _gather_reference_embeddings_sync(
                session, job.organization_id, model, collection_id, class_id, max_references, crop_size_px
            )
            tier = references[0].tile_tier

            items = session.execute(
                select(DatasetItem).where(
                    DatasetItem.id.in_(dataset_item_ids),
                    DatasetItem.organization_id == job.organization_id,
                    DatasetItem.is_active.is_(True),
                )
            ).scalars().all()

            _run_similarity_scan_core(
                session, job, model, tier, references, [], sam_model, items, aoi_bbox,
                crop_size_px, top_k, min_similarity, max_patches, iou_thresh, output_class_id,
                name_prefix=f"Auto Annotate — {class_name}",
            )

        except Exception as exc:
            session.rollback()
            logger.exception("run_auto_annotate failed job=%s", job_id)
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


__all__ = ["run_exemplar_scan", "run_auto_annotate", "overlapping_tiles"]
