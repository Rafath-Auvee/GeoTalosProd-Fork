import asyncio
import io
from collections import OrderedDict
from uuid import UUID

import numpy as np

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_session, require_org_role
from app.core.enums import JobStatus, JobType
from app.core.geometry import parse_geometry
from app.models.annotation import Annotation
from app.models.annotation_class import AnnotationClass
from app.models.annotation_schema import AnnotationSchema
from app.models.annotation_set import AnnotationSet
from app.models.dataset_item import DatasetItem
from app.models.map import Map
from app.models.map_annotation_set import MapAnnotationSet
from app.models.map_aoi import MapAOI
from app.models.project import Project
from app.models.project_annotation_set import ProjectAnnotationSet
from app.models.job import Job
from app.models.user import User
from app.schemas.job import JobRead
from app.services import storage_service

from geoops.dino_service import DinoExportService
from geoops.sam3_point_service import PointMaskError, compute_point_mask
from geoops.interactive_tasks import MAX_AOI_HECTARES, MAX_TILES, masks_key, plan_tiles
from geoops.interactive_cluster import group, normalize, similarity
from geoops.interactive_tasks import EMBED_TIER, MAX_EMBED_MASKS, embeddings_prefix
from app.services.annotation_set_service import _set_in_org_clause
from geoops.sam3_refine import refine_masks
from geoops.schemas import (
    Sam3RefineMasksRequest,
    Sam3RefineMasksResponse,
    Sam3ApplyRequest,
    Sam3ApplyResponse,
    Sam3ExistingAnnotation,
    Sam3ExistingResponse,
    Sam3AutoAnnotateRequest,
    Sam3ClusterRequest,
    Sam3CommitRequest,
    Sam3CommitResponse,
    Sam3CropRequest,
    Sam3ClusterResponse,
    Sam3MaskEmbedRequest,
    Sam3SimilarRequest,
    Sam3SimilarResponse,
    Sam3ThumbsRequest,
    Sam3ThumbsResponse,
    Sam3PointMaskRequest,
    Sam3RefineSetRequest,
    Sam3PointMaskResponse,
    Sam3SegmentAllMasks,
    Sam3SegmentAllRequest,
)

router = APIRouter(prefix="/sam3", tags=["sam3"])


@router.post("/auto-annotate", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def auto_annotate(
    payload: Sam3AutoAnnotateRequest,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """"Pick a class, click run" SAM3 box+text auto-annotate over an AOI
    (async job).

    Same collection + class inputs as ``POST /dino/finetune`` (Train Head)
    build the reference set — no manual reference-box drawing needed — but
    unlike a single-tile refinement, every tile of ``dataset_item_ids``
    (the AOI) gets scanned: a reference's own tile gets its bbox as a SAM3
    box exemplar, everything else is scanned via SAM3's text-only prompt.
    Poll ``GET /api/jobs/{job_id}`` for ``config.result`` once ``status ==
    "completed"``.
    """
    service = DinoExportService(db)
    preview = await service.build_preview(org_id, payload.annotation_set_collection_id, payload.class_id)
    if not preview.ready:
        raise HTTPException(
            status_code=400,
            detail="Nothing eligible: 0 verified, resolvable-item annotations of this class.",
        )

    rows = await db.scalars(
        select(DatasetItem).where(
            DatasetItem.id.in_(payload.dataset_item_ids),
            DatasetItem.organization_id == org_id,
            DatasetItem.is_active.is_(True),
        )
    )
    if len(rows.all()) != len(set(payload.dataset_item_ids)):
        raise HTTPException(status_code=404, detail="One or more dataset items not found")

    job = Job(
        organization_id=org_id,
        type=JobType.SAM3_AUTO_ANNOTATE,
        status=JobStatus.QUEUED,
        config={"request": payload.model_dump(mode="json")},
        input_refs=[
            {"type": "annotation_set_collection", "id": str(payload.annotation_set_collection_id)},
            {"type": "annotation_class", "id": str(payload.class_id)},
        ]
        + [{"type": "dataset_item", "id": str(item_id)} for item_id in payload.dataset_item_ids],
        created_by_user_id=current_user.id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    from geoops.sam3_tasks import run_sam3_auto_annotate  # noqa: PLC0415

    run_sam3_auto_annotate.apply_async(args=[str(job.id)])
    return job


@router.post("/point-mask", response_model=Sam3PointMaskResponse)
async def point_mask(
    payload: Sam3PointMaskRequest,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """One object's mask from positive/negative clicks (SAM 3 interactive head).

    Synchronous on purpose: the Interactive annotation picker calls it on every
    click. Positive points (label 1) add to the object, negative points
    (label 0) exclude areas; all points refine the same mask.
    """
    item = await db.scalar(
        select(DatasetItem).where(
            DatasetItem.dataset_id == payload.dataset_id,
            DatasetItem.stac_item_id == payload.stac_item_id,
            DatasetItem.organization_id == org_id,
            DatasetItem.is_active.is_(True),
        )
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Imagery item not found")
    try:
        return await asyncio.to_thread(compute_point_mask, item, payload)
    except PointMaskError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@router.post("/refine-masks", response_model=Sam3RefineMasksResponse)
async def refine_existing_masks(
    payload: Sam3RefineMasksRequest,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Second opinion on existing masks: SAM 3 from 1–2 points inside each, unioned with the original
    when it plausibly describes the same object. Per-mask failures come back as a reason, not an error."""
    item = await db.scalar(
        select(DatasetItem).where(
            DatasetItem.dataset_id == payload.dataset_id, DatasetItem.stac_item_id == payload.stac_item_id,
            DatasetItem.organization_id == org_id, DatasetItem.is_active.is_(True),
        )
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Imagery item not found")
    return Sam3RefineMasksResponse(results=await asyncio.to_thread(refine_masks, item, payload))


# ── Segment all (Interactive annotation mode) ────────────────────────────────

_SEGMENT_ALL_TIER = "interactive_segment_all"


async def _segment_all_job(db: AsyncSession, org_id: UUID, job_id: UUID) -> Job:
    job = await db.get(Job, job_id)
    if job is None or job.organization_id != org_id or (job.config or {}).get("tier") != _SEGMENT_ALL_TIER:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/segment-all", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def segment_all(
    payload: Sam3SegmentAllRequest,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """"Segment everything" in a small AOI: SAM 3 point-grid masks at each scale,
    cleaned with the platform's IoU NMS. Poll ``GET /jobs/{id}`` for progress and
    ``GET /sam3/segment-all/{id}/masks`` for the masks found so far."""
    from shapely.geometry import shape  # noqa: PLC0415

    item = await db.scalar(
        select(DatasetItem).where(
            DatasetItem.dataset_id == payload.dataset_id,
            DatasetItem.stac_item_id == payload.stac_item_id,
            DatasetItem.organization_id == org_id,
            DatasetItem.is_active.is_(True),
        )
    )
    if item is None or not item.geometry:
        raise HTTPException(status_code=404, detail="Imagery item not found")

    from shapely.geometry import box  # noqa: PLC0415

    from geoops.scale import bbox_coords_span_m  # noqa: PLC0415

    b = payload.aoi_bbox
    width_m = bbox_coords_span_m([b[0], (b[1] + b[3]) / 2, b[2], (b[1] + b[3]) / 2])
    height_m = bbox_coords_span_m([(b[0] + b[2]) / 2, b[1], (b[0] + b[2]) / 2, b[3]])
    hectares = width_m * height_m / 10_000
    if hectares > MAX_AOI_HECTARES:
        raise HTTPException(status_code=422, detail=f"This AOI is {hectares:.0f} ha; interactive mode is limited to {MAX_AOI_HECTARES:.0f} ha.")

    aoi_poly = shape(payload.aoi_geometry) if payload.aoi_geometry else box(*b)
    plan = plan_tiles(list(shape(item.geometry).bounds), b, aoi_poly, payload.scales_m)
    tiles = sum(len(v) for v in plan.values())
    if tiles == 0:
        raise HTTPException(status_code=422, detail="The AOI doesn't overlap the selected imagery.")
    if tiles > MAX_TILES:
        raise HTTPException(status_code=422, detail=f"{tiles} tiles is over the {MAX_TILES} limit. Use fewer or larger scales.")

    job = Job(
        organization_id=org_id,
        type=JobType.SAM3_AUTO_ANNOTATE,
        status=JobStatus.QUEUED,
        config={"tier": _SEGMENT_ALL_TIER, "request": payload.model_dump(mode="json")},
        input_refs=[{"type": "dataset_item", "id": str(item.id)}],
        created_by_user_id=current_user.id,
        total_items=tiles,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    from geoops.interactive_tasks import run_interactive_segment_all  # noqa: PLC0415

    run_interactive_segment_all.apply_async(args=[str(job.id)])
    return job


@router.post("/refine-annotation-set", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def refine_annotation_set(
    payload: Sam3RefineSetRequest,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Generate masks for a set of boxes: SAM 3 is prompted with each box, and the masks are saved as a
    new annotation set laid out like the interactive ones (geometry = mask, ``properties.bbox`` = its box).
    Poll ``GET /jobs/{id}``; ``config.result.annotation_set_id`` is the new set."""
    src = await db.scalar(
        select(AnnotationSet).where(
            AnnotationSet.id == payload.annotation_set_id, AnnotationSet.organization_id == org_id, AnnotationSet.deleted_at.is_(None)
        )
    )
    if src is None:
        raise HTTPException(status_code=404, detail="Annotation set not found")
    if payload.map_id is not None:
        map_row = await db.get(Map, payload.map_id)
        project = await db.get(Project, map_row.project_id) if map_row else None
        if map_row is None or project is None or project.organization_id != org_id:
            raise HTTPException(status_code=404, detail="Map not found")
    n = await db.scalar(
        select(func.count(Annotation.id)).where(Annotation.annotation_set_id == src.id, Annotation.deleted_at.is_(None))
    )
    if not n:
        raise HTTPException(status_code=422, detail="This annotation set has no annotations.")

    job = Job(
        organization_id=org_id,
        type=JobType.SAM3_AUTO_ANNOTATE,
        status=JobStatus.QUEUED,
        config={"tier": "sam3_refine_boxes", "request": payload.model_dump(mode="json")},
        input_refs=[{"type": "annotation_set", "id": str(src.id)}],
        created_by_user_id=current_user.id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    from geoops.refine_tasks import run_refine_boxes_to_masks  # noqa: PLC0415

    run_refine_boxes_to_masks.apply_async(args=[str(job.id)])
    return job


@router.get("/segment-all/{job_id}/masks", response_model=Sam3SegmentAllMasks)
async def segment_all_masks(
    job_id: UUID,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """The cleaned masks found so far (all finished scales)."""
    await _segment_all_job(db, org_id, job_id)
    try:
        data = await asyncio.to_thread(storage_service.get_object_json, org_id, masks_key(job_id))
    except Exception:  # noqa: BLE001 — no scale has finished yet
        return Sam3SegmentAllMasks(job_id=job_id, scales_done=[], masks=[])
    return Sam3SegmentAllMasks(job_id=job_id, scales_done=data.get("scales_done", []), masks=data.get("masks", []))


@router.post("/segment-all/{job_id}/cancel", response_model=JobRead)
async def segment_all_cancel(
    job_id: UUID,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    job = await _segment_all_job(db, org_id, job_id)
    if job.status in (JobStatus.QUEUED, JobStatus.PENDING, JobStatus.RUNNING):
        job.status = JobStatus.CANCELLED
        await db.commit()
        await db.refresh(job)
    return job


# ── Grouping masks by look (Interactive annotation mode) ─────────────────────

_EMB_CACHE: OrderedDict[str, tuple[list[str], np.ndarray]] = OrderedDict()
_THUMB_CACHE: OrderedDict[str, dict[str, str]] = OrderedDict()


async def _embed_job(db: AsyncSession, org_id: UUID, job_id: UUID, *, need_done: bool = True) -> Job:
    job = await db.get(Job, job_id)
    if job is None or job.organization_id != org_id or (job.config or {}).get("tier") != EMBED_TIER:
        raise HTTPException(status_code=404, detail="Embedding job not found")
    if need_done and job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Embeddings are {job.status}, not ready yet.")
    return job


def _load_embeddings(org_id: UUID, job_id: UUID) -> tuple[list[str], np.ndarray]:
    key = str(job_id)
    if key in _EMB_CACHE:
        _EMB_CACHE.move_to_end(key)
        return _EMB_CACHE[key]
    prefix = embeddings_prefix(job_id)
    ids = storage_service.get_object_json(org_id, f"{prefix}/ids.json")
    arr = np.load(io.BytesIO(storage_service.get_object_bytes(org_id, f"{prefix}/embeddings.npy")))
    _EMB_CACHE[key] = (ids, normalize(arr))
    while len(_EMB_CACHE) > 4:
        _EMB_CACHE.popitem(last=False)
    return _EMB_CACHE[key]


@router.post("/mask-embeddings", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def mask_embeddings(
    payload: Sam3MaskEmbedRequest,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Embed proposed masks with DINOv2 (async job) so they can be clustered and searched by similarity."""
    if len(payload.masks) > MAX_EMBED_MASKS:
        raise HTTPException(status_code=422, detail=f"At most {MAX_EMBED_MASKS} masks can be grouped at once.")
    item = await db.scalar(
        select(DatasetItem).where(
            DatasetItem.dataset_id == payload.dataset_id,
            DatasetItem.stac_item_id == payload.stac_item_id,
            DatasetItem.organization_id == org_id,
            DatasetItem.is_active.is_(True),
        )
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Imagery item not found")
    job = Job(
        organization_id=org_id,
        type=JobType.SAM3_AUTO_ANNOTATE,
        status=JobStatus.QUEUED,
        config={"tier": EMBED_TIER, "request": payload.model_dump(mode="json")},
        input_refs=[{"type": "dataset_item", "id": str(item.id)}],
        created_by_user_id=current_user.id,
        total_items=len(payload.masks),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    from geoops.interactive_tasks import run_mask_embeddings  # noqa: PLC0415

    run_mask_embeddings.apply_async(args=[str(job.id)])
    return job


@router.post("/mask-embeddings/{job_id}/clusters", response_model=Sam3ClusterResponse)
async def mask_clusters(
    job_id: UUID,
    payload: Sam3ClusterRequest,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Group the embedded masks into ``n_clusters`` looks (hierarchical; re-cutting is instant)."""
    await _embed_job(db, org_id, job_id)

    def run():
        ids, xn = _load_embeddings(org_id, job_id)
        if payload.mask_ids is None:
            return group(xn, ids, payload.n_clusters, cache_key=str(job_id))
        pos = {mid: i for i, mid in enumerate(ids)}
        keep = sorted({pos[m] for m in payload.mask_ids if m in pos})
        if len(keep) < 2:
            raise HTTPException(status_code=422, detail="Select at least two embedded masks to split.")
        sub_key = f"{job_id}:{hash(tuple(keep))}"
        return group(xn[keep], [ids[i] for i in keep], payload.n_clusters, cache_key=sub_key)

    return Sam3ClusterResponse(job_id=job_id, **await asyncio.to_thread(run))


@router.post("/mask-embeddings/{job_id}/similar", response_model=Sam3SimilarResponse)
async def mask_similar(
    job_id: UUID,
    payload: Sam3SimilarRequest,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Cosine similarity of every embedded mask to the mean of the query masks (1 = identical look)."""
    await _embed_job(db, org_id, job_id)

    def run():
        ids, xn = _load_embeddings(org_id, job_id)
        pos = {mid: i for i, mid in enumerate(ids)}
        q = [pos[m] for m in payload.query_ids if m in pos]
        if not q:
            raise HTTPException(status_code=422, detail="None of those masks were embedded.")
        sims = similarity(xn, q)
        return {mid: round(float(s), 4) for mid, s in zip(ids, sims, strict=True)}

    return Sam3SimilarResponse(job_id=job_id, similarity=await asyncio.to_thread(run))


@router.post("/mask-embeddings/{job_id}/thumbs", response_model=Sam3ThumbsResponse)
async def mask_thumbs(
    job_id: UUID,
    payload: Sam3ThumbsRequest,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Small JPEG thumbnails (base64) of masks, for the group gallery."""
    await _embed_job(db, org_id, job_id)

    def run():
        key = str(job_id)
        if key not in _THUMB_CACHE:
            _THUMB_CACHE[key] = storage_service.get_object_json(org_id, f"{embeddings_prefix(job_id)}/thumbs.json")
            while len(_THUMB_CACHE) > 4:
                _THUMB_CACHE.popitem(last=False)
        all_thumbs = _THUMB_CACHE[key]
        return {i: all_thumbs[i] for i in payload.ids if i in all_thumbs}

    return Sam3ThumbsResponse(thumbs=await asyncio.to_thread(run))


MAX_EXISTING = 20000


async def _owned_set(db: AsyncSession, org_id: UUID, set_id: UUID) -> AnnotationSet:
    aset = await db.scalar(select(AnnotationSet).where(
        AnnotationSet.id == set_id, AnnotationSet.deleted_at.is_(None), _set_in_org_clause(org_id)))
    if aset is None:
        raise HTTPException(status_code=404, detail="Annotation set not found")
    return aset


async def _owned_aoi(db: AsyncSession, org_id: UUID, map_id: UUID, aoi_id: UUID) -> MapAOI:
    aoi = await db.scalar(select(MapAOI).where(
        MapAOI.id == aoi_id, MapAOI.map_id == map_id, MapAOI.organization_id == org_id, MapAOI.deleted_at.is_(None)))
    if aoi is None:
        raise HTTPException(status_code=404, detail="AOI not found on this map")
    return aoi


# ── Commit an Interactive annotation draft ───────────────────────────────────


@router.post("/interactive-commit", response_model=Sam3CommitResponse, status_code=status.HTTP_201_CREATED)
async def interactive_commit(
    payload: Sam3CommitRequest,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Save the masks the user labelled and cleaned as a **verified** annotation set.

    Everything left in the draft after cleaning is treated as verified: the set is
    created with ``review_status="verified"``, each annotation is stamped with the AOI
    it was made in (so Train Head can use it), the set is grouped into its schema's
    collection, linked to the map's project and mounted on the map. Each annotation's
    geometry is the mask and its ``properties.bbox`` is the bounding box. All or nothing:
    one transaction.
    """
    from shapely.geometry import mapping, shape  # noqa: PLC0415

    from app.services.annotation_set_grouping import ensure_schema_collection_async  # noqa: PLC0415

    map_row = await db.get(Map, payload.map_id)
    project = await db.get(Project, map_row.project_id) if map_row else None
    if map_row is None or project is None or project.organization_id != org_id:
        raise HTTPException(status_code=404, detail="Map not found")
    aoi = await db.scalar(
        select(MapAOI).where(
            MapAOI.id == payload.aoi_id, MapAOI.map_id == payload.map_id,
            MapAOI.organization_id == org_id, MapAOI.deleted_at.is_(None),
        )
    )
    if aoi is None:
        raise HTTPException(status_code=404, detail="AOI not found on this map")
    item = await db.scalar(
        select(DatasetItem).where(
            DatasetItem.dataset_id == payload.dataset_id, DatasetItem.stac_item_id == payload.stac_item_id,
            DatasetItem.organization_id == org_id, DatasetItem.is_active.is_(True),
        )
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Imagery item not found")
    schema = await db.scalar(
        select(AnnotationSchema).where(AnnotationSchema.id == payload.schema_id, AnnotationSchema.organization_id == org_id)
    )
    if schema is None:
        raise HTTPException(status_code=404, detail="Annotation schema not found")

    class_ids = {m.class_id for m in payload.masks}
    valid = set(
        (await db.scalars(select(AnnotationClass.id).where(AnnotationClass.schema_id == schema.id, AnnotationClass.id.in_(class_ids)))).all()
    )
    if valid != class_ids:
        raise HTTPException(status_code=422, detail="Some masks are labelled with classes that aren't in the chosen schema.")

    polygons = []
    for i, m in enumerate(payload.masks):
        try:
            geom = shape(m.geometry)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
                raise ValueError("not a polygon")
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"Mask {i + 1} doesn't have a valid polygon outline.") from None
        polygons.append(geom)

    def new_set(name: str, description: str) -> AnnotationSet:
        return AnnotationSet(
            organization_id=org_id, schema_id=schema.id, dataset_id=payload.dataset_id, dataset_item_id=item.id,
            source_type="manual", name=name, description=description, review_status="verified",
            created_by_user_id=current_user.id,
        )

    def stamp(geom, m) -> dict:
        return {
            "aoi_id": str(aoi.id), "aoi_name": aoi.name, "review_status": "verified",
            "source": "interactive_annotation", "bbox": [round(v, 9) for v in geom.bounds],
            **({"scale_m": m.scale_m} if m.scale_m else {}),
        }

    main = new_set(payload.name, f"Interactive annotation in {aoi.name}")
    db.add(main)
    await db.flush()
    await ensure_schema_collection_async(db, main)
    db.add(ProjectAnnotationSet(project_id=map_row.project_id, annotation_set_id=main.id, linked_by=current_user.id))
    db.add(MapAnnotationSet(map_id=map_row.id, annotation_set_id=main.id, visible=True, opacity=1.0, z_index=0))

    # One annotation holds both forms of the object: the mask is its geometry, and its
    # bounding box rides along in properties as a standard GeoJSON ``bbox``.
    rows = [
        Annotation(
            annotation_set_id=main.id, class_id=m.class_id, dataset_item_id=item.id, confidence=m.score,
            geometry=parse_geometry(mapping(geom)), properties=stamp(geom, m), created_by_user_id=current_user.id,
        )
        for m, geom in zip(payload.masks, polygons, strict=True)
    ]
    db.add_all(rows)
    main_id = main.id  # read before the commit expires the objects
    await db.commit()

    counts: dict[str, int] = {}
    for m in payload.masks:
        counts[str(m.class_id)] = counts.get(str(m.class_id), 0) + 1
    return Sam3CommitResponse(
        annotation_set_id=main_id, annotation_count=len(payload.masks), class_counts=counts,
    )


@router.get("/annotation-sets/{set_id}/in-aoi", response_model=Sam3ExistingResponse)
async def existing_in_aoi(
    set_id: UUID,
    map_id: UUID,
    aoi_id: UUID,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Annotations of an existing set that intersect the AOI, for cleaning in Interactive mode."""
    import json  # noqa: PLC0415

    from shapely.geometry import box, mapping, shape  # noqa: PLC0415

    from app.core.geometry import serialize_geometry  # noqa: PLC0415

    aset = await _owned_set(db, org_id, set_id)
    aoi = await _owned_aoi(db, org_id, map_id, aoi_id)
    area = shape(aoi.geometry) if aoi.geometry else box(*aoi.bbox_4326)
    area_geojson = json.dumps(mapping(area))
    hit = func.ST_Intersects(Annotation.geometry, func.ST_SetSRID(func.ST_GeomFromGeoJSON(area_geojson), 4326))
    rows = (
        await db.scalars(
            select(Annotation).where(
                Annotation.annotation_set_id == set_id, Annotation.deleted_at.is_(None), hit,
            ).order_by(Annotation.created_at).limit(MAX_EXISTING + 1)
        )
    ).all()
    if len(rows) > MAX_EXISTING:
        raise HTTPException(status_code=422, detail=f"This set has more than {MAX_EXISTING} annotations in the AOI. Use a smaller AOI.")
    return Sam3ExistingResponse(
        schema_id=aset.schema_id, dataset_id=aset.dataset_id, total=len(rows),
        annotations=[
            Sam3ExistingAnnotation(
                id=r.id, class_id=r.class_id, geometry=serialize_geometry(r.geometry),
                confidence=r.confidence, properties=r.properties,
            )
            for r in rows
        ],
    )


@router.post("/annotation-sets/{set_id}/apply", response_model=Sam3ApplyResponse)
async def apply_to_set(
    set_id: UUID,
    payload: Sam3ApplyRequest,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Add, update and soft-delete annotations of an existing set in one transaction.

    Inserts are one ``add_all``; deletes are a single UPDATE scoped to the set. Updated
    geometries get their ``properties.bbox`` recomputed; annotations the user did not
    change are left alone. Verified sets stay verified; a raw model set becomes corrected.
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    from shapely.geometry import mapping, shape  # noqa: PLC0415

    from app.services.annotation_set_service import AnnotationSetService  # noqa: PLC0415

    aset = await _owned_set(db, org_id, set_id)
    aoi = await _owned_aoi(db, org_id, payload.map_id, payload.aoi_id)
    if aset.schema_id is None:
        raise HTTPException(status_code=422, detail="This set has no schema, so it can't be edited here.")
    item = await db.scalar(
        select(DatasetItem).where(
            DatasetItem.dataset_id == payload.dataset_id, DatasetItem.stac_item_id == payload.stac_item_id,
            DatasetItem.organization_id == org_id, DatasetItem.is_active.is_(True),
        )
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Imagery item not found")

    wanted = {m.class_id for m in payload.add} | {u.class_id for u in payload.update if u.class_id}
    if wanted:
        valid = set((await db.scalars(select(AnnotationClass.id).where(AnnotationClass.schema_id == aset.schema_id, AnnotationClass.id.in_(wanted)))).all())
        if valid != wanted:
            raise HTTPException(status_code=422, detail="Some classes aren't in this set's schema.")

    def polygon(geojson: dict, label: str):
        try:
            geom = shape(geojson)
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
                raise ValueError("not a polygon")
            return geom
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"{label} doesn't have a valid polygon outline.") from None

    stamp = {"aoi_id": str(aoi.id), "aoi_name": aoi.name, "review_status": "verified", "source": "interactive_annotation"}
    new_rows = []
    for i, m in enumerate(payload.add):
        geom = polygon(m.geometry, f"New mask {i + 1}")
        new_rows.append(Annotation(
            annotation_set_id=set_id, class_id=m.class_id, dataset_item_id=item.id, confidence=m.score,
            geometry=parse_geometry(mapping(geom)),
            properties={**stamp, "bbox": [round(v, 9) for v in geom.bounds], **({"scale_m": m.scale_m} if m.scale_m else {})},
            created_by_user_id=current_user.id,
        ))
    db.add_all(new_rows)

    # Only ids that really belong to this set can be touched.
    touched = {u.id for u in payload.update} | set(payload.delete_ids)
    owned = {}
    if touched:
        found = (await db.scalars(select(Annotation).where(
            Annotation.annotation_set_id == set_id, Annotation.deleted_at.is_(None), Annotation.id.in_(touched),
        ))).all()
        owned = {a.id: a for a in found}
    if touched - owned.keys():
        raise HTTPException(status_code=409, detail="Some annotations were already removed from this set. Reload and try again.")

    for u in payload.update:
        a = owned[u.id]
        if u.class_id:
            a.class_id = u.class_id
        props = dict(a.properties or {})
        if u.geometry is not None:
            geom = polygon(u.geometry, "An edited mask")
            a.geometry = parse_geometry(mapping(geom))
            props["bbox"] = [round(v, 9) for v in geom.bounds]
        a.properties = {**props, "review_status": "verified"}

    deleted = 0
    if payload.delete_ids:
        res = await db.execute(
            sa_update(Annotation)
            .where(Annotation.annotation_set_id == set_id, Annotation.deleted_at.is_(None), Annotation.id.in_(payload.delete_ids))
            .values(deleted_at=datetime.now(UTC).replace(tzinfo=None))
        )
        deleted = res.rowcount or 0

    AnnotationSetService(db).mark_corrected_if_model(aset)
    aset.updated_at = datetime.now(UTC)  # annotations changed: this is what "last modified" should mean
    status_now = aset.review_status
    await db.commit()
    return Sam3ApplyResponse(
        annotation_set_id=set_id, added=len(new_rows), updated=len(payload.update), deleted=deleted, review_status=status_now,
    )


@router.post("/mask-crops", response_model=Sam3ThumbsResponse)
async def mask_crops(
    payload: Sam3CropRequest,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Thumbnails of the imagery around given masks (the mask outlined). Works for any mask —
    unlike the grouping thumbnails, nothing has to have been embedded first."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    from geoops.export_common import resolve_rescale_params  # noqa: PLC0415
    from geoops.interactive_tasks import render_thumb  # noqa: PLC0415

    item = await db.scalar(
        select(DatasetItem).where(
            DatasetItem.dataset_id == payload.dataset_id, DatasetItem.stac_item_id == payload.stac_item_id,
            DatasetItem.organization_id == org_id, DatasetItem.is_active.is_(True),
        )
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Imagery item not found")

    def run() -> dict[str, str]:
        rescale = resolve_rescale_params(item.stac_collection_id, item.stac_item_id).get("rescale")

        def one(m):
            try:
                return m.id, render_thumb(item, m.geometry, rescale)
            except Exception:  # noqa: BLE001 — a mask that can't be rendered just has no thumbnail
                return m.id, None

        with ThreadPoolExecutor(max_workers=8) as pool:
            return {i: t for i, t in pool.map(one, payload.masks) if t}

    return Sam3ThumbsResponse(thumbs=await asyncio.to_thread(run))
