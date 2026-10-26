import asyncio
import io
import json
import logging
import re
import tempfile
import zipfile
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_session, require_org_role
from app.core.enums import JobStatus, JobType
from app.core.exceptions import bad_request, not_found
from app.models.job import Job
from app.models.user import User
from app.schemas.job import JobRead
from app.services import storage_service

from geoops.dino_schemas import (
    DinoAoiSetRead,
    DinoAoiSetsRequest,
    DinoDatasetCreateRequest,
    DinoDatasetItemsResponse,
    DinoDatasetRead,
    DinoDatasetRebuildRequest,
    DinoDatasetRenameRequest,
    DinoExportPreviewRequest,
    DinoExportPreviewResponse,
    DinoFinetuneRequest,
    DinoRunReport,
    DinoTunedConfigRead,
)
from geoops.dino_service import DinoExportService
from geoops.models import DinoDataset, DinoTunedConfig

logger = logging.getLogger(__name__)

# Cap on tiles read back for the dataset browser — this tier's datasets are
# few-shot by design (tens to low hundreds of tiles), so this is a safety
# bound against a runaway read, not a real-world limit.
MAX_DATASET_ITEMS = 400

router = APIRouter(prefix="/dino", tags=["dino"])


@router.post("/export-preview", response_model=DinoExportPreviewResponse)
async def export_preview(
    payload: DinoExportPreviewRequest,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Validate + preview the training corpus for ONE class-specific head.

    Scoped to a single class_id, unlike /yolo/export-preview which is
    collection-wide — the DINOv2 tier trains one head per class.
    """
    service = DinoExportService(db)
    return await service.build_preview(
        org_id, payload.annotation_set_collection_id, payload.class_id, payload.aoi_id,
        aoi_ids=payload.aoi_ids, set_ids=payload.annotation_set_ids, task=payload.task, class_ids=payload.class_ids,
    )


@router.post("/aoi-annotation-sets", response_model=list[DinoAoiSetRead])
async def aoi_annotation_sets(
    payload: DinoAoiSetsRequest,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Annotation sets (same schema as the collection) that have annotations inside the given AOIs,
    found by PostGIS intersection — the choices for training from several sets / AOIs."""
    service = DinoExportService(db)
    collection = await service._get_collection(org_id, payload.annotation_set_collection_id)
    return await service.sets_in_aois(org_id, collection.schema_id, payload.aoi_ids, payload.class_id, payload.class_ids)


@router.get("/runs", response_model=list[JobRead])
async def list_runs(
    annotation_set_collection_id: UUID,
    class_id: UUID,
    aoi_id: UUID | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Past Train Head runs for one class, newest first — powers the
    "previous runs" report list in the Train Head advanced settings popup.
    Each run's ``config.result`` is a report (dataset reference, loss
    curves, metrics, timings); the tiles live on the dataset, not the run.

    Matches on the request payload stored in ``job.config["request"]``
    (JSONB) rather than a dedicated FK column, since jobs are generic and
    the dino tier's request shape lives entirely in that blob. When
    ``aoi_id`` is given, only runs that were themselves scoped to that AOI
    are returned — a run with no aoi_id in its own request (unrestricted,
    or predates the AOI-restriction feature) won't match.
    """
    query = (
        select(Job)
        .where(
            Job.organization_id == org_id,
            Job.type == JobType.FINETUNE_MODEL,
            Job.config["tier"].astext == "dino_head",
            Job.config["request"]["annotation_set_collection_id"].astext == str(annotation_set_collection_id),
            or_(
                Job.config["request"]["class_id"].astext == str(class_id),
                Job.config["request"]["class_ids"].contains([str(class_id)]),
            ),
        )
        .order_by(Job.created_at.desc())
        .limit(limit)
    )
    if aoi_id is not None:
        query = query.where(Job.config["request"]["aoi_id"].astext == str(aoi_id))
    jobs = (await db.scalars(query)).all()
    return list(jobs)


# ── Datasets ─────────────────────────────────────────────────────────────────


def _with_scales(params: dict, preview) -> dict:
    """Make the build's scale list explicit: the caller's ``scales_m``, else
    their single ``tile_scale_m``, else the scales suggested from the object
    sizes (several when the sizes spread widely). Stored on the dataset so a
    rebuild reproduces exactly what was shown."""
    params = dict(params)
    if not params.get("scales_m"):
        single = params.get("tile_scale_m")
        params["scales_m"] = [single] if single else (preview.suggested_scales_m or [10.0])
    params["tile_scale_m"] = None  # derived from scales_m at build time (primary scale)
    return params


async def _get_dataset(db: AsyncSession, org_id: UUID, dataset_id: UUID) -> DinoDataset:
    ds = await db.get(DinoDataset, dataset_id)
    if ds is None or ds.organization_id != org_id:
        raise not_found("Dataset")
    return ds


async def _to_read(db: AsyncSession, service: DinoExportService, ds: DinoDataset) -> DinoDatasetRead:
    """Serialise a dataset and work out whether it is stale. Staleness only
    applies to a successfully built dataset: its annotation data hashes to
    something other than what the build stored, or (for a clone) its source
    dataset has been rebuilt since."""
    read = DinoDatasetRead.model_validate(ds)
    if ds.status != "ready":
        return read
    if ds.source_dataset_id:
        src = await db.get(DinoDataset, ds.source_dataset_id)
        if src is not None and src.build_no != ds.source_build_no:
            read.is_stale, read.stale_reason = True, "The source dataset was rebuilt after this one was created."
            return read
    try:
        current = await service.data_fingerprint(
            ds.organization_id, ds.annotation_set_collection_id, ds.class_id, ds.aoi_id, ds.task,
            aoi_ids=[UUID(a) for a in ds.aoi_ids] if ds.annotation_set_ids and ds.aoi_ids else None,
            set_ids=[UUID(i) for i in ds.annotation_set_ids] if ds.annotation_set_ids else None,
            class_ids=[UUID(c) for c in ds.class_ids] if ds.class_ids else None,
        )
    except Exception:  # noqa: BLE001 — a deleted AOI/collection also means the data changed
        read.is_stale, read.stale_reason = True, "The AOI or collection this dataset was built from changed."
        return read
    if current != ds.data_fingerprint:
        read.is_stale, read.stale_reason = True, "Annotations changed since this dataset was built."
    return read


async def _enqueue_build(
    db: AsyncSession, ds: DinoDataset, org_id: UUID, user: User, params: dict, fresh: bool = False
) -> DinoDataset:
    if not fresh and ds.status in ("pending", "building"):
        raise bad_request("This dataset is already being built.")
    ds.params = params
    ds.status = "pending"
    ds.error = None
    job = Job(
        organization_id=org_id,
        type=JobType.FINETUNE_MODEL,
        status=JobStatus.QUEUED,
        config={"tier": "dino_dataset_build", "dataset_id": str(ds.id)},
        input_refs=[
            {"type": "annotation_set_collection", "id": str(ds.annotation_set_collection_id)},
            {"type": "annotation_class", "id": str(ds.class_id)},
            *[{"type": "map_aoi", "id": a} for a in (ds.aoi_ids or [str(ds.aoi_id)])],
        ],
        created_by_user_id=user.id,
    )
    db.add(job)
    await db.flush()
    ds.job_id = job.id
    await db.commit()
    await db.refresh(ds)

    from geoops.dino_dataset_tasks import run_dino_dataset_build  # noqa: PLC0415

    run_dino_dataset_build.apply_async(args=[str(job.id)])
    return ds


@router.post("/datasets", response_model=DinoDatasetRead, status_code=status.HTTP_202_ACCEPTED)
async def create_dataset(
    payload: DinoDatasetCreateRequest,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Create a new dataset and build it (async). Training never builds one
    implicitly. Review the result with ``GET /dino/datasets/{id}/items``,
    rebuild it with ``POST /dino/datasets/{id}/rebuild``, then train.

    Pass ``source_dataset_id`` (a ready detect dataset) with ``task="classify"``
    to derive a classifier dataset from the same tiles; omit it to build an
    independent one."""
    service = DinoExportService(db)
    preview = await service.build_preview(
        org_id, payload.annotation_set_collection_id, payload.class_id, payload.aoi_id,
        aoi_ids=payload.aoi_ids, set_ids=payload.annotation_set_ids, task=payload.task, class_ids=payload.class_ids,
    )
    if not preview.ready:
        raise bad_request(
            "Nothing eligible to build: 0 annotations of this class in the chosen sets and AOIs."
            if payload.annotation_set_ids
            else "Nothing eligible to build: 0 verified annotations of this class tagged to this AOI."
        )
    aoi = await service.get_aoi(org_id, payload.aoi_id)
    await service.get_aois(org_id, payload.aoi_ids or [])  # every chosen AOI must exist in this org
    multi = bool(payload.annotation_set_ids)

    params = payload.params.model_dump()
    if payload.source_dataset_id:
        src = await _get_dataset(db, org_id, payload.source_dataset_id)
        if (
            src.task != "detect"
            or src.status != "ready"
            or (src.annotation_set_collection_id, src.class_id, src.aoi_id)
            != (payload.annotation_set_collection_id, payload.class_id, payload.aoi_id)
            or (src.annotation_set_ids or None) != ([str(i) for i in payload.annotation_set_ids] if multi else None)
        ):
            raise bad_request("The source must be a ready detect dataset for the same collection, class and AOI.")
        params = dict(src.params)
    else:
        params = _with_scales(params, preview)

    ds = DinoDataset(
        organization_id=org_id,
        name=payload.name or f"{preview.class_name} · {aoi.name}{f' +{len(payload.aoi_ids) - 1}' if len(payload.aoi_ids or []) > 1 else ''} · {payload.task}",
        annotation_set_collection_id=payload.annotation_set_collection_id,
        class_id=payload.class_id,
        class_ids=[str(c) for c in payload.class_ids] if payload.class_ids else None,
        aoi_id=payload.aoi_id,
        aoi_ids=[str(a) for a in payload.aoi_ids] if multi else None,
        annotation_set_ids=[str(i) for i in payload.annotation_set_ids] if multi else None,
        task=payload.task,
        params=params,
        status="pending",
        source_dataset_id=payload.source_dataset_id,
        created_by=current_user.id,
    )
    db.add(ds)
    await db.flush()
    ds = await _enqueue_build(db, ds, org_id, current_user, params, fresh=True)
    return await _to_read(db, service, ds)


@router.post("/datasets/{dataset_id}/rebuild", response_model=DinoDatasetRead, status_code=status.HTTP_202_ACCEPTED)
async def rebuild_dataset(
    dataset_id: UUID,
    payload: DinoDatasetRebuildRequest,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Regenerate this dataset — after new annotations, or with a new tile
    scale / crop size / split. Writes a fresh build folder and overwrites only
    this dataset. A dataset cloned from another keeps the source's params, so
    its params can't be changed here."""
    ds = await _get_dataset(db, org_id, dataset_id)
    if ds.source_dataset_id:
        params = dict(ds.params)
        if payload.params is not None:
            raise bad_request("A cloned dataset inherits its source's params — rebuild the source instead.")
    else:
        params = payload.params.model_dump() if payload.params is not None else dict(ds.params)
        if not params.get("scales_m"):
            service = DinoExportService(db)
            preview = await service.build_preview(
                org_id, ds.annotation_set_collection_id, ds.class_id, ds.aoi_id,
                aoi_ids=[UUID(a) for a in ds.aoi_ids] if ds.aoi_ids else None,
                set_ids=[UUID(i) for i in ds.annotation_set_ids] if ds.annotation_set_ids else None, task=ds.task,
                class_ids=[UUID(c) for c in ds.class_ids] if ds.class_ids else None,
            )
            params = _with_scales(params, preview)
    ds = await _enqueue_build(db, ds, org_id, current_user, params)
    return await _to_read(db, DinoExportService(db), ds)


@router.patch("/datasets/{dataset_id}", response_model=DinoDatasetRead)
async def rename_dataset(
    dataset_id: UUID,
    payload: DinoDatasetRenameRequest,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Rename a dataset (nothing else about it changes)."""
    ds = await _get_dataset(db, org_id, dataset_id)
    ds.name = payload.name.strip()
    await db.commit()
    await db.refresh(ds)
    return await _to_read(db, DinoExportService(db), ds)


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: UUID,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Delete a dataset and its tile files. Past training runs keep their reports.
    A detector dataset that classifier datasets were cloned from can't be deleted
    first: the clones read their tile images from it."""
    ds = await _get_dataset(db, org_id, dataset_id)
    if ds.status in ("pending", "building"):
        raise bad_request("This dataset is being generated. Wait for it to finish.")
    clones = (await db.scalars(select(DinoDataset.name).where(DinoDataset.source_dataset_id == ds.id))).all()
    if clones:
        raise bad_request(f"Delete the classifier splits made from this one first: {', '.join(clones)}.")
    await db.delete(ds)
    await db.commit()
    try:
        # Data only: trained models live in their own folders (dino-models/<run>/) and are not touched.
        await asyncio.to_thread(storage_service.delete_objects_by_prefix, org_id, f"dino-datasets/{dataset_id}/")
    except Exception:  # noqa: BLE001 — the row is gone; stray files are harmless
        logger.warning("dino_dataset_files_delete_failed dataset=%s", dataset_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/datasets", response_model=list[DinoDatasetRead])
async def list_datasets(
    annotation_set_collection_id: UUID,
    class_id: UUID,
    aoi_id: UUID | None = None,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    query = select(DinoDataset).where(
        DinoDataset.organization_id == org_id,
        DinoDataset.annotation_set_collection_id == annotation_set_collection_id,
        or_(DinoDataset.class_id == class_id, DinoDataset.class_ids.contains([str(class_id)])),
    )
    if aoi_id is not None:
        # Datasets covering several AOIs show up under each of them.
        query = query.where(or_(DinoDataset.aoi_id == aoi_id, DinoDataset.aoi_ids.contains([str(aoi_id)])))
    datasets = (await db.scalars(query.order_by(DinoDataset.created_at.desc()))).all()
    service = DinoExportService(db)
    return [await _to_read(db, service, ds) for ds in datasets]


@router.get("/datasets/{dataset_id}", response_model=DinoDatasetRead)
async def get_dataset(
    dataset_id: UUID,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    ds = await _get_dataset(db, org_id, dataset_id)
    return await _to_read(db, DinoExportService(db), ds)


# ── Auto-tuned configs and multi-run reports ─────────────────────────────────


def _tuned_read(row: DinoTunedConfig, ds: DinoDataset) -> DinoTunedConfigRead:
    return DinoTunedConfigRead(
        id=row.id, dataset_id=row.dataset_id, dataset_build_no=row.dataset_build_no, model_key=row.model_key,
        best_params=row.best_params, tuning=row.tuning, multirun=row.multirun, job_id=row.job_id,
        is_stale=row.dataset_build_no != ds.build_no, created_at=row.created_at, updated_at=row.updated_at,
    )


@router.get("/datasets/{dataset_id}/tuned-configs", response_model=list[DinoTunedConfigRead])
async def list_tuned_configs(
    dataset_id: UUID,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Every saved auto-tune result of a dataset (one per model: backbone or YOLO base weights)."""
    ds = await _get_dataset(db, org_id, dataset_id)
    rows = await db.scalars(
        select(DinoTunedConfig).where(DinoTunedConfig.dataset_id == ds.id).order_by(DinoTunedConfig.updated_at.desc())
    )
    return [_tuned_read(r, ds) for r in rows.all()]


async def _get_tuned(db: AsyncSession, ds: DinoDataset, model_key: str) -> DinoTunedConfig:
    row = await db.scalar(
        select(DinoTunedConfig).where(DinoTunedConfig.dataset_id == ds.id, DinoTunedConfig.model_key == model_key)
    )
    if row is None:
        raise not_found("Tuned config (this dataset + model has not been auto-tuned yet)")
    return row


@router.get("/datasets/{dataset_id}/tuned-config", response_model=DinoTunedConfigRead)
async def get_tuned_config(
    dataset_id: UUID,
    model_key: str = Query(description="The DINOv2 backbone key, or for YOLO the base weights the run starts from."),
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """The final best config (+ every trial, and the latest multi-run mean ± std report) for a dataset + model."""
    ds = await _get_dataset(db, org_id, dataset_id)
    return _tuned_read(await _get_tuned(db, ds, model_key), ds)


@router.delete("/datasets/{dataset_id}/tuned-config", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tuned_config(
    dataset_id: UUID,
    model_key: str = Query(),
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Forget the saved config so the next auto-tuned run searches again."""
    ds = await _get_dataset(db, org_id, dataset_id)
    await db.delete(await _get_tuned(db, ds, model_key))
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/runs/{job_id}/report", response_model=DinoRunReport)
async def run_report(
    job_id: UUID,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """The tuning summary and multi-run report (mean ± std, per-seed values) of one finished training run."""
    job = await db.get(Job, job_id)
    if job is None or job.organization_id != org_id or (job.config or {}).get("tier") != "dino_head":
        raise not_found("Training run")
    result = (job.config or {}).get("result") or {}
    return DinoRunReport(
        job_id=job.id, dataset_id=result.get("dataset_id"), task=result.get("task"),
        hyperparameters=result.get("hyperparameters"), tuning=result.get("tuning"), multirun=result.get("multirun"),
    )


@router.get("/datasets/{dataset_id}/items", response_model=DinoDatasetItemsResponse)
async def get_dataset_items(
    dataset_id: UUID,
    split: Literal["train", "eval", "test"] | None = None,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Every tile of the dataset's current build, for the dataset browser.
    Reads manifest.json (+ one targets.json per tile for detect) from object
    storage; uncached, so only sensible for few-shot sizes (MAX_DATASET_ITEMS).

    Returns each tile's raw S3 key, not a presigned URL — MinIO's own port
    often isn't reachable from the browser (remote dev boxes, tunnels); the
    frontend fetches bytes from ``GET /dino/datasets/{id}/image`` instead.
    """
    ds = await _get_dataset(db, org_id, dataset_id)
    if ds.status != "ready" or not ds.s3_prefix:
        raise bad_request("Dataset has not been built yet.")

    try:
        manifest = await asyncio.to_thread(storage_service.get_object_json, org_id, f"{ds.s3_prefix}/manifest.json")
    except Exception as exc:  # noqa: BLE001
        raise not_found("Dataset manifest (its files may have been removed from storage)") from exc

    task = manifest.get("task", ds.task)
    manifest_items = [it for it in manifest.get("items", []) if split is None or it.get("split") == split]
    truncated = len(manifest_items) > MAX_DATASET_ITEMS
    manifest_items = manifest_items[:MAX_DATASET_ITEMS]

    async def _load_item(it: dict) -> dict:
        box_count = None
        boxes = None
        polygons = None
        if task in ("detect", "segment", "yolo") and it.get("targets"):
            try:
                doc = await asyncio.to_thread(storage_service.get_object_json, org_id, it["targets"])
                raw = doc.get("targets", [])
                box_count = len(raw)
                boxes = [t["bbox"] for t in raw if "bbox" in t] or None
                polygons = [t["polygon"] for t in raw if "polygon" in t] or None
            except Exception:  # noqa: BLE001
                logger.warning("dino_dataset_targets_read_failed dataset=%s key=%s", dataset_id, it.get("targets"))
        return {
            "item_id": it.get("item_id"),
            "split": it.get("split"),
            "image_key": it["image"],
            "scale_m": it.get("scale_m"),
            "box_count": box_count,
            "boxes": boxes,
            "polygons": polygons,
            "label": it.get("label"),
        }

    items = await asyncio.gather(*[_load_item(it) for it in manifest_items])
    return DinoDatasetItemsResponse(
        dataset_id=dataset_id, task=task, total=len(items), truncated=truncated, items=items
    )


def _build_coco_zip(org_id: UUID, ds: DinoDataset, manifest: dict) -> tuple[object, str]:
    """Write the dataset as a COCO zip into a temp file (sync; run in a thread).

    images/{train,val,test}/…png and annotations/instances_{train,val,test}.json.
    A detector dataset has one box per labelled object (COCO ``bbox`` = [x, y, w, h] in
    pixels). A classifier dataset has no boxes: each image carries ``label`` (1 = the class
    is present, 0 = not) instead. ``eval`` is written as ``val``, the COCO name for it.
    """
    from PIL import Image  # noqa: PLC0415

    class_name = manifest.get("class_name") or "object"
    categories = [{"id": c["index"] + 1, "name": c["name"]} for c in manifest.get("classes", [])] or [{"id": 1, "name": class_name}]
    task = manifest.get("task", ds.task)
    folder = {"train": "train", "eval": "val", "test": "test"}
    coco: dict[str, dict] = {
        sp: {
            "info": {"description": f"{ds.name} ({sp})", "task": task, "dataset_build": ds.build_no},
            "categories": categories,
            "images": [],
            "annotations": [],
        }
        for sp in folder
    }
    ids = {sp: {"image": 0, "ann": 0} for sp in folder}
    used: set[str] = set()
    tmp = tempfile.TemporaryFile()  # noqa: SIM115 — streamed and closed by the response
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for it in manifest.get("items", []):
            sp = it.get("split")
            if sp not in folder:
                continue
            data = storage_service.get_object_bytes(org_id, it["image"])
            width, height = Image.open(io.BytesIO(data)).size
            base = it["image"].rsplit("/", 1)[-1]
            name = base
            n = 1
            while f"{sp}/{name}" in used:  # file names are unique within a split, keep them so
                n += 1
                name = f"{n}_{base}"
            used.add(f"{sp}/{name}")
            zf.writestr(f"images/{folder[sp]}/{name}", data)
            ids[sp]["image"] += 1
            image = {"id": ids[sp]["image"], "file_name": f"{folder[sp]}/{name}", "width": width, "height": height}
            if it.get("scale_m") is not None:
                image["scale_m"] = it["scale_m"]
            if task in ("detect", "segment", "yolo") and it.get("targets"):
                targets = storage_service.get_object_json(org_id, it["targets"]).get("targets", [])
                for t in targets:
                    cx, cy, w, h = t["bbox"]
                    ids[sp]["ann"] += 1
                    coco[sp]["annotations"].append({
                        "id": ids[sp]["ann"], "image_id": ids[sp]["image"], "category_id": int(t.get("class", 0)) + 1,
                        "bbox": [round((cx - w / 2) * width, 2), round((cy - h / 2) * height, 2), round(w * width, 2), round(h * height, 2)],
                        "area": round(w * width * h * height, 2), "iscrowd": 0,
                        **({"segmentation": [[round(v * (width if k % 2 == 0 else height), 2) for k, v in enumerate(t["polygon"])]]} if t.get("polygon") else {}),
                    })
            elif task == "classify":
                image["label"] = int(it.get("label") or 0)
            coco[sp]["images"].append(image)
        for sp, doc in coco.items():
            zf.writestr(f"annotations/instances_{folder[sp]}.json", json.dumps(doc))
    tmp.seek(0)
    return tmp, re.sub(r"[^A-Za-z0-9._-]+", "_", ds.name).strip("_") or "dataset"


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(
    dataset_id: UUID,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """The dataset as a zip in COCO format (images + instances_{train,val,test}.json)."""
    ds = await _get_dataset(db, org_id, dataset_id)
    if ds.status != "ready" or not ds.s3_prefix:
        raise bad_request("Dataset has not been built yet.")
    try:
        manifest = await asyncio.to_thread(storage_service.get_object_json, org_id, f"{ds.s3_prefix}/manifest.json")
    except Exception as exc:  # noqa: BLE001
        raise not_found("Dataset manifest (its files may have been removed from storage)") from exc
    handle, filename = await asyncio.to_thread(_build_coco_zip, org_id, ds, manifest)

    def chunks():
        try:
            while block := handle.read(1024 * 1024):
                yield block
        finally:
            handle.close()

    return StreamingResponse(
        chunks(), media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}_coco.zip"'},
    )


# ── YOLO: what a training can start from ─────────────────────────────────────


@router.get("/yolo-bases")
async def yolo_bases(
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Starting points for a YOLO training: the pretrained weights files in the service's models/yolo folder, and
    this org's earlier YOLO models (start from one to add classes to it)."""

    def files() -> list[dict]:
        import json as _json  # noqa: PLC0415
        from urllib import request as _rq  # noqa: PLC0415

        from app.config import settings  # noqa: PLC0415

        req = _rq.Request(f"{settings.GEOOPS_SERVICE_URL.rstrip('/')}/yolo/bases", headers={"Authorization": f"Bearer {settings.GEOOPS_SERVICE_API_TOKEN}"})
        with _rq.urlopen(req, timeout=15) as resp:  # nosec B310
            return _json.loads(resp.read())

    try:
        pretrained = await asyncio.to_thread(files)
    except Exception:  # noqa: BLE001 — the service may be down
        pretrained = []
    jobs = (
        await db.scalars(
            select(Job).where(
                Job.organization_id == org_id, Job.type == JobType.FINETUNE_MODEL,
                Job.config["tier"].astext == "dino_head", Job.config["result"]["task"].astext == "yolo",
                Job.status == JobStatus.COMPLETED,
            ).order_by(Job.created_at.desc()).limit(20)
        )
    ).all()
    earlier = [
        {"id": j.config["result"]["train_result"]["artifact_key"], "name": f"{j.config['result'].get('class_name', 'YOLO model')} · {j.created_at:%d %b %H:%M}"}
        for j in jobs if ((j.config or {}).get("result") or {}).get("train_result", {}).get("artifact_key")
    ]
    return {"pretrained": pretrained, "earlier": earlier}


# ── A trained model: download, ONNX export ───────────────────────────────────


async def _run_model(db: AsyncSession, org_id: UUID, job_id: UUID, seed: int | None = None) -> tuple[Job, str]:
    """The run's kept model, or with ``seed`` the model of that seed run (multi-seed experiments keep every seed's file)."""
    job = await db.get(Job, job_id)
    if job is None or job.organization_id != org_id or (job.config or {}).get("tier") != "dino_head":
        raise not_found("Training run")
    result = (job.config or {}).get("result") or {}
    key = (result.get("train_result") or {}).get("artifact_key")
    if seed is not None:  # only a key this run recorded itself, never one from the client
        key = next((r.get("artifact_key") for r in (result.get("multirun") or {}).get("runs") or [] if r.get("seed") == seed), None)
        if not key:
            raise not_found("Seed run")
    if not key:
        raise bad_request("This run did not produce a model.")
    return job, key


def _onnx_key(artifact_key: str) -> str:
    return artifact_key.rsplit(".", 1)[0] + ".onnx"


async def _exports(org_id: UUID, artifact_key: str) -> dict:
    pt, onnx = await asyncio.gather(
        asyncio.to_thread(storage_service.object_size, org_id, artifact_key),
        asyncio.to_thread(storage_service.object_size, org_id, _onnx_key(artifact_key)),
    )
    return {"pt_size_bytes": pt, "onnx_size_bytes": onnx}


@router.get("/runs/{job_id}/model")
async def run_model_exports(
    job_id: UUID,
    seed: int | None = Query(default=None, ge=0, description="A seed run of a multi-seed experiment; default is the kept model."),
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Which files exist for a run's trained model (the PyTorch artifact, and the ONNX export once made)."""
    _job, key = await _run_model(db, org_id, job_id, seed)
    return await _exports(org_id, key)


@router.post("/runs/{job_id}/model/onnx")
async def export_run_model_onnx(
    job_id: UUID,
    seed: int | None = Query(default=None, ge=0),
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Convert the run's model to ONNX (backbone included), stored beside the artifact. Safe to repeat."""
    from geoops.dino_tasks import _call_geoops_service  # noqa: PLC0415


    job, key = await _run_model(db, org_id, job_id, seed)
    task = ((job.config or {}).get("result") or {}).get("task", "detect")
    try:
        await asyncio.to_thread(
            _call_geoops_service, "/export-onnx",
            {"org_id": str(org_id), "artifact_key": key, "kind": {"classify": "classify", "segment": "segment", "yolo": "yolo"}.get(task, "fcos"), "out_key": _onnx_key(key)},
            600.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("onnx_export_request_failed job=%s", job_id)
        raise bad_request("Could not convert the model to ONNX. Its trained file may no longer exist.") from exc
    return await _exports(org_id, key)


@router.delete("/runs/{job_id}/model", status_code=status.HTTP_204_NO_CONTENT)
async def delete_run_model(
    job_id: UUID,
    org_id: UUID = Depends(require_org_role("org:admin")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Delete a run's trained model: its files (.pt and .onnx) and its entry in the model list. The run's
    report stays, and the dataset it trained on is not touched."""
    from app.services.model_service import AIModelService  # noqa: PLC0415

    job, key = await _run_model(db, org_id, job_id)
    result = (job.config or {}).get("result") or {}
    ai_model_id = result.get("ai_model_id")
    if ai_model_id:
        try:
            await AIModelService(db).delete_model(UUID(ai_model_id), organization_id=org_id)
        except Exception:  # noqa: BLE001 — the model row may already be gone
            logger.warning("dino_model_row_delete_failed job=%s", job_id)
    # A multi-seed run keeps the winning seed's model under <run folder>/seeds/seed-N/; the run folder owns all seeds.
    folder = key.split("/seeds/")[0] if "/seeds/" in key else key.rsplit("/", 1)[0]
    if folder.startswith("dino-models/"):
        await asyncio.to_thread(storage_service.delete_objects_by_prefix, org_id, f"{folder}/")
    else:  # a run from before models had their own folder: remove just its files
        for k in (key, _onnx_key(key)):
            await asyncio.to_thread(storage_service.delete_object, org_id, k)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/runs/{job_id}/model/download")
async def download_run_model(
    job_id: UUID,
    format: Literal["pt", "onnx"] = "pt",
    seed: int | None = Query(default=None, ge=0),
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """The trained model file: ``pt`` (the trained head's weights; DINOv2 is loaded separately) or ``onnx``."""
    job, key = await _run_model(db, org_id, job_id, seed)
    target = key if format == "pt" else _onnx_key(key)
    try:
        data = await asyncio.to_thread(storage_service.get_object_bytes, org_id, target)
    except Exception as exc:  # noqa: BLE001
        raise not_found("Model file (convert it to ONNX first, or its files were removed)") from exc
    cls = ((job.config or {}).get("result") or {}).get("class_name") or "model"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", cls).strip("_") or "model"
    return Response(
        content=data, media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{name}_head{f"_seed{seed}" if seed is not None else ""}.{format}"'},
    )


@router.get("/datasets/{dataset_id}/image")
async def get_dataset_image(
    dataset_id: UUID,
    key: str,
    org_id: UUID = Depends(require_org_role("org:viewer")),
    db: AsyncSession = Depends(get_session),
    _current_user: User = Depends(get_current_user),
):
    """Stream one tile's PNG through the API (see get_dataset_items for why).
    ``key`` must live under this dataset's own folder, or its source
    dataset's when it is a clone that shares the source's tile images."""
    ds = await _get_dataset(db, org_id, dataset_id)
    allowed = [f"dino-datasets/{ds.id}/"]
    if ds.source_dataset_id:
        allowed.append(f"dino-datasets/{ds.source_dataset_id}/")
    if ".." in key or not any(key.startswith(p) for p in allowed):
        raise not_found("Image (not part of this dataset)")

    try:
        data = await asyncio.to_thread(storage_service.get_object_bytes, org_id, key)
    except Exception as exc:  # noqa: BLE001
        raise not_found("Image (removed from storage)") from exc
    return Response(content=data, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


# ── Training ─────────────────────────────────────────────────────────────────


@router.post("/finetune", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def finetune(
    payload: DinoFinetuneRequest,
    org_id: UUID = Depends(require_org_role("org:member")),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    """Train + evaluate a head from an already-built dataset (async job). The
    dataset is used exactly as built — nothing is regenerated. Poll
    ``GET /api/jobs/{job_id}`` for ``config.result``."""
    ds = await _get_dataset(db, org_id, payload.dataset_id)
    if ds.status != "ready":
        raise bad_request(f"Dataset is {ds.status} — build it (and check it) before training.")
    if not payload.allow_stale:
        read = await _to_read(db, DinoExportService(db), ds)
        if read.is_stale:
            raise bad_request(f"{read.stale_reason} Rebuild the dataset, or train anyway with allow_stale.")

    request = payload.model_dump(mode="json")
    request.update(
        annotation_set_collection_id=str(ds.annotation_set_collection_id),
        class_id=str(ds.class_id),
        class_ids=ds.class_ids,
        aoi_id=str(ds.aoi_id),
        aoi_ids=ds.aoi_ids,
        task=ds.task,
    )
    job = Job(
        organization_id=org_id,
        type=JobType.FINETUNE_MODEL,
        status=JobStatus.QUEUED,
        config={"request": request, "tier": "dino_head"},
        input_refs=[
            {"type": "dino_dataset", "id": str(ds.id)},
            {"type": "annotation_class", "id": str(ds.class_id)},
            {"type": "map_aoi", "id": str(ds.aoi_id)},
        ],
        created_by_user_id=current_user.id,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    from geoops.dino_tasks import run_dino_train  # noqa: PLC0415

    run_dino_train.apply_async(args=[str(job.id)])
    return job
