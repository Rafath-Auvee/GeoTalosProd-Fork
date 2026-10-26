"""Celery task: train + evaluate ONE class-specific DINOv2 head from an
already-built DinoDataset, and register the result in the model catalogue
(ai_models + model_class_mappings) so it is immediately usable through the
existing ModelManager inference pipeline.

Dataset *building* (tiling the AOI, region split, writing tiles) lives in
geoops.dino_dataset_tasks and happens once per dataset, not once per run —
this task only reads the dataset's manifest prefix via geoops-service
``/train`` and ``/eval``. A run's ``config.result`` is therefore a report
(dataset reference, hyperparameters, loss curves, metrics, timings), not a
copy of the data.

Sync (WorkerSession / psycopg2). HTTP calls to geoops-service are
deliberately synchronous (urllib), mirroring geoops/tasks.py.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from urllib import request as urlrequest

from geoalchemy2.shape import to_shape
from shapely.geometry import mapping
from sqlalchemy import func, select

from app.config import settings
from app.core.enums import JobStatus
from app.models.ai_model import AIModel
from app.models.annotation import Annotation
from app.models.annotation_class import AnnotationClass
from app.models.annotation_set import AnnotationSet
from app.models.annotation_set_collection import AnnotationSetCollection
from app.models.dataset_item import DatasetItem
from app.models.job import Job
from app.models.job_output import JobOutput
from app.models.model_class_mapping import ModelClassMapping
from app.services import storage_service
from app.workers.celery_app import celery_app
from app.workers.db import WorkerSession
from app.workers.queues import TRAINING

from geoops.export_common import partition_member_sets_sync
from geoops.models import DinoDataset, DinoTunedConfig
from geoops.scale import SIZE_FRAC_RANGE

logger = logging.getLogger(__name__)

DEFAULT_BACKBONE_KEY = "dinov2-vits14"  # small variant — 21M params vs. vitb14's 86.6M, much faster on CPU
BACKBONE_HF_IDS = {
    "dinov2-vits14": "facebook/dinov2-small",
    "dinov2-vitb14": "facebook/dinov2-base",
}

# Per-task: geoops-service training kind, registered model type, default
# decision threshold, and the display name suffix.
_TASKS = {
    "detect": {"kind": "fcos", "model_type": "detect_head", "score_thresh": 0.3, "suffix": "DINOv2 head"},
    "classify": {"kind": "classify", "model_type": "classify_head", "score_thresh": 0.5, "suffix": "DINOv2 classifier"},
    "segment": {"kind": "segment", "model_type": "segment_head", "score_thresh": 0.5, "suffix": "DINOv2 segmenter"},
    "yolo": {"kind": "yolo", "model_type": "yolo_detect", "score_thresh": 0.25, "suffix": "YOLO detector"},
}


def _call_geoops_service(path: str, body: dict, timeout: float = 1800.0) -> dict:
    url = f"{settings.GEOOPS_SERVICE_URL.rstrip('/')}{path}"
    req = urlrequest.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.GEOOPS_SERVICE_API_TOKEN}",
        },
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return json.loads(resp.read().decode("utf-8"))


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def explicit_set_ids(session, org_id: uuid.UUID, collection, set_ids: list[uuid.UUID]) -> list[uuid.UUID]:
    """The chosen annotation sets that are live and share the collection's schema (any review status)."""
    sets = session.scalars(
        select(AnnotationSet).where(
            AnnotationSet.id.in_(set_ids),
            AnnotationSet.organization_id == org_id,
            AnnotationSet.deleted_at.is_(None),
        )
    ).all()
    return [s.id for s in sets if s.schema_id == collection.schema_id]


def _gather_rows_for_class(
    session, org_id: uuid.UUID, collection_id: uuid.UUID, class_id: uuid.UUID,
    set_ids: list[uuid.UUID] | None = None,
):
    """Sync equivalent of DinoExportService.build_preview, but returns full rows
    (item + geometry) rather than just counts. Returns (rows, skipped, collection, class_name).
    """
    collection = session.get(AnnotationSetCollection, collection_id)
    if collection is None or collection.organization_id != org_id:
        raise ValueError("AnnotationSetCollection not found")

    cls = session.get(AnnotationClass, class_id)
    if cls is None or cls.schema_id != collection.schema_id:
        raise ValueError("AnnotationClass not found for this collection's schema")

    if set_ids:
        included_ids, skipped = explicit_set_ids(session, org_id, collection, set_ids), []
    else:
        included_ids, skipped = partition_member_sets_sync(session, collection)
    if not included_ids:
        return [], skipped, collection, cls.name

    item_id_expr = func.coalesce(Annotation.dataset_item_id, AnnotationSet.dataset_item_id)
    query = (
        select(Annotation, item_id_expr.label("item_id"))
        .select_from(Annotation)
        .join(AnnotationSet, AnnotationSet.id == Annotation.annotation_set_id)
        .where(
            Annotation.annotation_set_id.in_(included_ids),
            Annotation.class_id == class_id,
            Annotation.deleted_at.is_(None),
        )
    )

    rows: list[dict] = []
    items_cache: dict[uuid.UUID, DatasetItem | None] = {}
    for ann, item_id in session.execute(query).all():
        if item_id is None:
            continue
        if item_id not in items_cache:
            items_cache[item_id] = session.get(DatasetItem, item_id)
        item = items_cache[item_id]
        if item is None or not item.is_active:
            continue
        rows.append({
            "item": item,
            "annotation_id": ann.id,
            "set_id": str(ann.annotation_set_id),
            "class_id": str(ann.class_id),
            "updated_at": ann.updated_at,
            "geometry": mapping(to_shape(ann.geometry)),
            "aoi_id": (ann.properties or {}).get("aoi_id"),
        })
    return rows, skipped, collection, cls.name


def _gather_rows_for_classes(
    session, org_id: uuid.UUID, collection_id: uuid.UUID, class_ids: list[uuid.UUID], set_ids: list[uuid.UUID] | None = None,
):
    """``_gather_rows_for_class`` for several classes at once. Returns (rows, skipped, collection, class names in order)."""
    rows: list[dict] = []
    skipped: list = []
    names: list[str] = []
    collection = None
    for i, cid in enumerate(class_ids):
        r, sk, collection, name = _gather_rows_for_class(session, org_id, collection_id, cid, set_ids)
        rows += r
        skipped = skipped or sk
        names.append(name)
    return rows, skipped, collection, names


def _filter_rows_by_aoi(rows: list[dict], aoi_id: str | None) -> tuple[list[dict], int]:
    """Restrict export rows to one tracked AOI, when the caller (POST
    /dino/finetune's ``aoi_id``) asked for it — e.g. when another AOI has
    annotations of the same class but isn't yet exhaustively labeled, so its
    empty tiles can't be trusted as real negatives (see module docstring).
    Returns (filtered_rows, skipped_other_aoi_count).
    """
    if not aoi_id:
        return rows, 0
    kept = [r for r in rows if r.get("aoi_id") == aoi_id]
    return kept, len(rows) - len(kept)


def _fail(session, job: Job, message: str, result: dict | None = None) -> None:
    job.status = JobStatus.FAILED
    job.logs = message[:2000]
    if result is not None:
        job.config = {**(job.config or {}), "result": result}
    job.finished_at = _now()
    session.commit()


def _find_or_create_backbone(session, org_id: uuid.UUID, backbone_key: str) -> AIModel:
    row = session.execute(
        select(AIModel).where(
            AIModel.organization_id == org_id,
            AIModel.type == "backbone",
            AIModel.framework == "dinov2",
            AIModel.deleted_at.is_(None),
        )
    ).scalars().first()
    if row is None:
        row = AIModel(
            organization_id=org_id,
            name=f"DINOv2 backbone ({backbone_key})",
            type="backbone",
            framework="dinov2",
            config={"source": "huggingface", "model_id": BACKBONE_HF_IDS.get(backbone_key, backbone_key)},
        )
        session.add(row)
        session.flush()
    return row


@celery_app.task(bind=True, queue=TRAINING, max_retries=1, default_retry_delay=30)
def run_dino_train(self, job_id: str) -> None:
    with WorkerSession() as session:
        job = session.get(Job, uuid.UUID(job_id))
        if job is None:
            logger.warning("run_dino_train: job %s not found", job_id)
            return

        req = (job.config or {}).get("request") or {}
        job.status = JobStatus.RUNNING
        job.started_at = _now()
        session.commit()

        try:
            ds = session.get(DinoDataset, uuid.UUID(req["dataset_id"]))
            if ds is None or ds.organization_id != job.organization_id:
                _fail(session, job, "Dataset not found.")
                return
            if ds.status != "ready" or not ds.s3_prefix:
                _fail(session, job, f"Dataset is {ds.status}, not ready — build it before training.")
                return

            task = _TASKS[ds.task]
            collection = session.get(AnnotationSetCollection, ds.annotation_set_collection_id)
            class_ids = [uuid.UUID(c) for c in (ds.class_ids or [str(ds.class_id)])]
            classes = [session.get(AnnotationClass, c) for c in class_ids]
            if collection is None or any(c is None for c in classes):
                _fail(session, job, "The dataset's collection or a class no longer exists.")
                return
            class_label = " + ".join(c.name for c in classes)
            is_yolo = ds.task == "yolo"
            stage1, stage2 = int(req.get("stage1_epochs", 40)), int(req.get("stage2_epochs", 40))
            base_weights = req.get("base_weights") or "yolo26l.pt"
            backbone_req = req.get("backbone") if req.get("backbone") in BACKBONE_HF_IDS else DEFAULT_BACKBONE_KEY
            # What the saved tuned config is keyed on, next to the dataset: the backbone, or YOLO's starting weights.
            model_key = base_weights if is_yolo else backbone_req

            # Auto-tune: reuse this dataset build's saved config for the model if there is one, else tune this run.
            seeds = [int(v) for v in (req.get("seeds") or [])]
            tuned_row, tune_trials, tuned_params = None, 0, {}
            if req.get("auto_tune"):
                tuned_row = session.scalars(
                    select(DinoTunedConfig).where(DinoTunedConfig.dataset_id == ds.id, DinoTunedConfig.model_key == model_key)
                ).first()
                # A config saved under an older search rule (epochs fixed, or sampled) is not reused: search again.
                if (
                    tuned_row is not None and tuned_row.dataset_build_no == ds.build_no and not req.get("force_retune")
                    and (is_yolo or tuned_row.tuning.get("epoch_search") == "best_epoch")
                ):
                    tuned_params = dict(tuned_row.best_params)  # cache hit: no tuning, train with the saved config
                else:
                    tune_trials = int(req.get("n_trials", 6))
            cache_hit = bool(req.get("auto_tune")) and tune_trials == 0
            stage1_lr = float(tuned_params.get("stage1_lr", req.get("stage1_lr", 0.002)))
            stage2_lr = float(tuned_params.get("stage2_lr", req.get("stage2_lr", 0.0001)))
            weight_decay = float(tuned_params.get("weight_decay", req.get("weight_decay", 0.01)))

            splits = (ds.stats or {}).get("splits", {})
            tiles = {s: int(splits.get(s, {}).get("tiles", 0)) for s in ("train", "eval", "test")}
            scale_m = (ds.params or {}).get("tile_scale_m")
            epochs = stage1 + stage2 if is_yolo else int(tuned_params.get("epochs", req.get("epochs", 60)))
            lr = 0.001 if is_yolo else float(tuned_params.get("lr", req.get("lr", 1e-4)))
            backbone_key = backbone_req
            if is_yolo:
                backbone_key = "earlier model" if "/" in base_weights else base_weights.removesuffix(".pt")  # what it started from, for the report
            base_result = {
                "dataset_id": str(ds.id),
                "dataset_build_no": ds.build_no,
                "task": ds.task,
                "class_id": str(ds.class_id),
                "class_name": class_label,
                "train_items": tiles["train"],
                "eval_items": tiles["eval"],
                "test_items": tiles["test"],
                "patch_span_m": scale_m,
                "hyperparameters": {"epochs": epochs, "lr": lr, "backbone": backbone_key},
            }
            job.total_items = epochs
            session.commit()

            t0 = time.monotonic()
            try:
                train_result = _call_geoops_service(
                    "/train",
                    {
                        "org_id": str(job.organization_id),
                        "backbone_key": backbone_key if not is_yolo else DEFAULT_BACKBONE_KEY,
                        "kind": task["kind"],
                        "s3_prefix": ds.s3_prefix,
                        "epochs": epochs,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "tune_trials": tune_trials,
                        "seeds": seeds,
                        # Models live apart from the data: one folder per run, so neither deletes the other.
                        "artifact_prefix": f"dino-models/{job.id}",
                        **(
                            {
                                "base_weights": base_weights, "stage1_epochs": stage1, "stage2_epochs": stage2,
                                "stage1_lr": stage1_lr, "stage2_lr": stage2_lr,
                                # The model trains on the tiles as exported: no resizing between the UI and the network.
                                "imgsz": -(-int((ds.params or {}).get("crop_px") or 640) // 32) * 32,
                            }
                            if is_yolo
                            else {}
                        ),
                    },
                    # One request covers every tuning trial and every seed, each a full training run.
                    timeout=(7200.0 if is_yolo else 1800.0) * (tune_trials + max(1, len(seeds))),
                )
                t1 = time.monotonic()
                eval_body = {
                    "org_id": str(job.organization_id),
                    "s3_prefix": ds.s3_prefix,
                    "artifact_key": train_result["artifact_key"],
                    "kind": task["kind"],
                    "split": "test",
                }
                if ds.task in ("detect", "segment", "yolo"):
                    eval_body["iou_thresh"] = 0.5
                if ds.task == "segment":
                    eval_body["score_thresh"] = task["score_thresh"]
                eval_result = _call_geoops_service("/eval", eval_body)
                t2 = time.monotonic()
            except Exception as exc:  # noqa: BLE001
                logger.exception("dino_train_or_eval_failed job_id=%s", job.id)
                _fail(
                    session, job, f"Training/eval failed: {exc}",
                    {**base_result, "training_triggered": False, "error": str(exc)},
                )
                return

            tuning, multirun = train_result.pop("tuning", None), train_result.pop("multirun", None)
            applied = train_result.pop("applied_params", None) or tuned_params
            lr = float(applied.get("lr", lr))
            weight_decay = float(applied.get("weight_decay", weight_decay))
            stage1_lr = float(applied.get("stage1_lr", stage1_lr))
            stage2_lr = float(applied.get("stage2_lr", stage2_lr))
            if applied.get("epochs") is not None:
                epochs = int(applied["epochs"])  # the epochs the search found, which every final run used
            tuned_config_id = None
            if tuning is not None:  # tuned just now: save it for every later run on this dataset build + model
                if tuned_row is None:
                    tuned_row = DinoTunedConfig(organization_id=job.organization_id, dataset_id=ds.id, model_key=model_key)
                    session.add(tuned_row)
                tuned_row.dataset_build_no, tuned_row.best_params, tuned_row.tuning = ds.build_no, applied, tuning
            if tuned_row is not None and req.get("auto_tune"):
                if multirun is not None:
                    tuned_row.multirun = multirun
                if tuning is not None:
                    tuned_row.job_id = job.id
                session.flush()
                tuned_config_id = str(tuned_row.id)
            tuning_summary = None
            if req.get("auto_tune"):
                tuning_summary = {
                    "cache_hit": cache_hit, "tuned_config_id": tuned_config_id, "model_key": model_key,
                    "best_params": applied, **({k: tuning[k] for k in ("metric", "n_trials", "best_value", "best_trial", "trials", "seconds", "fixed")} if tuning else {
                        "n_trials": tuned_row.tuning.get("n_trials"), "best_value": tuned_row.tuning.get("best_value"),
                        "metric": tuned_row.tuning.get("metric"), "trials": tuned_row.tuning.get("trials"), "fixed": tuned_row.tuning.get("fixed"),
                    }),
                }
            hp_source = "manual" if not req.get("auto_tune") else ("saved_config" if cache_hit else "auto_tuned")
            base_result["hyperparameters"] = {
                "epochs": epochs, "lr": lr, "backbone": backbone_key, "weight_decay": weight_decay, "seeds": seeds or None, "source": hp_source,
            }

            timings = {
                "train_seconds": round(t1 - t0, 2),
                "eval_seconds": round(t2 - t1, 2),
                "total_seconds": round(t2 - t0, 2),
            }

            backbone_row = None if is_yolo else _find_or_create_backbone(session, job.organization_id, backbone_key)
            artifact_key = train_result["artifact_key"]
            bucket = storage_service.bucket_name(job.organization_id)
            payload = {
                "org_id": str(job.organization_id),
                "artifact_key": artifact_key,
                "class_label": class_label,
                "backbone_key": backbone_key,
                "score_thresh": task["score_thresh"],
            }
            if ds.task in ("classify", "segment", "yolo"):
                payload["kind"] = ds.task
            # The head scans at EXACTLY the scales it was trained on — real-world
            # metres, not pixel counts. ModelManager converts each to pixels
            # using the target image's OWN resolution at scan time, so train
            # and scan scale match by construction. ``scale_locked`` stops a
            # run overriding it; ``size_frac_range`` lets inference discard
            # detections too big/small to be trusted at a given scale;
            # ``train_render`` pins the band/rescale recipe the tiles were
            # rendered with (see geoops.dino_dataset_tasks), which inference
            # reuses instead of whatever bands are open in the map view.
            scales = [float(v) for v in ((ds.params or {}).get("scales_m") or [scale_m])]
            stats = ds.stats or {}
            training = {
                "trained_at": _now().isoformat(),
                "task": ds.task,
                "class_id": str(ds.class_id),
                "class_name": class_label,
                "classes": [{"id": str(c.id), "name": c.name} for c in classes],
                "aoi_id": str(ds.aoi_id),
                "dataset": {
                    "id": str(ds.id),
                    "name": ds.name,
                    "build_no": ds.build_no,
                    "scales_m": scales,
                    "crop_px": (ds.params or {}).get("crop_px"),
                    "region_tiles": (ds.params or {}).get("region_tiles"),
                    "eval_split": (ds.params or {}).get("eval_split"),
                    "test_split": (ds.params or {}).get("test_split"),
                    "object_size_p10_m": stats.get("object_size_p10_m"),
                    "object_size_p90_m": stats.get("object_size_p90_m"),
                    "annotations": stats.get("annotations"),
                    "splits": stats.get("splits"),
                    "by_scale": stats.get("by_scale"),
                },
                "hyperparameters": {"epochs": epochs, "lr": lr, "weight_decay": weight_decay, "source": hp_source, "backbone": backbone_key, **({"stages": train_result.get("stages")} if is_yolo else {})},
                "tuning": tuning_summary,
                "multirun": multirun,
                "best_epoch": train_result.get("best_epoch"),
                "best_val_loss": train_result.get("best_val_loss"),
                "final_loss": train_result.get("final_loss"),
                "timings": timings,
                "eval_metrics": eval_result,
            }
            head_row = AIModel(
                organization_id=job.organization_id,
                name=req.get("result_model_name") or f"{class_label} ({task['suffix']})",
                type=task["model_type"],
                framework="yolo" if is_yolo else "dinov2",
                backbone_model_id=backbone_row.id if backbone_row else None,
                artifact_uri=f"s3://{bucket}/{artifact_key}",
                endpoint_url=f"{settings.GEOOPS_SERVICE_URL.rstrip('/')}/infer",
                request_config={"payload": payload},
                auth_config={"bearer_token": settings.GEOOPS_SERVICE_API_TOKEN},
                output_config={
                    "adapter": "platform_passthrough",
                    "patch_size_m": scale_m,
                    "patch_sizes_m": scales,
                    "size_frac_range": list(SIZE_FRAC_RANGE),
                    "scale_locked": True,
                    "train_render": {"assets": "data", "asset_bidx": "data|1,2,3", "rescale": "auto_per_item"},
                },
                config={
                    "kind": task["kind"],
                    "eval_metrics": eval_result,
                    "dataset_id": str(ds.id),
                    "dataset_build_no": ds.build_no,
                    "training": training,
                },
                annotation_schema_id=collection.schema_id,
                created_by=job.created_by_user_id,
            )
            session.add(head_row)
            session.flush()
            for c in classes:  # a multi-class model labels each detection with its class name
                session.add(ModelClassMapping(model_id=head_row.id, model_label=c.name, annotation_class_id=c.id))
            session.add(JobOutput(job_id=job.id, output_type="ai_model", output_id=head_row.id))
            job.model_id = head_row.id

            headline = (
                f"AP50={eval_result.get('ap50')}" if ds.task in ("detect", "segment", "yolo") else f"Accuracy={eval_result.get('accuracy')}"
            )
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.processed_items = epochs
            job.logs = (
                f"Trained + registered {class_label} {ds.task} model (ai_model_id={head_row.id}) from dataset "
                f"b{ds.build_no}. {headline} on {eval_result.get('num_images', 0)} test tiles."
            )
            job.config = {
                **(job.config or {}),
                "result": {
                    **base_result,
                    "training_triggered": True,
                    "ai_model_id": str(head_row.id),
                    "backbone_model_id": str(backbone_row.id) if backbone_row else None,
                    "train_result": train_result,
                    "eval_result": eval_result,
                    "tuning": tuning_summary,
                    "multirun": multirun,
                    "timings": timings,
                },
            }
            job.finished_at = _now()
            session.commit()
        except Exception as exc:
            session.rollback()
            job2 = session.get(Job, uuid.UUID(job_id))
            if job2 is not None:
                _fail(session, job2, str(exc))
            raise
