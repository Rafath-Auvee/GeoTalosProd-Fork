from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.schemas.common import ORMModel

from geoops.export_common import SkippedSet

Task = Literal["detect", "classify", "segment", "yolo"]


class DinoExportPreviewRequest(ORMModel):
    """Validate + preview the training corpus for ONE class-specific head,
    without touching TiTiler, S3, or geoops-service.

    Unlike the YOLO tier (one multi-class model per collection), the DINOv2
    tier trains one head per class — so preview/dataset build are scoped to a
    single class_id within the collection, picked manually in the UI.
    """

    annotation_set_collection_id: UUID
    class_id: UUID
    aoi_id: UUID | None = Field(
        default=None,
        description="Restrict to annotations of this class tagged to this one tracked AOI "
        "(see MapAOI). Omit to count every AOI-tagged annotation of this class in the "
        "collection.",
    )
    aoi_ids: list[UUID] | None = Field(
        default=None, max_length=20,
        description="Several AOIs. Used together with annotation_set_ids: annotations are matched to the "
        "AOIs by position (PostGIS intersection), not by the aoi_id stamp.",
    )
    annotation_set_ids: list[UUID] | None = Field(
        default=None, max_length=100,
        description="Use exactly these annotation sets (any review status, same schema as the collection) "
        "instead of the collection's verified member sets.",
    )
    task: Task | None = Field(default=None, description="'segment' counts only annotations that are masks, not boxes.")
    class_ids: list[UUID] | None = Field(default=None, max_length=80, description="Several classes (YOLO): counted together.")


class DinoExportPreviewResponse(ORMModel):
    annotation_set_collection_id: UUID
    class_id: UUID
    class_name: str
    schema_id: UUID
    geometry_types: list[str]
    verified_sets_included: int
    sets_skipped: list[SkippedSet]
    dataset_item_count: int = Field(description="Distinct dataset_items contributing at least one eligible annotation of this class.")
    total_annotations: int
    annotations_skipped_no_item: int = Field(
        description="Verified annotations of this class excluded because neither the "
        "annotation nor its (dataset-wide) set carries a dataset_item_id.",
    )
    annotations_skipped_inactive_item: int
    median_object_span_m: float | None = Field(
        default=None, description="Median longer-side size (meters) of this class's eligible annotations."
    )
    object_size_p10_m: float | None = Field(default=None, description="10th percentile object size (m).")
    object_size_p90_m: float | None = Field(default=None, description="90th percentile object size (m).")
    size_spread: float | None = Field(default=None, description="p90 / p10 object size.")
    suggested_scales_m: list[float] = Field(
        default_factory=list,
        description="Default tile scales: several (multi-scale) when the size spread exceeds 4x, otherwise one.",
    )
    by_class: dict[str, int] = Field(default_factory=dict, description="Eligible annotations per class id (several classes).")
    suggested_tile_scale_m: float | None = Field(
        default=None, description="The single scale nearest the median object size (single-scale default)."
    )
    ready: bool = Field(description="False if there are zero eligible annotations of this class to train on.")


# ── Datasets ─────────────────────────────────────────────────────────────────


class DinoDatasetParams(ORMModel):
    """Inputs that shape a built dataset. Changing any of them means a rebuild."""

    scales_m: list[float] | None = Field(
        default=None,
        max_length=6,
        description="Real-world tile edge lengths (meters). Several = multi-scale: each object is tiled at "
        "the scale that makes it a similar fraction of the tile. Omit to derive them from the object sizes "
        "(multi-scale when the size spread exceeds 4x). The trained head scans at these same scales.",
    )
    tile_scale_m: float | None = Field(
        default=None,
        ge=5,
        le=2560,
        description="Single-scale shorthand for scales_m=[tile_scale_m].",
    )
    crop_px: int = Field(default=1280, ge=128, le=1280, description="Pixel size of each exported tile image.")
    eval_split: float = Field(default=0.15, ge=0.05, le=0.4, description="Share of regions used for validation during training.")
    test_split: float = Field(default=0.15, ge=0.05, le=0.4, description="Share of regions held out for the final metrics.")
    neg_ratio: float = Field(default=2.0, ge=0, le=10, description="Max background tiles kept per split, as a multiple of its positives.")
    region_tiles: int = Field(default=3, ge=1, le=10, description="Tiles per side of the square regions that are split as one unit.")
    set_roles: dict[str, Literal["train", "test", "both"]] = Field(
        default_factory=dict,
        description="Where each annotation set (by id) may be used: only in train, only in test, or in both "
        "(split by region, the default). Sets not listed are 'both'.",
    )

    @model_validator(mode="after")
    def _scales_valid(self):
        if self.scales_m is not None:
            if not self.scales_m or any(not (5 <= v <= 2560) for v in self.scales_m):
                raise ValueError("scales_m must be 1-6 values between 5 and 2560 metres")
            self.scales_m = sorted({round(float(v), 2) for v in self.scales_m})
        return self

    @model_validator(mode="after")
    def _splits_leave_train(self):
        if self.eval_split + self.test_split > 0.7:
            raise ValueError("eval_split + test_split must leave at least 30% for training")
        return self


class DinoDatasetCreateRequest(ORMModel):
    """Create a dataset and build it (async). Always creates a new dataset;
    to rebuild one, use ``POST /dino/datasets/{id}/rebuild``."""

    annotation_set_collection_id: UUID
    class_id: UUID
    aoi_id: UUID | None = Field(default=None, description="The exhaustively-labeled tracked AOI the dataset is built from and split inside.")
    aoi_ids: list[UUID] | None = Field(default=None, max_length=20, description="Several AOIs (needs annotation_set_ids); regions are split across all of them.")
    class_ids: list[UUID] | None = Field(default=None, max_length=80, description="Several classes (task 'yolo'); the first is also class_id. Their order is the YOLO class index.")
    annotation_set_ids: list[UUID] | None = Field(default=None, max_length=100, description="Exact annotation sets to use (same schema as the collection).")
    task: Task = "detect"
    name: str | None = Field(default=None, max_length=255)
    params: DinoDatasetParams = Field(default_factory=DinoDatasetParams)
    source_dataset_id: UUID | None = Field(
        default=None,
        description="Clone from a ready *detect* dataset (same collection/class/AOI) into a *classify* one: "
        "reuses its tile images and splits, only the labels change. Params are inherited. Omit to build "
        "an independent dataset from the annotations.",
    )

    @model_validator(mode="after")
    def _classes(self):
        if self.class_ids:
            self.class_ids = list(dict.fromkeys(self.class_ids))
            self.class_id = self.class_ids[0]
        elif self.task == "yolo":
            self.class_ids = [self.class_id]
        else:
            self.class_ids = None
        return self

    @model_validator(mode="after")
    def _scope(self):
        if self.aoi_ids:
            self.aoi_ids = list(dict.fromkeys(self.aoi_ids))
            self.aoi_id = self.aoi_ids[0]
        elif self.aoi_id:
            self.aoi_ids = [self.aoi_id]
        else:
            raise ValueError("Choose at least one AOI")
        if len(self.aoi_ids) > 1 and not self.annotation_set_ids:
            raise ValueError("Several AOIs need explicit annotation_set_ids")
        return self

    @model_validator(mode="after")
    def _clone_is_classify(self):
        if self.source_dataset_id and self.task != "classify":
            raise ValueError("Only a classify dataset can be cloned from a detect dataset")
        return self


class DinoDatasetRenameRequest(ORMModel):
    name: str = Field(min_length=1, max_length=255)


class DinoDatasetRebuildRequest(ORMModel):
    params: DinoDatasetParams | None = Field(
        default=None, description="New build params. Omit to rebuild with the dataset's current params."
    )


class DinoDatasetRead(ORMModel):
    id: UUID
    name: str
    annotation_set_collection_id: UUID
    class_id: UUID
    aoi_id: UUID
    aoi_ids: list[UUID] | None = None
    annotation_set_ids: list[UUID] | None = None
    class_ids: list[UUID] | None = None
    task: Task
    params: dict
    status: Literal["pending", "building", "ready", "failed"]
    build_no: int
    stats: dict | None = None
    source_dataset_id: UUID | None = None
    job_id: UUID | None = None
    error: str | None = None
    is_stale: bool = Field(default=False, description="Annotation data (or the source dataset) changed since the last build.")
    stale_reason: str | None = None
    created_at: datetime
    updated_at: datetime


class DinoDatasetItem(ORMModel):
    """One tile of a built dataset (GET /dino/datasets/{id}/items)."""

    item_id: str
    split: Literal["train", "eval", "test"]
    image_key: str = Field(description="Raw S3 key — fetch bytes via GET /dino/datasets/{id}/image?key=...")
    scale_m: float | None = Field(default=None, description="Tile edge length (m) this tile was cut at.")
    box_count: int | None = Field(default=None, description="detect only — number of boxes in this tile.")
    boxes: list[list[float]] | None = Field(
        default=None,
        description="detect/segment — each box as normalized [cx, cy, w, h] in [0,1] relative to this tile.",
    )
    polygons: list[list[float]] | None = Field(
        default=None,
        description="segment only — each object's outline as flat normalized [x1, y1, x2, y2, ...] in [0,1].",
    )
    label: int | None = Field(default=None, description="1 if the class is present in this tile, 0 otherwise.")


class DinoDatasetItemsResponse(ORMModel):
    dataset_id: UUID
    task: str
    total: int
    truncated: bool = Field(description="True if more tiles exist than were returned (see the endpoint's limit).")
    items: list[DinoDatasetItem]


# ── Training ─────────────────────────────────────────────────────────────────


class DinoFinetuneRequest(ORMModel):
    """Train one class-specific head from an already-built dataset (async job —
    poll ``GET /api/jobs/{job_id}``). Nothing is regenerated: the dataset's
    tiles and split are used exactly as built. The split fractions and tile
    scale are properties of the dataset, not of a training run."""

    dataset_id: UUID
    backbone_model_id: UUID | None = Field(
        default=None,
        description="Existing 'backbone' ai_models row to fine-tune against. If unset, "
        "geoops-service falls back to its default DINOv2 backbone.",
    )
    backbone: Literal["dinov2-vits14", "dinov2-vitb14"] = Field(
        default="dinov2-vits14", description="Frozen DINOv2 backbone: small (faster) or base (more capacity)."
    )
    epochs: int = Field(default=60, ge=1, le=2000)
    # 1e-3 (AdamW) destabilised this tier: with few-shot data each epoch is a
    # handful of high-variance steps against a fresh pyramid+head, and the
    # loss would climb back up after the first epochs. Training now keeps the
    # best-validation epoch, but 1e-4 is still the stabler starting point.
    lr: float = Field(default=1e-4, gt=0, le=1.0)
    result_model_name: str | None = Field(default=None, description="Name for the resulting per-class head AIModel row.")
    allow_stale: bool = Field(
        default=False, description="Train even though the dataset's source data changed since it was built."
    )
    base_weights: str | None = Field(
        default=None,
        description="YOLO only: the file to start from: a name in the service's models/yolo folder, or the storage key "
        "of an earlier YOLO model (to add classes to it). Default: yolo26l.pt.",
    )
    stage1_epochs: int = Field(default=40, ge=0, le=500, description="YOLO: neck and head only, backbone frozen.")
    stage2_epochs: int = Field(default=40, ge=0, le=500, description="YOLO: all layers, lower learning rate. 0 skips this stage.")
    stage1_lr: float = Field(default=0.002, gt=0, le=0.1, description="YOLO: learning rate of stage 1 (AdamW).")
    stage2_lr: float = Field(default=0.0001, gt=0, le=0.1, description="YOLO: learning rate of stage 2 (AdamW).")
    weight_decay: float = Field(default=0.01, ge=0, le=1.0, description="AdamW weight decay (DINOv2 heads).")
    auto_tune: bool = Field(
        default=False,
        description="Pick the learning rate (and, for DINOv2 heads, epochs and weight decay) with Optuna. Tuned once per "
        "dataset build + model; later runs reuse the saved config unless force_retune is set.",
    )
    n_trials: int = Field(default=6, ge=1, le=20, description="Optuna search trials when tuning runs (the request's own values are trial 0).")
    force_retune: bool = Field(default=False, description="Tune again even if this dataset + model already has a saved config.")
    seeds: list[int] = Field(
        default_factory=list,
        description="Multi-run: train once per seed (3 to 5 seeds), evaluate each on the test split and report mean ± std. "
        "Empty = a single run. The kept model is the seed with the best validation loss.",
    )

    @model_validator(mode="after")
    def _seeds_valid(self):
        self.seeds = list(dict.fromkeys(self.seeds))
        if self.seeds and not (3 <= len(self.seeds) <= 5):
            raise ValueError("seeds must be empty (single run) or 3 to 5 distinct values")
        if any(not (0 <= v < 2**31) for v in self.seeds):
            raise ValueError("seeds must be non-negative integers")
        return self


class DinoTrainTimings(ORMModel):
    train_seconds: float
    eval_seconds: float
    total_seconds: float


class DinoFinetuneResult(ORMModel):
    """Shape of ``job.config["result"]`` once a finetune_model (dino head) job completes.

    A run is a *report* — dataset reference, hyperparameters, loss curves,
    metrics and timings — not a copy of the data it trained on.
    """

    dataset_id: str
    dataset_build_no: int
    task: str
    class_id: str
    class_name: str
    train_items: int
    eval_items: int
    test_items: int
    patch_span_m: float | None = Field(
        default=None, description="Fixed real-world tile size (meters) used for both training tiles and scan tiles."
    )
    hyperparameters: dict | None = None
    training_triggered: bool
    ai_model_id: str | None = None
    backbone_model_id: str | None = None
    train_result: dict | None = None
    eval_result: dict | None = None
    tuning: dict | None = Field(default=None, description="Auto-tune summary: cache_hit, best_params, best_value, trials, tuned_config_id.")
    multirun: dict | None = Field(default=None, description="Multi-seed report: per-metric mean/std/min/max and the per-seed runs.")
    timings: DinoTrainTimings | None = None
    note: str | None = None
    error: str | None = None


# ── Annotation sets inside AOIs ──────────────────────────────────────────────


class DinoAoiSetRead(ORMModel):
    """One annotation set that has annotations inside the chosen AOIs."""

    id: UUID
    name: str
    review_status: str
    source_type: str | None = None
    annotations_in_aois: int = Field(description="Annotations of the set inside the chosen AOIs (any class).")
    mask_annotations: int = Field(default=0, description="Of those, how many are masks (more than a box's four corners).")
    class_annotations: int = Field(description="Of those, how many carry the chosen class.")
    by_aoi: dict[str, int] = Field(description="Annotations inside each AOI, keyed by AOI id (any class).")


class DinoAoiSetsRequest(ORMModel):
    annotation_set_collection_id: UUID
    class_id: UUID | None = None
    class_ids: list[UUID] | None = Field(default=None, max_length=80)
    aoi_ids: list[UUID] = Field(min_length=1, max_length=20)


# ── Tuned configs & multi-run reports ────────────────────────────────────────


class DinoTunedConfigRead(ORMModel):
    """The saved Optuna result for one dataset + model (GET /dino/datasets/{id}/tuned-config)."""

    id: UUID
    dataset_id: UUID
    dataset_build_no: int
    model_key: str
    best_params: dict = Field(description="The final hyperparameters: lr / epochs / weight_decay, or stage1_lr / stage2_lr for YOLO.")
    tuning: dict = Field(description="metric, direction, n_trials, best_value, best_trial, seed, seconds and every trial.")
    multirun: dict | None = Field(default=None, description="Latest multi-seed report run with best_params (mean ± std per metric).")
    job_id: UUID | None = None
    is_stale: bool = Field(default=False, description="The dataset was rebuilt since this was tuned; the next auto-tuned run re-tunes.")
    created_at: datetime
    updated_at: datetime


class DinoRunReport(ORMModel):
    """Tuning + multi-run report of one finished run (GET /dino/runs/{job_id}/report)."""

    job_id: UUID
    dataset_id: str | None = None
    task: str | None = None
    hyperparameters: dict | None = None
    tuning: dict | None = None
    multirun: dict | None = None
