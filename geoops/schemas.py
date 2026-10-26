from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.schemas.common import ORMModel


def _validate_bbox(bbox: list[float]) -> list[float]:
    minx, miny, maxx, maxy = bbox
    if minx >= maxx or miny >= maxy:
        raise ValueError("bbox must be [minx, miny, maxx, maxy] with min < max")
    if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
        raise ValueError("bbox must be within EPSG:4326 bounds")
    return bbox


class EmbeddingCreateRequest(ORMModel):
    """Generate + store an embedding for one bbox/mask selection.

    Synchronous — a single crop + one call to the embed model, not a bulk job.
    """

    model_id: UUID
    dataset_item_id: UUID
    geometry: dict[str, Any] = Field(
        description="GeoJSON geometry (EPSG:4326) of the bbox or mask the user selected. "
        "Only its bounding box is used to crop the source patch."
    )
    annotation_id: UUID | None = Field(
        default=None,
        description="If the selection is already a saved Annotation, link the embedding to it "
        "and inherit its class_id (an explicit class_id is ignored when this is set).",
    )
    class_id: UUID | None = Field(
        default=None,
        description="Annotation class this embedding represents. Ignored if annotation_id is set.",
    )
    crop_size_px: int = Field(default=256, ge=32, le=1024)


class EmbeddingRead(ORMModel):
    id: UUID
    organization_id: UUID
    model_id: UUID | None
    model_name: str
    embedding_dim: int
    dataset_item_id: UUID
    class_id: UUID | None
    annotation_id: UUID | None
    source_geometry: dict[str, Any]
    tile_tier: int
    created_at: datetime


class EmbeddingSearchRequest(ORMModel):
    """Nearest-neighbour search over the embedding bank.

    Always scoped to the reference embedding's ``model_name`` — vectors from
    different embedding models are never compared against each other.
    """

    embedding_id: UUID
    top_k: int = Field(default=10, ge=1, le=100)
    class_id: UUID | None = Field(default=None, description="Restrict candidates to this class.")
    dataset_id: UUID | None = Field(
        default=None, description="Restrict candidates to items belonging to this dataset."
    )
    exclude_self: bool = True


class EmbeddingSearchMatch(ORMModel):
    embedding_id: UUID
    dataset_item_id: UUID
    class_id: UUID | None
    annotation_id: UUID | None
    source_geometry: dict[str, Any]
    similarity: float = Field(description="1 - cosine distance. 1.0 = identical, 0.0 = orthogonal.")
    distance: float


class AOIScanRequest(ORMModel):
    """Async similarity scan over an AOI (runs as a Celery job — see
    ``POST /embeddings/aoi-scan``, which returns a ``job_id`` to poll).

    Tile size is derived entirely from the reference embedding's own scale
    tier (``geoops.scale``) — not configurable per request — so repeated
    scans of overlapping AOIs always produce the same tile grid and can
    reuse cached ``embedding_tiles`` rows instead of re-embedding.
    """

    reference_embedding_id: UUID
    dataset_item_ids: list[UUID] = Field(min_length=1, max_length=5)
    aoi_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    crop_size_px: int = Field(default=256, ge=32, le=1024)
    top_k: int = Field(default=20, ge=1, le=200)
    min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    max_patches: int = Field(
        default=200, ge=1, le=400,
        description="Cap on new tiles requiring a fresh model call — cached tiles from a "
        "prior scan don't count against this.",
    )
    output_class_id: UUID = Field(
        description="Annotation class the returned matches (top_k) are persisted under, as a "
        "new annotation_set (source_type='analysis') mounted as a map layer once the job "
        "completes.",
    )

    @model_validator(mode="after")
    def validate_aoi_bbox(self):
        if self.aoi_bbox is not None:
            _validate_bbox(self.aoi_bbox)
        return self


class AOIScanMatch(ORMModel):
    dataset_item_id: UUID
    bbox: list[float]
    similarity: float
    distance: float


class AOIScanResponse(ORMModel):
    """Shape of ``job.config["result"]`` once an AOI-scan job completes."""

    reference_embedding_id: UUID
    model_name: str
    patches_scanned: int
    patches_cached: int
    patches_embedded: int
    patches_truncated: bool = False
    matches: list[AOIScanMatch]


class ExemplarScanRequest(ORMModel):
    """Async multi-exemplar similarity scan over an AOI (runs as a Celery
    job — see ``POST /embeddings/exemplar-scan``, which returns a ``job_id``
    to poll).

    Unlike ``AOIScanRequest``, tile size/overlap are auto-inferred from the
    reference embeddings' own real-world size (target ~40-100px per object)
    rather than a fixed ``crop_size_px`` grid, and multiple references plus
    optional hard negatives can be supplied — the workflow for "accept a
    candidate, reject another, scan again" (see geoops/README.md's exemplar-
    scan design note).
    """

    reference_embedding_ids: list[UUID] = Field(
        min_length=1, max_length=10, description="Exemplar embeddings, all same model_name + tile_tier."
    )
    negative_embedding_ids: list[UUID] = Field(
        default_factory=list, max_length=50,
        description="Previously-rejected candidates' embeddings — a tile scoring closer to a "
        "negative than to every positive reference is dropped.",
    )
    dataset_item_ids: list[UUID] = Field(min_length=1, max_length=5)
    aoi_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    crop_size_px: int = Field(default=128, ge=32, le=512)
    overlap_ratio: float | None = Field(
        default=None, gt=0.0, lt=1.0,
        description="Fraction of tile width/height consecutive tiles overlap by. Defaults to "
        "the reference object's own pixel footprint (at least one object width), clamped to "
        "[0.2, 0.6].",
    )
    top_k: int = Field(default=20, ge=1, le=200)
    min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    iou_thresh: float = Field(default=0.3, ge=0.0, le=1.0, description="NMS threshold for merging duplicate candidates from overlapping tiles.")
    max_patches: int = Field(default=200, ge=1, le=400)
    refine_with_sam: bool = Field(
        default=False, description="Tighten top_k candidate boxes via a registered SAM3 model."
    )
    sam_model_id: UUID | None = Field(
        default=None, description="Required when refine_with_sam=True — a registered SAM3 AIModel."
    )
    output_class_id: UUID = Field(
        description="Annotation class the returned matches (top_k) are persisted under, as a "
        "new annotation_set (source_type='analysis') mounted as a map layer once the job "
        "completes.",
    )

    @model_validator(mode="after")
    def validate_aoi_bbox(self):
        if self.aoi_bbox is not None:
            _validate_bbox(self.aoi_bbox)
        if self.refine_with_sam and self.sam_model_id is None:
            raise ValueError("sam_model_id is required when refine_with_sam=True")
        return self


class ExemplarScanMatch(ORMModel):
    dataset_item_id: UUID
    bbox: list[float]
    similarity: float
    refined: bool = False


class ExemplarScanResponse(ORMModel):
    """Shape of ``job.config["result"]`` once an exemplar-scan job completes."""

    model_name: str
    meters_per_pixel: float
    object_px: int
    overlap_ratio: float
    patches_scanned: int
    patches_cached: int
    patches_embedded: int
    patches_truncated: bool = False
    candidates_before_nms: int
    refine_errors: int = 0
    matches: list[ExemplarScanMatch]


class AutoAnnotateRequest(ORMModel):
    """"Pick a class, click run" async similarity scan (runs as a Celery job —
    see ``POST /embeddings/auto-annotate``, which returns a ``job_id`` to
    poll). Takes the same "verified collection + class" inputs as
    ``POST /dino/finetune`` (Train Head) instead of manually drawn/embedded
    reference boxes: a handful of that class's existing verified annotations
    are auto-embedded and used as the exemplar set, then scored/tiled/merged
    exactly like ``POST /embeddings/exemplar-scan``.
    """

    annotation_set_collection_id: UUID
    class_id: UUID = Field(description="References are drawn from this class's existing verified annotations.")
    embedding_model_id: UUID
    dataset_item_ids: list[UUID] = Field(min_length=1, max_length=5)
    aoi_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    max_references: int = Field(
        default=5, ge=1, le=10,
        description="How many of the class's existing verified annotations to use as references.",
    )
    crop_size_px: int = Field(default=128, ge=32, le=512)
    top_k: int = Field(default=20, ge=1, le=200)
    min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    iou_thresh: float = Field(default=0.3, ge=0.0, le=1.0)
    max_patches: int = Field(default=200, ge=1, le=400)
    refine_with_sam: bool = Field(default=False)
    sam_model_id: UUID | None = Field(default=None)
    output_class_id: UUID | None = Field(
        default=None,
        description="Annotation class results are tagged under — defaults to class_id (same class) "
        "when omitted, so results extend the same class's annotation set.",
    )

    @model_validator(mode="after")
    def validate_aoi_bbox(self):
        if self.aoi_bbox is not None:
            _validate_bbox(self.aoi_bbox)
        if self.refine_with_sam and self.sam_model_id is None:
            raise ValueError("sam_model_id is required when refine_with_sam=True")
        return self


class Sam3AutoAnnotateRequest(ORMModel):
    """Direct SAM3 box+text auto-annotate over an AOI (async job — runs as a
    Celery job, see ``POST /sam3/auto-annotate``, which returns a ``job_id``
    to poll).

    Same "verified collection + class" input as ``POST /dino/finetune``
    (Train Head) supplies the exemplar set, plus an explicit ``dataset_item_ids``
    AOI to scan (same shape as ``AutoAnnotateRequest``/``ExemplarScanRequest``).
    SAM3's box-prompt API is single-image only (a box is only meaningful in
    the same image's pixel space it came from — see geoops-service's
    ``app/sam3_infer.py`` docstring), so a reference's bbox is only usable as
    a box exemplar on tiles cropped from *that same reference's own dataset
    item* where it geometrically falls inside the tile; every other AOI tile
    is scanned with SAM3's text-only prompt (hardcoded to "crop" on the
    geoops-service side for now) — text prompting needs no image-local
    coordinates, so it works across every tile regardless of source image.

    Working resolution (tile size/overlap) is auto-inferred from the
    references' own real-world object size via
    ``geoops.scale.working_resolution_from_refs`` — dynamic per scan, same
    as ``exemplar-scan``/``auto-annotate`` — not a fixed tier.
    """

    annotation_set_collection_id: UUID
    class_id: UUID = Field(description="References are drawn from this class's existing verified annotations.")
    max_references: int = Field(
        default=5, ge=1, le=10,
        description="How many of the class's existing verified annotations to use for scale "
        "inference and as box exemplars (on their own tile only).",
    )
    dataset_item_ids: list[UUID] = Field(min_length=1, max_length=5, description="The AOI to scan.")
    aoi_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    crop_size_px: int = Field(default=256, ge=64, le=1024)
    overlap_ratio: float | None = Field(
        default=None, gt=0.0, lt=1.0,
        description="Fraction of tile width/height consecutive tiles overlap by. Defaults to "
        "the reference object's own pixel footprint (at least one object width), clamped to "
        "[0.2, 0.6] — same default as exemplar-scan.",
    )
    conf: float = Field(default=0.3, ge=0.0, le=1.0, description="SAM3 confidence threshold.")
    iou_thresh: float = Field(
        default=0.3, ge=0.0, le=1.0, description="NMS threshold for merging duplicate detections from overlapping tiles."
    )
    top_k: int = Field(default=50, ge=1, le=500)
    max_patches: int = Field(
        default=200, ge=1, le=400, description="Cap on AOI tiles scanned — one SAM3 call per tile.",
    )
    output_class_id: UUID | None = Field(
        default=None,
        description="Annotation class results are tagged under — defaults to class_id (same class) "
        "when omitted, so results extend the same class's annotation set.",
    )

    @model_validator(mode="after")
    def validate_aoi_bbox(self):
        if self.aoi_bbox is not None:
            _validate_bbox(self.aoi_bbox)
        return self


class Sam3AutoAnnotateMatch(ORMModel):
    dataset_item_id: UUID
    bbox: list[float]
    score: float
    used_box_exemplar: bool = Field(
        description="True if this tile got a reference bbox exemplar (same dataset item, reference "
        "geometrically inside the tile); False if scanned with SAM3's text-only prompt."
    )


class Sam3AutoAnnotateResponse(ORMModel):
    """Shape of ``job.config["result"]`` once a sam3_auto_annotate job completes."""

    references_used: int
    meters_per_pixel: float
    object_px: int
    overlap_ratio: float
    patches_scanned: int
    patches_truncated: bool = False
    candidates_before_nms: int
    errors_total: int = 0
    errors_sample: list[dict]
    matches: list[Sam3AutoAnnotateMatch]


class AnomalyDetectionRequest(ORMModel):
    """Async anomaly scan over one annotation class (runs as a Celery job —
    see ``POST /embeddings/anomaly-scan``, which returns a ``job_id`` to poll).

    Only embeds annotations that already exist (created by a model run) —
    unlike ``aoi-scan``, this never tiles empty space. "Anomalous" means far
    (by cosine distance) from the centroid of its own scale-tier group, so a
    handful of annotations that look nothing like the rest of that class in
    this AOI surface first.
    """

    class_id: UUID
    model_id: UUID = Field(description="Embedding model used to embed each annotation.")
    dataset_item_ids: list[UUID] = Field(min_length=1, max_length=5)
    aoi_bbox: list[float] | None = Field(
        default=None, min_length=4, max_length=4,
        description="Optional further clip within the selected dataset items.",
    )
    min_group_size: int = Field(
        default=3, ge=2, le=50,
        description="A scale-tier group smaller than this is reported but not scored — "
        "distance-to-centroid isn't meaningful with too few peers.",
    )
    top_k: int = Field(default=20, ge=1, le=200)
    max_new_embeddings: int = Field(
        default=300, ge=1, le=1000,
        description="Cap on annotations requiring a fresh embed call — annotations already "
        "embedded by this model (from a prior run) don't count against this.",
    )
    output_class_id: UUID = Field(
        description="Annotation class flagged anomalies (top_k) are persisted under, as a new "
        "annotation_set (source_type='analysis') mounted as a map layer once the job "
        "completes. Independent of `class_id` above — that's the class being *scanned*, "
        "this is the class the *results* get tagged with, and the two may differ "
        "(e.g. scan 'palm_tree' detections, tag anomalies as a generic 'review_flag' class).",
    )

    @model_validator(mode="after")
    def validate_aoi_bbox(self):
        if self.aoi_bbox is not None:
            _validate_bbox(self.aoi_bbox)
        return self


class AnomalyGroup(ORMModel):
    tile_tier: int
    group_size: int
    scored: bool = Field(description="False when group_size < min_group_size — reported, not scored.")


class AnomalyMatch(ORMModel):
    annotation_id: UUID
    dataset_item_id: UUID
    tile_tier: int
    bbox: list[float]
    anomaly_score: float = Field(description="Cosine distance to its group's centroid. Higher = more anomalous.")
    similarity_to_group: float = Field(description="1 - anomaly_score.")


class AnomalyDetectionResponse(ORMModel):
    """Shape of ``job.config["result"]`` once an anomaly-detection job completes."""

    class_id: UUID
    model_name: str
    annotations_scanned: int
    annotations_cached: int
    annotations_embedded: int
    annotations_truncated: bool = False
    errors_total: int = 0
    groups: list[AnomalyGroup]
    anomalies: list[AnomalyMatch]


# ── SAM 3 interactive point mask (Interactive annotation mode) ───────────────


class Sam3ClickPoint(ORMModel):
    lat: float = Field(ge=-90, le=90)
    lng: float = Field(ge=-180, le=180)
    label: int = Field(description="1 = positive (part of the object), 0 = negative (not the object)")

    @model_validator(mode="after")
    def _label(self):
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")
        return self


class Sam3PointMaskRequest(ORMModel):
    """One object's mask from clicks, on a window of one dataset item's imagery.

    The window (``window_bbox``, EPSG:4326) is the context SAM sees; the UI
    keeps it fixed for all clicks of one object, so refinement clicks reuse the
    cached image embedding. It is rendered with the same band/rescale recipe the
    Train Head tiles use, so what is segmented matches what is trained on.
    """

    dataset_id: UUID
    stac_item_id: str
    window_bbox: list[float] = Field(min_length=4, max_length=4, description="[west, south, east, north]")
    points: list[Sam3ClickPoint] = Field(min_length=1, max_length=30)
    max_px: int = Field(default=1024, ge=256, le=2048, description="Longest side of the rendered window, pixels.")

    @model_validator(mode="after")
    def _checks(self):
        w, s, e, n = self.window_bbox
        if not (w < e and s < n):
            raise ValueError("window_bbox must be [west, south, east, north] with west < east and south < north")
        if not any(p.label == 1 for p in self.points):
            raise ValueError("at least one positive point is required")
        return self


class Sam3PointMaskResponse(ORMModel):
    geometry: dict[str, Any] = Field(description="GeoJSON Polygon (EPSG:4326) of the object.")
    score: float | None = None
    span_m: float = Field(description="Longer side of the mask's bounding box, metres.")
    area_m2: float


# ── SAM 3 "Segment all" (Interactive annotation mode) ────────────────────────


class Sam3SegmentAllRequest(ORMModel):
    """Propose every object in a small AOI at one or more real-world scales (async job)."""

    dataset_id: UUID
    stac_item_id: str
    aoi_bbox: list[float] = Field(min_length=4, max_length=4, description="[west, south, east, north]")
    aoi_geometry: dict[str, Any] | None = Field(default=None, description="GeoJSON Polygon of the AOI; defaults to its bbox.")
    scales_m: list[float] = Field(min_length=1, max_length=6, description="Tile edge lengths (metres), 2-2560.")
    points_per_side: int = Field(default=16, ge=8, le=32, description="Point-prompt grid per tile side.")
    min_score: float = Field(default=0.5, ge=0.2, le=0.95, description="Minimum SAM mask confidence.")
    nms_iou: float = Field(default=0.5, ge=0.2, le=0.9, description="IoU above which overlapping masks are merged (existing NMS).")
    reference_geometries: list[dict[str, Any]] = Field(
        default_factory=list, max_length=50,
        description="The user's reference objects: they win any overlap, so auto masks re-finding them are dropped.",
    )

    @model_validator(mode="after")
    def _checks(self):
        w, s, e, n = self.aoi_bbox
        if not (w < e and s < n):
            raise ValueError("aoi_bbox must be [west, south, east, north] with west < east and south < north")
        if any(not (2 <= v <= 2560) for v in self.scales_m):
            raise ValueError("scales_m values must be between 2 and 2560 metres")
        self.scales_m = sorted({round(float(v), 2) for v in self.scales_m})
        return self


class Sam3SegmentAllMask(ORMModel):
    scale_m: float
    geometry: dict[str, Any]
    score: float


class Sam3SegmentAllMasks(ORMModel):
    job_id: UUID
    scales_done: list[float]
    masks: list[Sam3SegmentAllMask]


# ── Mask grouping: embeddings, clusters, similarity (Interactive annotation mode) ──


class Sam3EmbedMask(ORMModel):
    id: str = Field(min_length=1, max_length=64)
    geometry: dict[str, Any]


class Sam3MaskEmbedRequest(ORMModel):
    """Embed proposed masks with DINOv2 so they can be grouped by look (async job)."""

    dataset_id: UUID
    stac_item_id: str
    masks: list[Sam3EmbedMask] = Field(min_length=1, max_length=6000)
    mode: Literal["mask", "box"] = Field(
        default="mask",
        description="mask = the object alone (background greyed out); box = its box with surrounding context.",
    )


class Sam3RefineSetRequest(ORMModel):
    """Turn an annotation set of boxes into masks with SAM 3 (async job). Writes a NEW set."""

    annotation_set_id: UUID
    map_id: UUID | None = Field(default=None, description="Mount the new set on this map (and link it to its project).")
    min_score: float = Field(default=0.5, ge=0.0, le=0.99, description="Only boxes with at least this detection score.")
    scale_m: float = Field(default=10.0, ge=2, le=2560, description="Tile edge (m) used to group boxes; the human masks use 10 m.")
    name: str | None = Field(default=None, max_length=255)


class Sam3CropRequest(ORMModel):
    """Imagery thumbnails for specific masks (with the mask outlined), no embedding needed."""

    dataset_id: UUID
    stac_item_id: str
    masks: list[Sam3EmbedMask] = Field(min_length=1, max_length=60)


class Sam3ClusterRequest(ORMModel):
    n_clusters: int = Field(ge=1, le=60)
    mask_ids: list[str] | None = Field(
        default=None, max_length=6000,
        description="Cluster only these masks (e.g. to split a selection by look). Omit for every embedded mask.",
    )


class Sam3ClusterResponse(ORMModel):
    job_id: UUID
    n_clusters: int
    sizes: list[int]
    labels: dict[str, int]
    outlier_z: dict[str, float]


class Sam3SimilarRequest(ORMModel):
    query_ids: list[str] = Field(min_length=1, max_length=100)


class Sam3SimilarResponse(ORMModel):
    job_id: UUID
    similarity: dict[str, float]


class Sam3ThumbsRequest(ORMModel):
    ids: list[str] = Field(min_length=1, max_length=300)


class Sam3ThumbsResponse(ORMModel):
    thumbs: dict[str, str] = Field(description="mask id -> base64 JPEG thumbnail")


# ── Interactive annotation: commit the verified masks ────────────────────────


class Sam3CommitMask(ORMModel):
    geometry: dict[str, Any] = Field(description="GeoJSON Polygon (EPSG:4326).")
    class_id: UUID
    score: float | None = Field(default=None, ge=0, le=1)
    scale_m: float | None = None


class Sam3CommitRequest(ORMModel):
    """Save an Interactive annotation draft as a verified annotation set (one transaction)."""

    map_id: UUID
    aoi_id: UUID
    schema_id: UUID
    dataset_id: UUID
    stac_item_id: str
    name: str = Field(min_length=1, max_length=255)
    masks: list[Sam3CommitMask] = Field(min_length=1, max_length=6000)


class Sam3CommitResponse(ORMModel):
    annotation_set_id: UUID
    annotation_count: int
    class_counts: dict[str, int] = Field(description="class_id -> annotations saved")


# ── Interactive annotation: clean an existing annotation set ─────────────────


class Sam3ExistingAnnotation(ORMModel):
    id: UUID
    class_id: UUID
    geometry: dict[str, Any]
    confidence: float | None = None
    properties: dict[str, Any] | None = None


class Sam3ExistingResponse(ORMModel):
    schema_id: UUID | None
    dataset_id: UUID | None
    total: int
    annotations: list[Sam3ExistingAnnotation]


class Sam3ApplyUpdate(ORMModel):
    id: UUID
    class_id: UUID | None = None
    geometry: dict[str, Any] | None = Field(default=None, description="GeoJSON Polygon (EPSG:4326).")


class Sam3ApplyRequest(ORMModel):
    """Add / update / delete annotations of an existing set in one transaction."""

    map_id: UUID
    aoi_id: UUID
    dataset_id: UUID
    stac_item_id: str
    add: list[Sam3CommitMask] = Field(default_factory=list, max_length=6000)
    update: list[Sam3ApplyUpdate] = Field(default_factory=list, max_length=6000)
    delete_ids: list[UUID] = Field(default_factory=list, max_length=20000)


class Sam3ApplyResponse(ORMModel):
    annotation_set_id: UUID
    added: int
    updated: int
    deleted: int
    review_status: str


# ── Interactive annotation: second opinion on existing masks ─────────────────


class Sam3RefineMask(ORMModel):
    id: str = Field(max_length=100)
    geometry: dict[str, Any] = Field(description="GeoJSON Polygon (EPSG:4326).")


class Sam3RefineMasksRequest(ORMModel):
    """Ask SAM 3 for each mask's object from 1–2 points inside it and union the result in."""

    dataset_id: UUID
    stac_item_id: str
    masks: list[Sam3RefineMask] = Field(min_length=1, max_length=120)
    min_overlap: float = Field(default=0.3, ge=0, le=1, description="Share of the original that SAM's mask must cover.")
    max_growth: float = Field(default=4.0, ge=1, le=20, description="SAM mask area limit, as a multiple of the original.")
    mode: Literal["replace", "union"] = Field(default="replace", description="replace = use SAM's mask; union = merge it with the original.")


class Sam3RefinedMask(ORMModel):
    id: str
    changed: bool
    geometry: dict[str, Any] | None = None
    growth: float = 0.0
    reason: str | None = None


class Sam3RefineMasksResponse(ORMModel):
    results: list[Sam3RefinedMask]
