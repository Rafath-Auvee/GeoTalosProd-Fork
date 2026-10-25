"""Embedding bank ORM model.

One row = one embedding vector for one bbox/mask selection on one dataset
item, produced by one embedding model. ``model_name`` + ``embedding_dim`` are
denormalized from the embed model's own response (not just looked up off
``ai_models.id``) so a bank entry stays self-describing even if the
``ai_models`` row is later edited or deleted, and so every similarity query
can scope its comparison set to "same model, same dimension" without a join.

The ``embedding`` column is an *unconstrained* pgvector ``vector`` (no fixed
dimension at the column level) — different embedding models legitimately
return different dimensions, and we never compare across models anyway.
"""
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Embedding(Base):
    __tablename__ = "embeddings"
    __table_args__ = (
        Index("idx_embeddings_org_model", "organization_id", "model_name"),
        Index("idx_embeddings_dataset_item", "dataset_item_id"),
        Index("idx_embeddings_class", "class_id"),
        Index("idx_embeddings_annotation", "annotation_id"),
        Index("idx_embeddings_org_tier", "organization_id", "model_name", "tile_tier"),
        # Cache key for annotation-tied embeddings (manual creates and the
        # anomaly-detection job both check this before embedding): one
        # annotation only ever needs one embedding per model. Partial so it
        # doesn't constrain the many rows with annotation_id IS NULL (plain
        # bbox selections with no saved annotation).
        Index(
            "uq_embeddings_annotation_model", "annotation_id", "model_id",
            unique=True, postgresql_where=text("annotation_id IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ai_models.id", ondelete="SET NULL"), nullable=True
    )
    # Denormalized from the embed endpoint's own response — see module docstring.
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_items.id", ondelete="CASCADE"), nullable=False
    )
    class_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("annotation_classes.id", ondelete="SET NULL"), nullable=True
    )
    annotation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("annotations.id", ondelete="SET NULL"), nullable=True
    )
    # GeoJSON geometry of the bbox/mask selection the embedding was computed
    # from, in EPSG:4326 — mirrors DatasetItem.geometry's plain-JSONB choice
    # rather than a PostGIS column, since nothing here needs spatial SQL on it.
    source_geometry: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Real-world scale tier of source_geometry's bbox (see geoops/scale.py).
    # search() always filters candidates to the reference's own tier, so a
    # 5m object and a 500m object never rank as "similar" purely because
    # they were both squeezed into the same crop_size_px pixel grid.
    tile_tier: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # Set when a batch job (e.g. anomaly detection) embedded this annotation
    # rather than a user clicking "embed" one at a time. Mutually informative
    # with created_by_user_id, not mutually exclusive by constraint — a row
    # can in principle have neither (e.g. future non-job automation).
    created_by_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    organization: Mapped["Organization"] = relationship("Organization")
    model: Mapped["AIModel | None"] = relationship("AIModel")
    dataset_item: Mapped["DatasetItem"] = relationship("DatasetItem")
    cls: Mapped["AnnotationClass | None"] = relationship("AnnotationClass")
    annotation: Mapped["Annotation | None"] = relationship("Annotation")


class EmbeddingTile(Base):
    """Cache of grid-aligned patch embeddings produced by AOI scans.

    Deliberately separate from ``Embedding`` (the curated bank behind
    ``search()``): every row here is auto-generated coverage from a scan, not
    something a user or annotation chose, and there can be far more of them
    than curated embeddings. Keeping them apart means ``search()`` never has
    to filter out scan noise, and a scan can be re-run over an overlapping
    AOI without re-embedding tiles it already has — the unique constraint on
    (dataset_item_id, model_id, tile_tier, tile_col, tile_row) is the cache
    key.
    """

    __tablename__ = "embedding_tiles"
    __table_args__ = (
        UniqueConstraint(
            "dataset_item_id", "model_id", "tile_tier", "tile_col", "tile_row",
            name="uq_embedding_tiles_grid",
        ),
        Index("idx_embedding_tiles_org_model", "organization_id", "model_name"),
        Index("idx_embedding_tiles_dataset_item", "dataset_item_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ai_models.id", ondelete="SET NULL"), nullable=True
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_items.id", ondelete="CASCADE"), nullable=False
    )
    tile_tier: Mapped[int] = mapped_column(Integer, nullable=False)
    tile_col: Mapped[int] = mapped_column(Integer, nullable=False)
    tile_row: Mapped[int] = mapped_column(Integer, nullable=False)
    # GeoJSON bbox of this tile, EPSG:4326 — same plain-JSONB choice as
    # Embedding.source_geometry.
    bbox: Mapped[dict] = mapped_column(JSONB, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    created_by_job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    organization: Mapped["Organization"] = relationship("Organization")
    model: Mapped["AIModel | None"] = relationship("AIModel")
    dataset_item: Mapped["DatasetItem"] = relationship("DatasetItem")
    created_by_job: Mapped["Job"] = relationship("Job")


class DinoDataset(Base):
    """A reusable, inspectable training dataset for one DINOv2 head: tiles of
    one exhaustively-labeled AOI, already split into train/eval/test.

    Built once (``POST /dino/datasets``), reviewed in the UI, then trained on
    any number of times (``POST /dino/finetune``) without being regenerated.
    Several datasets may exist for the same collection/class/AOI — e.g. a
    detect and a classify one, or the same task at two tile scales — so
    there is deliberately no uniqueness constraint; a rebuild targets one
    dataset by id and overwrites that dataset only.

    ``params`` holds the build inputs (tile scale, crop size, splits, ...).
    ``data_fingerprint`` hashes the annotation data the last successful build
    used; when the live data no longer hashes to it the dataset is *stale*
    and the user is asked to rebuild before training. A classify dataset
    cloned from a detect one (``source_dataset_id``) reuses the source's tile
    images and only writes its own manifest with 0/1 labels.
    """

    __tablename__ = "dino_datasets"
    __table_args__ = (
        CheckConstraint("task IN ('detect', 'classify', 'segment', 'yolo')", name="ck_dino_datasets_task"),
        CheckConstraint(
            "status IN ('pending', 'building', 'ready', 'failed')", name="ck_dino_datasets_status"
        ),
        Index("idx_dino_datasets_org_scope", "organization_id", "annotation_set_collection_id", "class_id", "aoi_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    annotation_set_collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("annotation_set_collections.id", ondelete="CASCADE"), nullable=False
    )
    class_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("annotation_classes.id", ondelete="CASCADE"), nullable=False
    )
    aoi_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("map_aois.id", ondelete="CASCADE"), nullable=False
    )
    # Optional wider scope (NULL = the original one-AOI / collection-verified-sets behaviour):
    # every AOI the dataset covers (``aoi_id`` is the first) and the exact annotation sets used.
    aoi_ids: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    # Several classes (a multi-class YOLO dataset); ``class_id`` is the first. NULL = the one class.
    class_ids: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    annotation_set_ids: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    task: Mapped[str] = mapped_column(String(20), nullable=False)
    params: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, server_default="pending")
    build_no: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    data_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stats: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    s3_prefix: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dino_datasets.id", ondelete="SET NULL"), nullable=True
    )
    source_build_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class DinoTunedConfig(Base):
    """The Optuna result for one (dataset, model): tuned once, reused by every later run.

    ``model_key`` is the DINOv2 backbone key ('dinov2-vits14') or, for YOLO, the base weights
    the run started from. A row is only valid for the dataset build it was tuned on
    (``dataset_build_no``) — after a rebuild the next auto-tuned run replaces it. ``multirun``
    holds the latest multi-seed report (mean / std per metric, per-seed values) that was run
    with ``best_params``.
    """

    __tablename__ = "dino_tuned_configs"
    __table_args__ = (UniqueConstraint("dataset_id", "model_key", name="uq_dino_tuned_configs_dataset_model"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dino_datasets.id", ondelete="CASCADE"), nullable=False
    )
    dataset_build_no: Mapped[int] = mapped_column(Integer, nullable=False)
    model_key: Mapped[str] = mapped_column(String(255), nullable=False)
    best_params: Mapped[dict] = mapped_column(JSONB, nullable=False)
    tuning: Mapped[dict] = mapped_column(JSONB, nullable=False)
    multirun: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
