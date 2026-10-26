"""Create dino_datasets: reusable, inspectable Train Head training datasets.

Splits dataset *building* (tiling an AOI, region-based train/eval/test
assignment, writing tiles to storage) from *training*, so a dataset can be
reviewed, re-scaled and rebuilt on its own, and trained on repeatedly
without being regenerated. See geoops/dino_dataset_tasks.py.

No uniqueness constraint on purpose: one AOI can hold several datasets
(detect vs classify, or the same task at different tile scales).

Revision ID: c7d8e9f0a1b2
Revises: b6c7d8e9f0a1
Create Date: 2026-10-04
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "c7d8e9f0a1b2"
down_revision = "b6c7d8e9f0a1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dino_datasets",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("annotation_set_collection_id", UUID(as_uuid=True), sa.ForeignKey("annotation_set_collections.id", ondelete="CASCADE"), nullable=False),
        sa.Column("class_id", UUID(as_uuid=True), sa.ForeignKey("annotation_classes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("aoi_id", UUID(as_uuid=True), sa.ForeignKey("map_aois.id", ondelete="CASCADE"), nullable=False),
        sa.Column("task", sa.String(20), nullable=False),
        sa.Column("params", JSONB, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("build_no", sa.Integer, nullable=False, server_default="0"),
        sa.Column("data_fingerprint", sa.String(64), nullable=True),
        sa.Column("stats", JSONB, nullable=True),
        sa.Column("s3_prefix", sa.String(512), nullable=True),
        sa.Column("source_dataset_id", UUID(as_uuid=True), sa.ForeignKey("dino_datasets.id", ondelete="SET NULL"), nullable=True),
        sa.Column("source_build_no", sa.Integer, nullable=True),
        sa.Column("job_id", UUID(as_uuid=True), sa.ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("task IN ('detect', 'classify')", name="ck_dino_datasets_task"),
        sa.CheckConstraint("status IN ('pending', 'building', 'ready', 'failed')", name="ck_dino_datasets_status"),
    )
    op.create_index(
        "idx_dino_datasets_org_scope",
        "dino_datasets",
        ["organization_id", "annotation_set_collection_id", "class_id", "aoi_id"],
    )

    # Same per-org isolation policy as embeddings / embedding_tiles.
    op.execute("ALTER TABLE dino_datasets ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY dino_datasets_org_isolation ON dino_datasets
        USING (organization_id = NULLIF(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS dino_datasets_org_isolation ON dino_datasets")
    op.drop_index("idx_dino_datasets_org_scope", table_name="dino_datasets")
    op.drop_table("dino_datasets")
