"""dino_tuned_configs: the Optuna result for one (dataset, model), tuned once and reused.

Revision ID: 7c3e91d2b45a
Revises: f0a1b2c3d4e5
Create Date: 2026-10-18
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "7c3e91d2b45a"
down_revision = "f0a1b2c3d4e5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dino_tuned_configs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("organization_id", UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_id", UUID(as_uuid=True), sa.ForeignKey("dino_datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("dataset_build_no", sa.Integer, nullable=False),
        sa.Column("model_key", sa.String(255), nullable=False),
        sa.Column("best_params", JSONB, nullable=False),
        sa.Column("tuning", JSONB, nullable=False),
        sa.Column("multirun", JSONB, nullable=True),
        sa.Column("job_id", UUID(as_uuid=True), sa.ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("dataset_id", "model_key", name="uq_dino_tuned_configs_dataset_model"),
    )
    op.create_index("idx_dino_tuned_configs_org", "dino_tuned_configs", ["organization_id"])

    # Same per-org isolation policy as dino_datasets.
    op.execute("ALTER TABLE dino_tuned_configs ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY dino_tuned_configs_org_isolation ON dino_tuned_configs
        USING (organization_id = NULLIF(current_setting('app.current_org_id', true), '')::uuid)
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS dino_tuned_configs_org_isolation ON dino_tuned_configs")
    op.drop_index("idx_dino_tuned_configs_org", table_name="dino_tuned_configs")
    op.drop_table("dino_tuned_configs")
