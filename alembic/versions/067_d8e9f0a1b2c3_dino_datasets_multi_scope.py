"""dino_datasets: optional multi-AOI / explicit annotation-set scope.

``aoi_ids`` and ``annotation_set_ids`` are NULL for datasets built the original
way (one AOI, the collection's verified sets); ``aoi_id`` stays the primary AOI.

Revision ID: d8e9f0a1b2c3
Revises: c7d8e9f0a1b2
Create Date: 2026-10-05
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "d8e9f0a1b2c3"
down_revision = "c7d8e9f0a1b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("dino_datasets", sa.Column("aoi_ids", JSONB, nullable=True))
    op.add_column("dino_datasets", sa.Column("annotation_set_ids", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("dino_datasets", "annotation_set_ids")
    op.drop_column("dino_datasets", "aoi_ids")
