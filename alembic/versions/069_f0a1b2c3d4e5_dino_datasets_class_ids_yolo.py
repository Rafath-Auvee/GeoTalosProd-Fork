"""dino_datasets: several classes per dataset, and the 'yolo' task (multi-class detector).

``class_ids`` is NULL for the one-class datasets; ``class_id`` stays the first class.

Revision ID: f0a1b2c3d4e5
Revises: e9f0a1b2c3d4
Create Date: 2026-10-12
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "f0a1b2c3d4e5"
down_revision = "e9f0a1b2c3d4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("dino_datasets", sa.Column("class_ids", JSONB, nullable=True))
    op.drop_constraint("ck_dino_datasets_task", "dino_datasets", type_="check")
    op.create_check_constraint("ck_dino_datasets_task", "dino_datasets", "task IN ('detect', 'classify', 'segment', 'yolo')")


def downgrade() -> None:
    op.drop_constraint("ck_dino_datasets_task", "dino_datasets", type_="check")
    op.create_check_constraint("ck_dino_datasets_task", "dino_datasets", "task IN ('detect', 'classify', 'segment')")
    op.drop_column("dino_datasets", "class_ids")
