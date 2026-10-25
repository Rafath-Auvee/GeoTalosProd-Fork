"""dino_datasets: allow the 'segment' task (mask training).

Revision ID: e9f0a1b2c3d4
Revises: d8e9f0a1b2c3
Create Date: 2026-10-11
"""

from alembic import op

revision = "e9f0a1b2c3d4"
down_revision = "d8e9f0a1b2c3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("ck_dino_datasets_task", "dino_datasets", type_="check")
    op.create_check_constraint("ck_dino_datasets_task", "dino_datasets", "task IN ('detect', 'classify', 'segment')")


def downgrade() -> None:
    op.drop_constraint("ck_dino_datasets_task", "dino_datasets", type_="check")
    op.create_check_constraint("ck_dino_datasets_task", "dino_datasets", "task IN ('detect', 'classify')")
