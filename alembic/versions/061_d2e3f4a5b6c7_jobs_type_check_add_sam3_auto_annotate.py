"""Update jobs type check to include sam3_auto_annotate.

Adds ``sam3_auto_annotate`` (the direct SAM3 box+text auto-annotate job in
geoops.sam3_tasks) to the ``ck_jobs_type`` check constraint. Rebuilt from
the ``JobType`` enum so it stays in sync with the code.

Revision ID: d2e3f4a5b6c7
Revises: c1d2e3f4a5b6
Create Date: 2026-09-28
"""

from alembic import op

from app.core.enums import JobType

revision = "d2e3f4a5b6c7"
down_revision = "c1d2e3f4a5b6"
branch_labels = None
depends_on = None


_TYPE_CHECK = "type IN ({})".format(", ".join(f"'{t}'" for t in JobType))
# Previous set (pre sam3_auto_annotate), for downgrade.
_PRIOR_CHECK = "type IN ({})".format(
    ", ".join(f"'{t}'" for t in JobType if t != JobType.SAM3_AUTO_ANNOTATE)
)


def upgrade() -> None:
    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _TYPE_CHECK)


def downgrade() -> None:
    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _PRIOR_CHECK)
