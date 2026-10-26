"""Update jobs type check to include exemplar_scan.

Adds ``exemplar_scan`` (the multi-reference similarity-scan job in
geoops.exemplar_tasks) to the ``ck_jobs_type`` check constraint. Rebuilt from
the ``JobType`` enum so it stays in sync with the code.

Revision ID: c1d2e3f4a5b6
Revises: b0c1d2e3f4a5
Create Date: 2026-09-27
"""

from alembic import op

from app.core.enums import JobType

revision = "c1d2e3f4a5b6"
down_revision = "b0c1d2e3f4a5"
branch_labels = None
depends_on = None


_TYPE_CHECK = "type IN ({})".format(", ".join(f"'{t}'" for t in JobType))
# Previous set (pre exemplar_scan), for downgrade.
_PRIOR_CHECK = "type IN ({})".format(
    ", ".join(f"'{t}'" for t in JobType if t != JobType.EXEMPLAR_SCAN)
)


def upgrade() -> None:
    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _TYPE_CHECK)


def downgrade() -> None:
    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _PRIOR_CHECK)
