"""Add missing index on annotations.annotation_set_id.

annotation_set_id is a foreign key (ON DELETE CASCADE) that's filtered or
joined on in several hot paths — AnnotationService.list_annotations,
AnnotationService's update/delete lookups, and
list_map_aoi_resources' vector_annotation_sets query (the one behind the
AOI panel's "Loading intersecting resources…" state) — but Postgres does
not create an index for a FK automatically, and none existed here. Only
annotations_pkey, idx_annotations_geometry (gist), and FK indexes on
created_by_user_id/created_by_job_id/dataset_item_id existed.

Without it, any query that needs to resolve "annotations in this set" (or,
transitively, "annotations in this org's sets") falls back to a full scan
of the annotations table — and because the planner estimates the GiST
bbox filter as low-selectivity for a large AOI, it was choosing a
sequential scan over idx_annotations_geometry too in that case, scanning
every annotation row in the deployment (not just this org's) before the
join to annotation_sets ever narrows it down. A composite
(annotation_set_id, deleted_at) index lets the org-scoped set of
annotation_set_ids (from annotation_sets) drive the lookup instead,
independent of AOI size.

Revision ID: a5b6c7d8e9f0
Revises: f4a5b6c7d8e9
Create Date: 2026-10-04
"""

from alembic import op

revision = "a5b6c7d8e9f0"
down_revision = "f4a5b6c7d8e9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX idx_annotations_set_id_not_deleted
        ON annotations (annotation_set_id, deleted_at)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_annotations_set_id_not_deleted")
