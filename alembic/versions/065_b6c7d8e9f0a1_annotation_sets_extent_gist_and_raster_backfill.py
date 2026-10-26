"""Index annotation_sets.extent_4326 and backfill it for raster mask sets.

extent_4326 already exists and is kept current for *vector* annotation
sets by the annotations_extent_aiud trigger (migration 047), but it had
no spatial index, and raster mask sets never populated it at all — raster
masks have no `annotations` rows to fire that trigger. Instead,
raster_config carried a `bounds_4326` key that nothing in the codebase
ever actually wrote (see app/api/v1/endpoints/maps.py's
raster_mask_annotation_sets query and app/automation/nodes/data_source.py,
both of which read a key that was always absent), so raster masks never
actually matched an AOI's bbox filter.

This migration:
1. Adds a GiST index on extent_4326 so both the vector and (now) raster
   paths through list_map_aoi_resources can use the same indexed column.
2. Backfills extent_4326 for existing raster mask annotation sets from
   their linked dataset_item's geometry (the raster file's footprint) —
   the same source configure_raster_mask_for_annotation_set now uses going
   forward for newly-configured raster masks.

Revision ID: b6c7d8e9f0a1
Revises: a5b6c7d8e9f0
Create Date: 2026-10-04
"""

from alembic import op

revision = "b6c7d8e9f0a1"
down_revision = "a5b6c7d8e9f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX idx_annotation_sets_extent_gist
        ON annotation_sets
        USING GIST (extent_4326)
        WHERE extent_4326 IS NOT NULL
        """
    )
    op.execute(
        """
        UPDATE annotation_sets AS a
        SET extent_4326 = ST_SetSRID(
            ST_GeomFromGeoJSON((di.geometry)::text), 4326
        )
        FROM dataset_items AS di
        WHERE a.raster_config IS NOT NULL
          AND a.extent_4326 IS NULL
          AND di.geometry IS NOT NULL
          AND di.id = COALESCE(
              a.dataset_item_id,
              NULLIF(a.raster_config->>'dataset_item_id', '')::uuid
          )
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_annotation_sets_extent_gist")
