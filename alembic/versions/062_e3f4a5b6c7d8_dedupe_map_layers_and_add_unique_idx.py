"""Dedupe duplicate map_layers rows and prevent them from recurring.

Root cause (see diagnostic session): the frontend's annotation-set layer
sync effect (MapEditorShell.tsx) tracks "already persisted to backend"
state only in memory (a useRef Set + a Zustand backendLayerIds map), both
of which reset on every page load. Each reload that rehydrates an already-
mounted annotation set as a local layer looks "new" to that effect, so it
POSTs another map_layers row via MapLayerService.create_layer — which has
never checked for an existing row before inserting. Over repeated sessions
this produced up to 15 duplicate rows for a single logical (map,
annotation_set) layer; 396 of 499 total map_layers rows were duplicates
at diagnosis time, all of them source_type='annotation_set'.

This migration:
1. Collapses each (map_id, source_type, dataset_id/stac_item_id/
   annotation_set_id) duplicate group down to one row — the oldest
   (lowest created_at), since duplicates were created with the same
   default style/z_index/opacity and carry no meaningful per-row
   customization to preserve.
2. Adds partial unique indexes for the three source types that have a
   single clean identity column (dataset, stac_item, annotation_set) so
   the same duplication can't recur even if the application-level guard
   (MapLayerService.create_layer, fixed separately) is ever bypassed.
   tile_service is intentionally left out: its identity is looser
   (tile_service_url OR tile_source_id, nullable+freeform) and no
   duplication was observed there.

Revision ID: e3f4a5b6c7d8
Revises: d2e3f4a5b6c7
Create Date: 2026-10-04
"""

from alembic import op

revision = "e3f4a5b6c7d8"
down_revision = "d2e3f4a5b6c7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Dedupe — keep the oldest row per logical (map, source) key.
    op.execute(
        """
        DELETE FROM map_layers ml
        USING (
            SELECT id,
                   row_number() OVER (
                       PARTITION BY map_id, source_type, dataset_id, stac_item_id, annotation_set_id
                       ORDER BY created_at ASC, id ASC
                   ) AS rn
            FROM map_layers
        ) ranked
        WHERE ml.id = ranked.id AND ranked.rn > 1
        """
    )

    # 2. Prevent recurrence — one row per (map, source) for the three
    # source types with a clean single-column identity.
    op.execute(
        """
        CREATE UNIQUE INDEX uq_map_layers_dataset
        ON map_layers (map_id, dataset_id)
        WHERE source_type = 'dataset'
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_map_layers_stac_item
        ON map_layers (map_id, stac_item_id)
        WHERE source_type = 'stac_item'
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_map_layers_annotation_set
        ON map_layers (map_id, annotation_set_id)
        WHERE source_type = 'annotation_set'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_map_layers_annotation_set")
    op.execute("DROP INDEX IF EXISTS uq_map_layers_stac_item")
    op.execute("DROP INDEX IF EXISTS uq_map_layers_dataset")
    # Deduped rows are not restored on downgrade — this migration only
    # removes data that was already a duplicate of a kept row.
