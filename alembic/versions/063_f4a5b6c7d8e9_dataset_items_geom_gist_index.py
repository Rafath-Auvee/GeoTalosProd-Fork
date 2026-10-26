"""Add a GiST expression index for dataset_items.geometry bbox lookups.

dataset_items.geometry is plain JSONB (GeoJSON), not a PostGIS geometry
column (see the comment on the model) — chosen originally to avoid a
PostGIS dependency for simple footprint lookups. That meant
GET /datasets/{id}/items?bbox=... had to load every active item in the
dataset and run a Shapely intersects() check in Python per row, which got
slow as datasets grew (no SQL-level filter at all, full result transfer
over the wire every request).

Rather than migrating the column itself (a larger change touching
ingestion + every other reader of this field as a dict), this follows the
same pattern already used in app/automation/nodes/data_source.py's
execute_aoi_filter: cast the JSONB to text and parse it with
ST_GeomFromGeoJSON() inside the query, then ST_Intersects against the
bbox envelope. ST_GeomFromGeoJSON is IMMUTABLE in PostGIS, so that
expression can be indexed directly — this index is what makes the SQL
version in list_dataset_items (app/api/v1/endpoints/datasets.py) fast
instead of just "no longer transferring every row over the wire."

Revision ID: f4a5b6c7d8e9
Revises: e3f4a5b6c7d8
Create Date: 2026-10-04
"""

from alembic import op

revision = "f4a5b6c7d8e9"
down_revision = "e3f4a5b6c7d8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX idx_dataset_items_geom_gist
        ON dataset_items
        USING GIST (ST_GeomFromGeoJSON(geometry::text))
        WHERE geometry IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_dataset_items_geom_gist")
