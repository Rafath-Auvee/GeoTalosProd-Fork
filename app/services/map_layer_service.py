import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.exceptions import bad_request, conflict, not_found
from app.models.annotation_set import AnnotationSet
from app.models.map import Map
from app.models.map_annotation_set import MapAnnotationSet
from app.models.map_layer import MapLayer
from app.models.project import Project
from app.schemas.map_layer import MapLayerCreate, MapLayerUpdate

logger = logging.getLogger(__name__)


class MapLayerService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def _get_map_for_org(self, map_id: UUID, organization_id: UUID) -> Map:
        """Verify the map exists and belongs to the org. Raises 404 otherwise."""
        result = await self.db.execute(
            select(Map)
            .join(Project, Project.id == Map.project_id)
            .where(Map.id == map_id, Map.deleted_at.is_(None))
            .where(Project.organization_id == organization_id)
        )
        map_row = result.scalar_one_or_none()
        if map_row is None:
            raise not_found("Map")
        return map_row

    async def list_layers(
        self,
        map_id: UUID,
        organization_id: UUID,
        limit: int,
        offset: int,
    ) -> tuple[Sequence[MapLayer], int]:
        # Verify org owns the map first
        await self._get_map_for_org(map_id, organization_id)

        query = select(MapLayer).where(MapLayer.map_id == map_id)
        count_query = (
            select(func.count()).select_from(MapLayer).where(MapLayer.map_id == map_id)
        )
        rows = await self.db.scalars(
            query.order_by(MapLayer.z_index.asc(), MapLayer.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        total = await self.db.scalar(count_query)
        return rows.all(), int(total or 0)

    async def get_layer(
        self, map_id: UUID, layer_id: UUID, organization_id: UUID
    ) -> MapLayer:
        await self._get_map_for_org(map_id, organization_id)
        result = await self.db.execute(
            select(MapLayer).where(
                MapLayer.id == layer_id, MapLayer.map_id == map_id
            )
        )
        layer = result.scalar_one_or_none()
        if layer is None:
            raise not_found("MapLayer")
        return layer

    async def _find_existing_layer(self, map_id: UUID, payload: MapLayerCreate) -> MapLayer | None:
        """Look up an existing row for this map + logical source, matching
        the three source types the DB now enforces uniqueness for (see
        migration 062). dataset/stac_item/annotation_set each have a clean
        single-column identity; tile_service is left unchecked (looser,
        nullable-url-or-source-id identity, no observed duplication).
        """
        if payload.source_type == "dataset" and payload.dataset_id is not None:
            where = (MapLayer.source_type == "dataset", MapLayer.dataset_id == payload.dataset_id)
        elif payload.source_type == "stac_item" and payload.stac_item_id is not None:
            where = (MapLayer.source_type == "stac_item", MapLayer.stac_item_id == payload.stac_item_id)
        elif payload.source_type == "annotation_set" and payload.annotation_set_id is not None:
            where = (MapLayer.source_type == "annotation_set", MapLayer.annotation_set_id == payload.annotation_set_id)
        else:
            return None
        result = await self.db.execute(select(MapLayer).where(MapLayer.map_id == map_id, *where))
        return result.scalar_one_or_none()

    async def create_layer(
        self, map_id: UUID, organization_id: UUID, payload: MapLayerCreate
    ) -> MapLayer:
        await self._get_map_for_org(map_id, organization_id)

        # Idempotent: a caller retrying after a dropped response, or a
        # frontend sync effect whose in-memory "already persisted" tracking
        # reset on reload (the actual root cause of the map_layers
        # duplication this guards against — see migration 062), should get
        # the existing row back rather than a duplicate. Treat it as an
        # upsert of the display-state fields the frontend actually owns.
        existing = await self._find_existing_layer(map_id, payload)
        if existing is not None:
            existing.visible = payload.visible
            existing.opacity = payload.opacity
            existing.z_index = payload.z_index
            await self.db.commit()
            await self.db.refresh(existing)
            logger.info("create_layer_idempotent_hit layer_id=%s map_id=%s", existing.id, map_id)
            return existing

        data = payload.model_dump()
        data["map_id"] = map_id

        if payload.source_type == "annotation_set" and payload.annotation_set_id is not None:
            data["source_config"] = await self._build_annotation_set_source_config(
                payload.annotation_set_id, organization_id, data.get("source_config")
            )

        layer = MapLayer(**data)
        self.db.add(layer)
        try:
            await self.db.commit()
        except IntegrityError as exc:
            await self.db.rollback()
            logger.warning("create_layer_conflict map_id=%s error=%s", map_id, exc)
            raise conflict("Layer creation violates a constraint (check annotation_set_id / dataset_id FK)") from exc
        await self.db.refresh(layer)
        logger.info("create_layer_success layer_id=%s map_id=%s", layer.id, map_id)
        return layer

    async def _build_annotation_set_source_config(
        self, annotation_set_id: UUID, organization_id: UUID, existing_config: dict | None
    ) -> dict:
        """Return source_config for an annotation_set layer.

        For raster annotation sets (those with raster_config populated):
          - Adds tile_url_template pointing to the server-side colormap tile endpoint.
          - Copies colormap so the frontend can render a legend without an extra request.

        For vector annotation sets:
          - Adds geojson_url pointing to the /features endpoint.
        """
        result = await self.db.execute(
            select(AnnotationSet).where(
                AnnotationSet.id == annotation_set_id,
                AnnotationSet.organization_id == organization_id,
                AnnotationSet.deleted_at.is_(None),
            )
        )
        ann_set = result.scalar_one_or_none()
        if ann_set is None:
            raise not_found("AnnotationSet")

        config: dict = dict(existing_config or {})
        base_url = settings.PUBLIC_API_URL.rstrip("/")

        if ann_set.raster_config:
            # Raster mask: tile URL uses the server-side colormap endpoint (no auth header required)
            config["tile_url_template"] = (
                f"{base_url}/api/v1/tiles/raster-masks/{annotation_set_id}/{{z}}/{{x}}/{{y}}.png"
            )
            config["colormap"] = ann_set.raster_config.get("colormap", {})
            config["band_index"] = ann_set.raster_config.get("band_index", 1)
            config["layer_kind"] = "raster_mask"
        else:
            # Vector annotation set: frontend fetches GeoJSON from the features endpoint
            config["geojson_url"] = f"{base_url}/api/v1/annotation-sets/{annotation_set_id}/features"
            config["layer_kind"] = "vector"

        return config

    async def update_layer(
        self,
        map_id: UUID,
        layer_id: UUID,
        organization_id: UUID,
        payload: MapLayerUpdate,
    ) -> MapLayer:
        layer = await self.get_layer(map_id, layer_id, organization_id)
        data = payload.model_dump(exclude_unset=True)
        if not data:
            raise bad_request("No fields to update")
        for key, value in data.items():
            setattr(layer, key, value)
        try:
            await self.db.commit()
        except IntegrityError as exc:
            await self.db.rollback()
            raise conflict("Layer update violates a constraint") from exc
        await self.db.refresh(layer)
        logger.info("update_layer_success layer_id=%s", layer_id)
        return layer

    async def delete_layer(
        self, map_id: UUID, layer_id: UUID, organization_id: UUID
    ) -> None:
        layer = await self.get_layer(map_id, layer_id, organization_id)

        # An annotation_set layer's real "is this mounted" record lives in
        # map_annotation_sets, not here — deleting only this row left the
        # mount in place, so the annotation set resurfaced as a "new" local
        # layer on the next page load (MapEditorShell's sync effect) and
        # got re-synced into a brand new map_layers row. Removing the mount
        # in the same transaction is what makes delete actually stick.
        if layer.source_type == "annotation_set" and layer.annotation_set_id is not None:
            mount = await self.db.get(
                MapAnnotationSet, {"map_id": map_id, "annotation_set_id": layer.annotation_set_id}
            )
            if mount is not None:
                await self.db.delete(mount)

        await self.db.delete(layer)
        await self.db.commit()
        logger.info("delete_layer_success layer_id=%s", layer_id)

    async def reorder_layers(
        self, map_id: UUID, organization_id: UUID, layer_ids: list[UUID]
    ) -> list[MapLayer]:
        """Assign z_index 0, 1, 2… based on the caller-supplied ordering.

        All provided layer IDs must belong to this map.  Any layers not
        mentioned in the list keep their existing z_index (they will sort
        after the reordered layers).  Returns layers ordered by new z_index.
        """
        await self._get_map_for_org(map_id, organization_id)

        result = await self.db.execute(
            select(MapLayer).where(MapLayer.map_id == map_id)
        )
        all_layers = {layer.id: layer for layer in result.scalars().all()}

        unknown = [lid for lid in layer_ids if lid not in all_layers]
        if unknown:
            raise bad_request(
                f"Layer IDs not found on this map: {[str(u) for u in unknown]}"
            )

        for z, lid in enumerate(layer_ids):
            all_layers[lid].z_index = z

        await self.db.commit()
        logger.info("reorder_layers_success map_id=%s count=%s", map_id, len(layer_ids))

        # Refresh to pick up server-side updated_at (onupdate=func.now() expires it after commit)
        for lid in layer_ids:
            await self.db.refresh(all_layers[lid])

        return sorted([all_layers[lid] for lid in layer_ids], key=lambda layer: layer.z_index)
