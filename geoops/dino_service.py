"""DINOv2-tier training-data export: validation + manifest building only.

Pure DB-read logic — no TiTiler calls, no S3 writes, no torch/transformers
import. Scoped to ONE class per call (unlike geoops.yolo_service, which is
collection-wide/multi-class) because this tier trains one head per class.

Reuses the pure geometry_to_yolo_bbox/geometry_to_yolo_polygon conversion
math from geoops.yolo_service — that math is generic normalized-coordinate
conversion, not YOLO-label-format specific; only the serialization (YOLO
.txt lines vs this tier's JSON manifest) differs.
"""
from __future__ import annotations

import json
from datetime import UTC
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import not_found
from app.models.annotation import Annotation
from app.models.annotation_class import AnnotationClass
from app.models.annotation_schema import AnnotationSchema
from app.models.annotation_set import AnnotationSet
from app.models.annotation_set_collection import AnnotationSetCollection
from app.models.dataset_item import DatasetItem
from app.models.map_aoi import MapAOI

from geoops.dino_schemas import DinoAoiSetRead, DinoExportPreviewResponse
from geoops.dino_dataset import compute_data_fingerprint
from geoops.export_common import partition_member_sets
from geoops.scale import SCALE_LADDER_M, bbox_coords_span_m, home_scale, suggest_scales


def aoi_geom_expr(aoi: MapAOI):
    """The AOI's outline as a PostGIS geometry (its drawn polygon, else its bbox)."""
    if aoi.geometry:
        return func.ST_SetSRID(func.ST_GeomFromGeoJSON(json.dumps(aoi.geometry)), 4326)
    return func.ST_MakeEnvelope(*[float(v) for v in aoi.bbox_4326], 4326)


class DinoExportService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def explicit_sets(
        self, org_id: UUID, collection: AnnotationSetCollection, set_ids: list[UUID]
    ) -> tuple[list[UUID], list]:
        """The user's chosen sets: any review status, but they must be live and share the collection's schema."""
        from geoops.export_common import SkippedSet  # noqa: PLC0415

        sets = (
            await self.session.scalars(
                select(AnnotationSet).where(
                    AnnotationSet.id.in_(set_ids),
                    AnnotationSet.organization_id == org_id,
                    AnnotationSet.deleted_at.is_(None),
                )
            )
        ).all()
        included = [s.id for s in sets if s.schema_id == collection.schema_id]
        skipped = [
            SkippedSet(annotation_set_id=s.id, name=s.name, reason="schema_id mismatch with collection")
            for s in sets if s.schema_id != collection.schema_id
        ]
        return included, skipped

    async def get_aois(self, org_id: UUID, aoi_ids: list[UUID]) -> list[MapAOI]:
        return [await self.get_aoi(org_id, a) for a in aoi_ids]

    async def sets_in_aois(
        self, org_id: UUID, schema_id: UUID, aoi_ids: list[UUID], class_id: UUID | None, class_ids: list[UUID] | None = None
    ) -> list[DinoAoiSetRead]:
        """Annotation sets of this schema that have annotations inside the AOIs (PostGIS intersection),
        with counts per AOI and for the class."""
        found: dict[UUID, dict] = {}
        wanted = class_ids or ([class_id] if class_id else [])
        for aoi in await self.get_aois(org_id, aoi_ids):
            inside = func.ST_Intersects(Annotation.geometry, aoi_geom_expr(aoi))
            rows = (
                await self.session.execute(
                    select(
                        AnnotationSet.id, AnnotationSet.name, AnnotationSet.review_status, AnnotationSet.source_type,
                        func.count(Annotation.id).label("n"),
                        func.count(Annotation.id).filter(func.ST_NPoints(Annotation.geometry) > 6).label("n_mask"),
                        func.count(Annotation.id).filter(Annotation.class_id.in_(wanted)).label("n_class")
                        if wanted else func.count(Annotation.id).label("n_class"),
                    )
                    .select_from(Annotation)
                    .join(AnnotationSet, AnnotationSet.id == Annotation.annotation_set_id)
                    .where(
                        AnnotationSet.organization_id == org_id,
                        AnnotationSet.schema_id == schema_id,
                        AnnotationSet.deleted_at.is_(None),
                        Annotation.deleted_at.is_(None),
                        inside,
                    )
                    .group_by(AnnotationSet.id)
                )
            ).all()
            for sid, name, status, source_type, n, n_mask, n_class in rows:
                e = found.setdefault(sid, {"name": name, "status": status, "source": source_type, "n": 0, "m": 0, "c": 0, "by": {}})
                e["n"] += n
                e["m"] += n_mask
                e["c"] += n_class if wanted else 0
                e["by"][str(aoi.id)] = n
        out = [
            DinoAoiSetRead(
                id=sid, name=e["name"], review_status=e["status"], source_type=e["source"],
                annotations_in_aois=e["n"], mask_annotations=e["m"], class_annotations=e["c"], by_aoi=e["by"],
            )
            for sid, e in found.items()
        ]
        out.sort(key=lambda r: (-r.class_annotations, r.name))
        return out

    async def _get_collection(self, org_id: UUID, collection_id: UUID) -> AnnotationSetCollection:
        collection = await self.session.scalar(
            select(AnnotationSetCollection).where(
                AnnotationSetCollection.id == collection_id,
                AnnotationSetCollection.organization_id == org_id,
                AnnotationSetCollection.deleted_at.is_(None),
            )
        )
        if collection is None:
            raise not_found("AnnotationSetCollection")
        return collection

    async def _get_class(self, org_id: UUID, class_id: UUID, schema_id: UUID) -> AnnotationClass:
        cls = await self.session.scalar(
            select(AnnotationClass)
            .join(AnnotationSchema, AnnotationSchema.id == AnnotationClass.schema_id)
            .where(
                AnnotationClass.id == class_id,
                AnnotationClass.schema_id == schema_id,
                AnnotationSchema.organization_id == org_id,
            )
        )
        if cls is None:
            raise not_found("AnnotationClass (must belong to the collection's schema)")
        return cls

    async def build_preview(
        self, org_id: UUID, collection_id: UUID, class_id: UUID, aoi_id: UUID | None = None,
        aoi_ids: list[UUID] | None = None, set_ids: list[UUID] | None = None, task: str | None = None,
        class_ids: list[UUID] | None = None,
    ) -> DinoExportPreviewResponse:
        collection = await self._get_collection(org_id, collection_id)
        wanted = class_ids or [class_id]
        classes = [await self._get_class(org_id, c, collection.schema_id) for c in wanted]
        class_name = " + ".join(c.name for c in classes)
        if set_ids:
            included_set_ids, skipped = await self.explicit_sets(org_id, collection, set_ids)
        else:
            included_set_ids, skipped = await partition_member_sets(self.session, collection)

        schema = await self.session.get(AnnotationSchema, collection.schema_id)

        if not included_set_ids:
            return DinoExportPreviewResponse(
                annotation_set_collection_id=collection_id,
                class_id=class_id,
                class_name=class_name,
                schema_id=collection.schema_id,
                geometry_types=list(schema.geometry_types) if schema else [],
                verified_sets_included=0,
                sets_skipped=skipped,
                dataset_item_count=0,
                total_annotations=0,
                annotations_skipped_no_item=0,
                annotations_skipped_inactive_item=0,
                ready=False,
            )

        item_id_expr = func.coalesce(Annotation.dataset_item_id, AnnotationSet.dataset_item_id)
        query = (
            select(
                item_id_expr.label("item_id"),
                Annotation.class_id.label("cid"),
                DatasetItem.is_active,
                func.ST_XMin(Annotation.geometry).label("minx"),
                func.ST_YMin(Annotation.geometry).label("miny"),
                func.ST_XMax(Annotation.geometry).label("maxx"),
                func.ST_YMax(Annotation.geometry).label("maxy"),
            )
            .select_from(Annotation)
            .join(AnnotationSet, AnnotationSet.id == Annotation.annotation_set_id)
            .outerjoin(DatasetItem, DatasetItem.id == item_id_expr)
            .where(
                Annotation.annotation_set_id.in_(included_set_ids),
                Annotation.class_id.in_(wanted),
                Annotation.deleted_at.is_(None),
            )
        )
        if task == "segment":
            query = query.where(func.ST_NPoints(Annotation.geometry) > 6)  # masks only; a box has 5 points
        if set_ids:
            # Explicit sets: an annotation belongs to an AOI by where it is, not by a stamp.
            aois = await self.get_aois(org_id, aoi_ids or ([aoi_id] if aoi_id else []))
            if aois:
                query = query.where(or_(*[func.ST_Intersects(Annotation.geometry, aoi_geom_expr(a)) for a in aois]))
        elif aoi_id is not None:
            # Restrict the preview to one tracked AOI's annotations, same
            # aoi_id provenance stamped by annotation_service._resolve_aoi_for_geometry
            # (see geoops.dino_tasks module docstring for why an AOI is required at all).
            query = query.where(Annotation.properties["aoi_id"].astext == str(aoi_id))
        rows = (await self.session.execute(query)).all()

        eligible_items: set[UUID] = set()
        by_class: dict[str, int] = {}
        total_eligible = 0
        skipped_no_item = 0
        skipped_inactive_item = 0
        spans: list[float] = []
        for row in rows:
            if row.item_id is None:
                skipped_no_item += 1
                continue
            if not row.is_active:
                skipped_inactive_item += 1
                continue
            total_eligible += 1
            by_class[str(row.cid)] = by_class.get(str(row.cid), 0) + 1
            eligible_items.add(row.item_id)
            if None not in (row.minx, row.miny, row.maxx, row.maxy):
                spans.append(bbox_coords_span_m([row.minx, row.miny, row.maxx, row.maxy]))
        spans.sort()
        median_span = spans[len(spans) // 2] if spans else None
        sizing = suggest_scales(spans) if spans else None

        return DinoExportPreviewResponse(
            annotation_set_collection_id=collection_id,
            class_id=class_id,
            class_name=class_name,
            by_class=by_class,
            schema_id=collection.schema_id,
            geometry_types=list(schema.geometry_types) if schema else [],
            verified_sets_included=len(included_set_ids),
            sets_skipped=skipped,
            dataset_item_count=len(eligible_items),
            total_annotations=total_eligible,
            annotations_skipped_no_item=skipped_no_item,
            annotations_skipped_inactive_item=skipped_inactive_item,
            median_object_span_m=round(median_span, 2) if median_span is not None else None,
            object_size_p10_m=sizing["p10_m"] if sizing else None,
            object_size_p90_m=sizing["p90_m"] if sizing else None,
            size_spread=sizing["spread"] if sizing else None,
            suggested_scales_m=sizing["scales_m"] if sizing else [],
            suggested_tile_scale_m=home_scale(median_span, SCALE_LADDER_M) if median_span else None,
            ready=total_eligible > 0,
        )


    async def get_aoi(self, org_id: UUID, aoi_id: UUID) -> MapAOI:
        aoi = await self.session.scalar(
            select(MapAOI).where(
                MapAOI.id == aoi_id, MapAOI.organization_id == org_id, MapAOI.deleted_at.is_(None)
            )
        )
        if aoi is None:
            raise not_found("MapAOI")
        return aoi

    async def data_fingerprint(
        self, org_id: UUID, collection_id: UUID, class_id: UUID, aoi_id: UUID, task: str,
        aoi_ids: list[UUID] | None = None, set_ids: list[UUID] | None = None, class_ids: list[UUID] | None = None,
    ) -> str:
        """Fingerprint of the live annotation data a dataset build would use.

        Mirrors the row selection in geoops.dino_tasks._gather_rows_for_class
        + _filter_rows_by_aoi (verified member sets, this class, not deleted,
        resolvable + active item, tagged to this AOI) so it compares equal to
        the fingerprint a build stored — compare it to ``DinoDataset.data_fingerprint``
        to decide whether the dataset is stale.
        """
        collection = await self._get_collection(org_id, collection_id)
        aoi = await self.get_aoi(org_id, aoi_id)
        aois = await self.get_aois(org_id, aoi_ids or [aoi_id])
        if set_ids:
            included_set_ids, _skipped = await self.explicit_sets(org_id, collection, set_ids)
        else:
            included_set_ids, _skipped = await partition_member_sets(self.session, collection)

        versions: list[tuple[str, str]] = []
        if included_set_ids:
            item_id_expr = func.coalesce(Annotation.dataset_item_id, AnnotationSet.dataset_item_id)
            query = (
                select(Annotation.id, Annotation.updated_at)
                .select_from(Annotation)
                .join(AnnotationSet, AnnotationSet.id == Annotation.annotation_set_id)
                .join(DatasetItem, DatasetItem.id == item_id_expr)
                .where(
                    Annotation.annotation_set_id.in_(included_set_ids),
                    Annotation.class_id.in_(class_ids or [class_id]),
                    Annotation.deleted_at.is_(None),
                    DatasetItem.is_active.is_(True),
                    or_(*[func.ST_Intersects(Annotation.geometry, aoi_geom_expr(a)) for a in aois])
                    if set_ids else Annotation.properties["aoi_id"].astext == str(aoi_id),
                )
            )
            if task == "segment":
                query = query.where(func.ST_NPoints(Annotation.geometry) > 6)  # the build only uses masks
            versions = [
                (str(aid), updated.astimezone(UTC).isoformat()) for aid, updated in (await self.session.execute(query)).all()
            ]
        return compute_data_fingerprint(
            annotation_versions=versions,
            set_ids=[str(i) for i in included_set_ids],
            # One AOI keeps the original (flat) shape so existing datasets don't turn stale.
            aoi_bbox=[list(a.bbox_4326) for a in aois] if set_ids else list(aoi.bbox_4326),
            aoi_geometry=[a.geometry for a in aois] if set_ids else aoi.geometry,  # type: ignore[arg-type]
            task=task,
        )
