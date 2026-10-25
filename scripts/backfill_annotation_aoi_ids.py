#!/usr/bin/env python3
"""Backfill two things older annotations can be missing, both of which make
an otherwise-correct annotation silently invisible to Train Head exports
(geoops.dino_tasks / geoops.dino_service — see AwakeForestProd CLAUDE.md):

1. ``properties.aoi_id`` / ``aoi_name`` — soft AOI provenance, stamped at
   create/verify time by AnnotationService._resolve_aoi_for_geometry. Missing
   for annotations drawn straight into an already-open set before
   app.services.annotation_service.create_annotation learned to stamp it
   (2026-09-28 fix adding its optional map_id parameter).

2. ``dataset_item_id`` — required for _gather_rows_for_class /
   DinoExportService.build_preview to resolve an image for the crop at all;
   an annotation with none is dropped silently (not counted in any "skipped"
   total) before it's even checked for aoi_id. See
   AnnotationService._resolve_dataset_item_via_aoi's docstring: "annotations
   were landing with no linked image ... even when the AOI panel showed a
   dataset already selected" — a real gap for older annotations, fixed
   forward-going for create_annotation_on_map but never backfilled.

Both use the exact same containment/selection logic as the async service
methods, reimplemented here in sync/shapely terms since this runs against
the sync WorkerSession, not the async API session.

Run inside the backend environment (it uses the worker DB role):

    python -m scripts.backfill_annotation_aoi_ids --set-id <uuid>
    python -m scripts.backfill_annotation_aoi_ids --set-name "Verified — Mining Benchmark"
    python -m scripts.backfill_annotation_aoi_ids --map-id <uuid>          # every set on this map
    python -m scripts.backfill_annotation_aoi_ids --set-id <uuid> --dry-run
    python -m scripts.backfill_annotation_aoi_ids --set-id <uuid> --force  # re-resolve even if already set

Prints counts for both backfills. dataset_item_id is only ever filled when
resolution is unambiguous (one candidate item) — genuine ambiguity (several
overlapping items) is left alone rather than guessed, same as the live
service.
"""
from __future__ import annotations

import argparse
import sys
import uuid

from geoalchemy2.shape import to_shape
from shapely.geometry import box as shapely_box, shape as shapely_shape
from sqlalchemy import select

from app.models.annotation import Annotation
from app.models.annotation_set import AnnotationSet
from app.models.dataset_item import DatasetItem
from app.models.map_annotation_set import MapAnnotationSet
from app.models.map_aoi import MapAOI
from app.workers.db import WorkerSession


def _resolve_sets(session, *, set_id: str | None, set_name: str | None, map_id: str | None) -> list[AnnotationSet]:
    if set_id:
        row = session.get(AnnotationSet, uuid.UUID(set_id))
        if row is None:
            sys.exit(f"No annotation set with id {set_id}")
        return [row]
    if set_name:
        rows = session.execute(
            select(AnnotationSet).where(
                AnnotationSet.name == set_name,
                AnnotationSet.deleted_at.is_(None),
            )
        ).scalars().all()
        if not rows:
            sys.exit(f"No annotation set named {set_name!r}")
        return list(rows)
    if map_id:
        rows = session.execute(
            select(AnnotationSet)
            .join(MapAnnotationSet, MapAnnotationSet.annotation_set_id == AnnotationSet.id)
            .where(
                MapAnnotationSet.map_id == uuid.UUID(map_id),
                AnnotationSet.deleted_at.is_(None),
            )
        ).scalars().all()
        if not rows:
            sys.exit(f"No annotation sets mounted on map {map_id}")
        return list(rows)
    sys.exit("Pass one of --set-id, --set-name, or --map-id")


def _aoi_poly(aoi: MapAOI):
    if isinstance(aoi.geometry, dict) and aoi.geometry:
        return shapely_shape(aoi.geometry)
    if isinstance(aoi.bbox_4326, (list, tuple)) and len(aoi.bbox_4326) == 4:
        return shapely_box(*[float(v) for v in aoi.bbox_4326])
    return None


def _aoi_for_point(aois: list[MapAOI], point) -> MapAOI | None:
    for aoi in aois:
        try:
            poly = _aoi_poly(aoi)
            if poly is None:
                continue
            if poly.contains(point) or poly.intersects(point):
                return aoi
        except Exception:  # noqa: BLE001
            continue
    return None


def _dataset_item_via_aoi(session, org_id: uuid.UUID, aois: list[MapAOI], point) -> uuid.UUID | None:
    """Mirrors AnnotationService._resolve_dataset_item_via_aoi (sync)."""
    for aoi in aois:
        try:
            poly = _aoi_poly(aoi)
            if poly is None or not (poly.contains(point) or poly.intersects(point)):
                continue
        except Exception:  # noqa: BLE001
            continue

        selection = aoi.selection_config or {}
        try:
            dataset_item_ids = [uuid.UUID(str(v)) for v in (selection.get("dataset_item_ids") or [])]
            dataset_ids = [uuid.UUID(str(v)) for v in (selection.get("dataset_ids") or [])]
        except (ValueError, TypeError):
            continue

        query = select(DatasetItem).where(
            DatasetItem.organization_id == org_id,
            DatasetItem.is_active.is_(True),
        )
        if dataset_item_ids:
            query = query.where(DatasetItem.id.in_(dataset_item_ids))
        elif dataset_ids:
            query = query.where(DatasetItem.dataset_id.in_(dataset_ids))
        else:
            continue

        items = session.execute(query).scalars().all()
        if not items:
            continue
        if len(items) == 1:
            return items[0].id

        matches = [item for item in items if item.geometry and shapely_shape(item.geometry).contains(point)]
        if len(matches) == 1:
            return matches[0].id
        return None  # multiple overlapping items, no unique match — don't guess

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set-id", help="Backfill this one annotation set")
    parser.add_argument("--set-name", help="Backfill the annotation set(s) with this exact name")
    parser.add_argument("--map-id", help="Backfill every annotation set mounted on this map")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-resolve annotations that already carry an aoi_id (e.g. an AOI's geometry changed)",
    )
    args = parser.parse_args()

    with WorkerSession() as session:
        sets = _resolve_sets(session, set_id=args.set_id, set_name=args.set_name, map_id=args.map_id)

        aois_by_map: dict[uuid.UUID, list[MapAOI]] = {}

        aoi_tagged = aoi_already = aoi_no_match = no_map = 0
        item_resolved = item_already = item_ambiguous = item_no_match = 0

        for ann_set in sets:
            map_ids = session.execute(
                select(MapAnnotationSet.map_id).where(MapAnnotationSet.annotation_set_id == ann_set.id)
            ).scalars().all()
            if not map_ids:
                annotations = session.execute(
                    select(Annotation).where(
                        Annotation.annotation_set_id == ann_set.id,
                        Annotation.deleted_at.is_(None),
                    )
                ).scalars().all()
                no_map += len(annotations)
                continue

            for map_id in map_ids:
                if map_id not in aois_by_map:
                    aois_by_map[map_id] = session.execute(
                        select(MapAOI).where(
                            MapAOI.map_id == map_id,
                            MapAOI.deleted_at.is_(None),
                        )
                    ).scalars().all()
                aois = aois_by_map[map_id]
                if not aois:
                    continue

                annotations = session.execute(
                    select(Annotation).where(
                        Annotation.annotation_set_id == ann_set.id,
                        Annotation.deleted_at.is_(None),
                    )
                ).scalars().all()

                for ann in annotations:
                    geom = to_shape(ann.geometry)
                    point = geom.representative_point()

                    # 1. aoi_id / aoi_name provenance
                    props = ann.properties or {}
                    if props.get("aoi_id") and not args.force:
                        aoi_already += 1
                    else:
                        aoi = _aoi_for_point(aois, point)
                        if aoi is None:
                            aoi_no_match += 1
                        else:
                            if not args.dry_run:
                                props = dict(props)
                                props["aoi_id"] = str(aoi.id)
                                props["aoi_name"] = aoi.name
                                ann.properties = props
                            aoi_tagged += 1

                    # 2. dataset_item_id (only if the set itself has none —
                    # same precedence create_annotation_on_map applies)
                    if ann.dataset_item_id is not None or ann_set.dataset_item_id is not None:
                        item_already += 1
                        continue
                    item_id = _dataset_item_via_aoi(session, ann_set.organization_id, aois, point)
                    if item_id is None:
                        # Could be "no candidate item" or "ambiguous" — both
                        # count together here; use --dry-run + a narrower
                        # --set-id to inspect a specific case if needed.
                        item_no_match += 1
                        continue
                    if not args.dry_run:
                        ann.dataset_item_id = item_id
                    item_resolved += 1

        if not args.dry_run:
            session.commit()

        prefix = "[dry-run] " if args.dry_run else ""
        print(
            f"{prefix}aoi_id: tagged={aoi_tagged} already_tagged={aoi_already} "
            f"no_containing_aoi={aoi_no_match} no_map_mounted={no_map}"
        )
        print(
            f"{prefix}dataset_item_id: resolved={item_resolved} already_set={item_already} "
            f"unresolved={item_no_match}"
        )


if __name__ == "__main__":
    main()
