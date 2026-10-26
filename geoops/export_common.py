"""Shared AnnotationSetCollection validation logic — the "which member sets
are actually eligible" filtering used by every training-data exporter
(currently geoops.yolo_service, geoops.dino_service), so the multi-tier
architecture doesn't re-derive this once per tier.

Pure DB-read logic (async, request-path). No torch/ultralytics import.
"""
from __future__ import annotations

import hashlib
import json
import logging
from urllib import parse, request as urlrequest
from uuid import UUID

from pydantic import Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.annotation_set import AnnotationSet
from app.models.annotation_set_collection import AnnotationSetCollection
from app.models.annotation_set_collection_item import AnnotationSetCollectionItem
from app.schemas.common import ORMModel

logger = logging.getLogger(__name__)


def stable_bucket(key: str, modulus: int = 100) -> int:
    """Deterministic bucket in [0, modulus) for a string key — used for
    train/eval/test split assignment (geoops.dino_tasks, geoops.yolo_tasks).

    Python's built-in hash() is NOT safe for this: string hashing is
    randomized per-process by default (PYTHONHASHSEED), so the same
    item_id would land in a different split every time a worker restarts.
    sha256 is stable across processes/restarts/machines.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulus


class SkippedSet(ORMModel):
    annotation_set_id: UUID
    name: str
    reason: str = Field(
        description="Why this member set was excluded, e.g. "
        "'review_status=raw (must be verified)' or 'schema_id mismatch with collection'."
    )


async def partition_member_sets(
    session: AsyncSession, collection: AnnotationSetCollection
) -> tuple[list[UUID], list[SkippedSet]]:
    """Split a collection's member sets into (eligible ids, skipped-with-reason).

    Re-validates schema match defensively rather than trusting that every
    link went through annotation_set_collection_service.add_set_to_collection
    (the only write path that enforces it today) — there's no DB-level
    constraint backing that invariant. See geoops/yolo_service.py's original
    design notes.
    """
    member_sets = (
        await session.scalars(
            select(AnnotationSet)
            .join(
                AnnotationSetCollectionItem,
                AnnotationSetCollectionItem.annotation_set_id == AnnotationSet.id,
            )
            .where(
                AnnotationSetCollectionItem.collection_id == collection.id,
                AnnotationSet.deleted_at.is_(None),
            )
        )
    ).all()

    included: list[UUID] = []
    skipped: list[SkippedSet] = []
    for s in member_sets:
        if s.schema_id != collection.schema_id:
            skipped.append(
                SkippedSet(annotation_set_id=s.id, name=s.name, reason="schema_id mismatch with collection")
            )
        elif s.review_status != "verified":
            skipped.append(
                SkippedSet(
                    annotation_set_id=s.id,
                    name=s.name,
                    reason=f"review_status={s.review_status} (must be verified)",
                )
            )
        else:
            included.append(s.id)
    return included, skipped


# Conservative fallback when an item is uint16/int16 but TiTiler's /statistics
# call fails or returns a degenerate (min >= max) range — matches
# app.services.titiler_service.DEFAULT_RESCALE_UINT16, the same constant the
# live map-tile path falls back to.
_DEFAULT_RESCALE_UINT16 = "0,10000"

_rescale_cache: dict[tuple[str, str], dict[str, str]] = {}


def _titiler_get_sync(path: str, params: dict | None = None) -> dict:
    url = f"{settings.TITILER_URL.rstrip('/')}{path}"
    if params:
        url = f"{url}?{parse.urlencode(params)}"
    req = urlrequest.Request(url, method="GET")
    with urlrequest.urlopen(req, timeout=30.0) as resp:  # nosec B310
        return json.loads(resp.read())


def resolve_rescale_params(collection_id: str, item_id: str, asset_name: str = "data") -> dict[str, str]:
    """Sync (Celery-safe, plain urllib) equivalent of
    app.services.titiler_service._get_item_rendering_params — auto-detects
    dtype/colorinterp for a STAC item and, for uint16/int16 data, resolves a
    real rescale range from TiTiler's own per-band p2/p98 statistics.

    Exists because geoops's training-crop exporters (geoops.yolo_tasks's
    _fetch_crop_png, reused by geoops.dino_tasks) run inside sync Celery
    tasks and previously requested crops with NO rescale at all — for a
    uint16/int16 COG, TiTiler then stretches the dtype's full numeric range
    (e.g. -32768..32767) instead of the dataset's actual value range, which
    renders as a near-flat mid-grey image whenever real values cluster far
    from that assumed range (exactly what was observed: every exported
    training/eval/test tile came back as solid RGB 127/128). The live map
    view never had this problem because it already computes a real rescale
    from band stats before rendering — this gives the training-crop
    exporters the same treatment.

    Cached per (collection_id, item_id) within this process: tiles from the
    same source item share one rescale, so this avoids one
    info+statistics round trip per tile when an export's ~75+ tiles usually
    come from a handful of items.
    """
    cache_key = (collection_id, item_id)
    if cache_key in _rescale_cache:
        return _rescale_cache[cache_key]

    params: dict[str, str] = {}
    try:
        info = _titiler_get_sync(
            f"/collections/{collection_id}/items/{item_id}/info", {"assets": asset_name}
        )
        asset_info = info.get(asset_name, {})
        dtype = asset_info.get("dtype", "uint8")
        color_interps = [ci.lower() for ci in asset_info.get("colorinterp", [])]
        is_rgb = color_interps[:3] == ["red", "green", "blue"]
        is_rgba = color_interps[:4] == ["red", "green", "blue", "alpha"]

        if dtype in ("uint16", "int16"):
            try:
                stats = _titiler_get_sync(
                    f"/collections/{collection_id}/items/{item_id}/statistics", {"assets": asset_name}
                )
                render_bands = [1, 2, 3] if (is_rgb or is_rgba) else [1]
                all_min, all_max = float("inf"), float("-inf")
                for bnum in render_bands:
                    bs = stats.get(f"{asset_name}_b{bnum}", {}) or stats.get(asset_name, {}).get(str(bnum), {})
                    bmin = bs.get("percentile_2", bs.get("min", 0))
                    bmax = bs.get("percentile_98", bs.get("max", 10000))
                    all_min, all_max = min(all_min, bmin), max(all_max, bmax)
                params["rescale"] = f"{int(all_min)},{int(all_max)}" if all_min < all_max else _DEFAULT_RESCALE_UINT16
            except Exception as exc:  # noqa: BLE001
                params["rescale"] = _DEFAULT_RESCALE_UINT16
                logger.warning(
                    "resolve_rescale_params_stats_failed collection=%s item=%s error=%s",
                    collection_id, item_id, exc,
                )
            nodata = asset_info.get("nodata_value")
            if nodata is not None:
                params["nodata"] = str(nodata)
    except Exception as exc:  # noqa: BLE001
        logger.warning("resolve_rescale_params_info_failed collection=%s item=%s error=%s", collection_id, item_id, exc)

    _rescale_cache[cache_key] = params
    return params


def partition_member_sets_sync(session, collection: AnnotationSetCollection) -> tuple[list[UUID], list[dict]]:
    """Sync (WorkerSession/psycopg2) equivalent of partition_member_sets, for
    Celery tasks. Returns skipped sets as plain dicts (JSON-serializable
    directly onto job.config), not SkippedSet models, since sync task code
    writes straight into JSONB.
    """
    member_sets = session.scalars(
        select(AnnotationSet)
        .join(
            AnnotationSetCollectionItem,
            AnnotationSetCollectionItem.annotation_set_id == AnnotationSet.id,
        )
        .where(
            AnnotationSetCollectionItem.collection_id == collection.id,
            AnnotationSet.deleted_at.is_(None),
        )
    ).all()

    included: list[UUID] = []
    skipped: list[dict] = []
    for s in member_sets:
        if s.schema_id != collection.schema_id:
            skipped.append(
                {"annotation_set_id": str(s.id), "name": s.name, "reason": "schema_id mismatch with collection"}
            )
        elif s.review_status != "verified":
            skipped.append(
                {
                    "annotation_set_id": str(s.id),
                    "name": s.name,
                    "reason": f"review_status={s.review_status} (must be verified)",
                }
            )
        else:
            included.append(s.id)
    return included, skipped
