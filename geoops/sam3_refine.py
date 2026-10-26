"""Batch "second opinion" for existing masks in Interactive annotation mode.

For each mask, SAM 3 is asked for the object under 1–2 points placed inside it; its mask replaces the
original (or is unioned with it, mode="union") when it plausibly describes the same object. Masks that fail
the plausibility checks are returned unchanged with a reason, so one bad mask never aborts a batch.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from shapely.ops import polylabel
from shapely.geometry import Polygon, mapping, shape

from app.models.dataset_item import DatasetItem

from geoops.sam3_point_service import PointMaskError, compute_point_mask
from geoops.schemas import Sam3ClickPoint, Sam3PointMaskRequest, Sam3RefineMasksRequest

WORKERS = 4
# The window SAM sees, as a multiple of the mask's longest side: enough context to separate the
# object from its surroundings without shrinking it to a few pixels.
CONTEXT_FACTOR = 3.0
MIN_WINDOW_DEG = 2e-6  # ~0.2 m: floor for tiny masks


def _largest_polygon(geom) -> Polygon | None:
    if geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    return max(polys, key=lambda g: g.area) if polys else None


def pick_points(poly: Polygon) -> list[tuple[float, float]]:
    """One point at the roomiest spot of the mask, plus a second one apart from it when there is room."""
    tol = max(poly.length * 1e-3, 1e-9)
    first = polylabel(poly, tolerance=tol)
    pts = [(first.x, first.y)]
    inradius = poly.exterior.distance(first)
    inner = poly.buffer(-inradius * 0.5)
    if not inner.is_empty:
        cand = inner.representative_point()
        if cand.distance(first) > inradius * 0.3 and poly.contains(cand):
            pts.append((cand.x, cand.y))
    return pts


def _refine_one(item: DatasetItem, req: Sam3RefineMasksRequest, mask_id: str, geometry: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"id": mask_id, "changed": False, "geometry": None, "growth": 0.0, "reason": None}
    try:
        orig = shape(geometry)
        if not orig.is_valid:
            orig = orig.buffer(0)
        orig = _largest_polygon(orig)
        if orig is None or orig.area == 0:
            out["reason"] = "empty mask"
            return out

        w, s, e, n = orig.bounds
        half = max(max(e - w, n - s) * CONTEXT_FACTOR / 2, MIN_WINDOW_DEG)
        cx, cy = (w + e) / 2, (s + n) / 2
        window = [cx - half, cy - half, cx + half, cy + half]
        points = [Sam3ClickPoint(lat=y, lng=x, label=1) for x, y in pick_points(orig)]
        res = compute_point_mask(
            item, Sam3PointMaskRequest(dataset_id=req.dataset_id, stac_item_id=req.stac_item_id, window_bbox=window, points=points, max_px=768)
        )
        sam = _largest_polygon(shape(res["geometry"]))
        if sam is None:
            out["reason"] = "no SAM mask"
            return out

        overlap = orig.intersection(sam).area / orig.area
        if overlap < req.min_overlap:
            out["reason"] = "SAM found a different object"
            return out
        if sam.area > orig.area * req.max_growth:
            out["reason"] = "SAM mask is much larger than the original"
            return out

        merged = _largest_polygon(sam if req.mode == "replace" else orig.union(sam))
        if merged is None:
            out["reason"] = "could not merge"
            return out
        growth = merged.area / orig.area - 1.0
        out["growth"] = round(growth, 4)
        # How different the result is from the original, whichever direction it moved.
        changed_share = merged.symmetric_difference(orig).area / orig.area
        if changed_share > 0.01:  # under 1% is noise: leave the annotation untouched
            out["changed"] = True
            out["geometry"] = mapping(merged)
        else:
            out["reason"] = "already matches SAM"
    except PointMaskError as exc:
        out["reason"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - one bad mask must not fail the batch
        out["reason"] = f"failed: {type(exc).__name__}"
    return out


def refine_masks(item: DatasetItem, req: Sam3RefineMasksRequest) -> list[dict[str, Any]]:
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_refine_one, item, req, m.id, m.geometry) for m in req.masks]
        return [f.result() for f in futures]


__all__ = ["refine_masks", "pick_points"]
