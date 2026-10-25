"""One-object SAM 3 point segmentation for Interactive annotation mode.

Crops a window of a dataset item with TiTiler (same band/rescale recipe the
Train Head tiles use), sends it plus the click points to geoops-service
``/sam3/point``, and converts the returned pixel contours back to a
georeferenced GeoJSON polygon.

Sync (urllib), like the other geoops services; the API endpoint runs it in a
worker thread.
"""
from __future__ import annotations

import base64
import math
from typing import Any
from urllib.error import HTTPError, URLError

from shapely.geometry import Polygon, mapping

from app.models.dataset_item import DatasetItem

from geoops.dino_tasks import _call_geoops_service
from geoops.export_common import resolve_rescale_params
from geoops.schemas import Sam3PointMaskRequest
from geoops.scale import bbox_coords_span_m
from geoops.yolo_tasks import _crop_dims, _fetch_crop_png

# A window much bigger than a few hundred metres would put a small object at a
# handful of pixels; larger views should zoom in before picking.
MAX_WINDOW_M = 1500.0
_M_PER_DEG_LAT = 110_540.0


class PointMaskError(Exception):
    """User-facing failure (maps to HTTP 422/502 in the API layer)."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def _item_bbox(item: DatasetItem) -> list[float] | None:
    geom = item.geometry
    if not geom:
        return None
    from shapely.geometry import shape  # noqa: PLC0415

    return list(shape(geom).bounds)


def compute_point_mask(item: DatasetItem, req: Sam3PointMaskRequest) -> dict[str, Any]:
    window = [float(v) for v in req.window_bbox]
    if bbox_coords_span_m(window) > MAX_WINDOW_M:
        raise PointMaskError("The view is too wide to pick an object. Zoom in closer to the object and try again.")

    ib = _item_bbox(item)
    if ib and (window[2] <= ib[0] or window[0] >= ib[2] or window[3] <= ib[1] or window[1] >= ib[3]):
        raise PointMaskError("That area isn't covered by the selected imagery.")

    w_deg, h_deg = window[2] - window[0], window[3] - window[1]
    width, height = _crop_dims(window, req.max_px)

    points: list[list[float]] = []
    labels: list[int] = []
    for p in req.points:
        if not (window[0] <= p.lng <= window[2] and window[1] <= p.lat <= window[3]):
            raise PointMaskError("A click fell outside the picking window. Reset and click the object again.")
        points.append([(p.lng - window[0]) / w_deg * width, (window[3] - p.lat) / h_deg * height])
        labels.append(p.label)

    rescale = resolve_rescale_params(item.stac_collection_id, item.stac_item_id).get("rescale")
    try:
        png = _fetch_crop_png(item, window, width, height, rescale=rescale)
    except (HTTPError, URLError, ValueError, TimeoutError) as exc:
        raise PointMaskError(f"Could not read the imagery for this view: {exc}", 502) from exc

    try:
        result = _call_geoops_service(
            "/sam3/point",
            {"patch_image_base64": base64.b64encode(png).decode("ascii"), "points": points, "labels": labels},
            timeout=180.0,
        )
    except HTTPError as exc:
        raise PointMaskError("SAM 3 rejected that request.", 422 if exc.code == 400 else 502) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise PointMaskError("The SAM 3 service is unavailable. Try again in a moment.", 502) from exc

    rings = result.get("rings") or []
    if not rings:
        raise PointMaskError("SAM 3 found no object at that point. Click the middle of the object.")

    img_w, img_h = result["image_px"]

    def to_geo(ring: list[list[float]]) -> list[tuple[float, float]]:
        return [(window[0] + x / img_w * w_deg, window[3] - y / img_h * h_deg) for x, y in ring]

    polygon = Polygon(to_geo(rings[0]), [to_geo(r) for r in rings[1:]])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise PointMaskError("SAM 3 returned an unusable outline. Try another click.")
    if polygon.geom_type != "Polygon":  # buffer(0) can split a bow-tie: keep the biggest piece
        polygon = max(polygon.geoms, key=lambda g: g.area)

    minx, miny, maxx, maxy = polygon.bounds
    return {
        "geometry": mapping(polygon),
        "score": result.get("score"),
        "span_m": round(bbox_coords_span_m([minx, miny, maxx, maxy]), 3),
        "area_m2": round(_area_m2(polygon), 3),
    }


def _area_m2(polygon: Polygon) -> float:
    """Area in m² with a local equirectangular scaling — accurate at object scale."""
    cy = (polygon.bounds[1] + polygon.bounds[3]) / 2
    kx = 111_320.0 * math.cos(math.radians(cy))
    ky = _M_PER_DEG_LAT
    return abs(Polygon([(x * kx, y * ky) for x, y in polygon.exterior.coords]).area) - sum(
        abs(Polygon([(x * kx, y * ky) for x, y in r.coords]).area) for r in polygon.interiors
    )
