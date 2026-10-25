"""Geographic scale helpers for the embedding bank.

Every embedding is a crop of a real-world area squeezed into the same
``crop_size_px`` pixel grid — two crops of very different real-world size
look nothing alike to an embedding model even if the underlying content is
similar (a 5m object and a 500m object both resized to 256x256 are encoding
completely different things). These helpers convert a GeoJSON geometry's
bbox into a real-world span (meters) and snap it to one of a small number of
scale tiers, so ``search()`` only ever compares same-tier candidates and
``aoi_scan`` tiles a dataset item at the same scale as its reference.

``DatasetItem``/``Embedding`` geometries are plain GeoJSON (JSONB), not
PostGIS columns, so this uses a simple equirectangular approximation
(accurate enough for tiling/tiering, not for precise area math) rather than a
geodesic library — consistent with the "no PostGIS needed for simple bbox
math" choice already made for ``DatasetItem.geometry``.
"""
from __future__ import annotations

import math
from typing import Any

from shapely.geometry import shape

# Log-spaced default tiers (meters), covering roughly a car to a large fire
# perimeter. Sanity-check against real annotation bbox percentiles once the
# DB is reachable (see geoops/README.md for the query) and adjust if the
# actual distribution warrants different boundaries.
DEFAULT_TIERS_M: list[float] = [10.0, 40.0, 160.0, 640.0, 2560.0]

# ── Object-normalised scale ladder (Train Head datasets) ─────────────────────
#
# A tile should be several times larger than the object it labels, so that
# the object lands at a similar *fraction* of the model's fixed-size input no
# matter how big it is in the real world. Each object gets one "home" scale —
# the ladder rung whose tile makes the object ~TARGET_OBJECT_FRAC of the tile
# edge — so big and small objects end up roughly the same pixel size.
SCALE_LADDER_M: list[float] = [10.0 * 2**k for k in range(9)]  # 10 ... 2560 m, x2 steps
TARGET_OBJECT_FRAC = 0.3
# Detections whose size is outside this fraction of their scan tile are not
# trusted at that scale (fragments of a bigger object / specks of a smaller one).
SIZE_FRAC_RANGE: tuple[float, float] = (0.1, 0.7)
MULTI_SCALE_SPREAD = 4.0  # p90/p10 object size above which multi-scale is the default
MAX_AUTO_SCALES = 5


def home_scale(span_m: float, scales: list[float] | tuple[float, ...]) -> float:
    """Scale (from ``scales``) at which an object of ``span_m`` metres is
    closest to TARGET_OBJECT_FRAC of the tile edge, in log-space."""
    span = max(span_m, 1e-6)
    return min(scales, key=lambda s: abs(math.log(span / (s * TARGET_OBJECT_FRAC))))


def suggest_scales(spans_m: list[float]) -> dict[str, Any]:
    """Pick the scale list for a class from its annotation sizes.

    Uses the 10th-90th percentile (robust to a stray mislabeled speck or one
    huge outlier). If p90/p10 > MULTI_SCALE_SPREAD the ladder rungs between
    the two ends are all used (multi-scale); otherwise a single scale.
    """
    if not spans_m:
        return {"scales_m": [SCALE_LADDER_M[0]], "p10_m": None, "p90_m": None, "spread": None, "mode": "single"}
    v = sorted(spans_m)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]  # noqa: E731
    p10, p50, p90 = q(0.10), q(0.50), q(0.90)
    spread = p90 / max(p10, 1e-6)
    lo, hi = home_scale(p10, SCALE_LADDER_M), home_scale(p90, SCALE_LADDER_M)
    if spread > MULTI_SCALE_SPREAD and hi > lo:
        scales = [s for s in SCALE_LADDER_M if lo <= s <= hi][:MAX_AUTO_SCALES]
        mode = "multi"
    else:
        scales = [home_scale(p50, SCALE_LADDER_M)]
        mode = "single"
    return {"scales_m": scales, "p10_m": round(p10, 2), "p90_m": round(p90, 2), "spread": round(spread, 2), "mode": mode}


_METERS_PER_DEGREE_LAT = 110_540.0
_METERS_PER_DEGREE_LON_AT_EQUATOR = 111_320.0


def _bbox_of(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = shape(geometry).bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def bbox_coords_span_m(bbox: list[float]) -> float:
    """Longer side (width or height) of an EPSG:4326 ``[minx, miny, maxx, maxy]``
    bbox, in meters. Same math as ``bbox_span_m`` but takes a plain bbox
    instead of a GeoJSON geometry, for callers that already have one (e.g. a
    training crop's padded bbox, which never had a Shapely geometry).
    """
    minx, miny, maxx, maxy = bbox
    mid_lat_rad = math.radians((miny + maxy) / 2.0)
    width_m = (maxx - minx) * _METERS_PER_DEGREE_LON_AT_EQUATOR * math.cos(mid_lat_rad)
    height_m = (maxy - miny) * _METERS_PER_DEGREE_LAT
    return max(abs(width_m), abs(height_m))


def bbox_span_m(geometry: dict[str, Any]) -> float:
    """Longer side (width or height) of ``geometry``'s bbox, in meters."""
    return bbox_coords_span_m(list(_bbox_of(geometry)))


def meters_to_degrees(span_m: float, mid_lat_deg: float) -> tuple[float, float]:
    """(lon_degrees, lat_degrees) spanning ``span_m`` meters at latitude
    ``mid_lat_deg`` — the inverse of ``bbox_coords_span_m``'s per-axis math,
    for callers that need to build a real-world-sized bbox from a center
    point rather than measure an existing one.
    """
    mid_lat_rad = math.radians(mid_lat_deg)
    lon_deg = span_m / (_METERS_PER_DEGREE_LON_AT_EQUATOR * math.cos(mid_lat_rad) or 1e-9)
    lat_deg = span_m / _METERS_PER_DEGREE_LAT
    return lon_deg, lat_deg


def nearest_tier(span_m: float, tiers: list[float] = DEFAULT_TIERS_M) -> int:
    """Index of the tier whose size is closest to ``span_m`` in log-space."""
    if span_m <= 0:
        return 0
    log_span = math.log(span_m)
    diffs = [abs(log_span - math.log(t)) for t in tiers]
    return diffs.index(min(diffs))


def canonical_tiles(
    item_bbox: list[float],
    tier: int,
    clip_bbox: list[float] | None = None,
    tiers: list[float] = DEFAULT_TIERS_M,
) -> list[tuple[int, int, list[float]]]:
    """Deterministic, non-overlapping tile grid for one item at one tier.

    Tiles are anchored to ``item_bbox``'s own origin (not the AOI's), so the
    same ``(item, tier, col, row)`` always maps to the same geographic tile
    across separate scan requests — this is what lets ``embedding_tiles`` act
    as a cache: re-scanning an overlapping AOI hits the same (col, row) keys
    instead of generating new offsets every time.
    """
    minx, miny, maxx, maxy = item_bbox
    mid_lat_rad = math.radians((miny + maxy) / 2.0)
    tier_span_m = tiers[max(0, min(tier, len(tiers) - 1))]
    tile_h_deg = tier_span_m / _METERS_PER_DEGREE_LAT
    lon_scale = math.cos(mid_lat_rad) or 1e-9
    tile_w_deg = tier_span_m / (_METERS_PER_DEGREE_LON_AT_EQUATOR * lon_scale)

    scan_minx, scan_miny, scan_maxx, scan_maxy = minx, miny, maxx, maxy
    if clip_bbox is not None:
        scan_minx = max(scan_minx, clip_bbox[0])
        scan_miny = max(scan_miny, clip_bbox[1])
        scan_maxx = min(scan_maxx, clip_bbox[2])
        scan_maxy = min(scan_maxy, clip_bbox[3])
        if scan_minx >= scan_maxx or scan_miny >= scan_maxy:
            return []

    col_start = math.floor((scan_minx - minx) / tile_w_deg)
    col_end = math.ceil((scan_maxx - minx) / tile_w_deg)
    row_start = math.floor((scan_miny - miny) / tile_h_deg)
    row_end = math.ceil((scan_maxy - miny) / tile_h_deg)

    tiles: list[tuple[int, int, list[float]]] = []
    for row in range(row_start, row_end):
        tile_miny = miny + row * tile_h_deg
        tile_maxy = tile_miny + tile_h_deg
        if tile_maxy <= scan_miny or tile_miny >= scan_maxy:
            continue
        for col in range(col_start, col_end):
            tile_minx = minx + col * tile_w_deg
            tile_maxx = tile_minx + tile_w_deg
            if tile_maxx <= scan_minx or tile_minx >= scan_maxx:
                continue
            tiles.append((col, row, [tile_minx, tile_miny, tile_maxx, tile_maxy]))
    return tiles


def working_resolution_from_refs(
    ref_geometries: list[dict[str, Any]], target_px: tuple[int, int] = (40, 100)
) -> tuple[float, int]:
    """Auto-infer the working resolution for an exemplar scan from the
    reference boxes' real-world size, so objects land roughly ``target_px``
    pixels wide/tall in the working crop — the scale-inference step of the
    exemplar-scan workflow (see geoops/exemplar_tasks.py).

    Returns ``(meters_per_pixel, object_px)`` where ``object_px`` is the
    midpoint of ``target_px`` (the pixel size a reference-sized object should
    occupy at the returned resolution).
    """
    if not ref_geometries:
        raise ValueError("At least one reference geometry is required")
    spans_m = [bbox_span_m(g) for g in ref_geometries]
    mean_span_m = sum(spans_m) / len(spans_m)
    object_px = (target_px[0] + target_px[1]) // 2
    meters_per_pixel = mean_span_m / object_px
    return meters_per_pixel, object_px


def cosine_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimension mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))
