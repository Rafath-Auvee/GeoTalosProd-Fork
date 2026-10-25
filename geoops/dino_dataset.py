"""Pure (no DB / S3 / torch) logic for building a reusable DINOv2 training
dataset from one exhaustively-labeled AOI:

- tiling the AOI into a deterministic grid and assigning annotations to tiles
- spatially-grouped ("region") train/eval/test assignment, stratified so
  every split gets a fair share of positive tiles when there are enough
- negative-tile subsampling per split
- the data fingerprint used to tell whether a built dataset is stale

Kept free of side effects so it can be unit-tested directly — see
tests/test_dino_dataset.py. The Celery task that fetches imagery and writes
to storage lives in geoops.dino_dataset_tasks.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Literal

from shapely.geometry import box as shapely_box, shape

from geoops.scale import canonical_tiles, meters_to_degrees

Split = Literal["train", "eval", "test"]

# Safety bound on the AOI grid at the chosen scale (a km-wide AOI at 5 m is
# ~40k tiles; beyond this the scale is almost certainly a mistake).
MAX_GRID_TILES = 200_000
MIN_NEGATIVE_TILES = 5


@dataclass(frozen=True)
class BuildParams:
    # Tile edge lengths (metres). One entry = single-scale; several = each
    # object is tiled at the scale that makes it a similar fraction of the
    # tile (see geoops.scale.home_scale).
    scales_m: tuple[float, ...]
    crop_px: int = 1280
    eval_split: float = 0.15
    test_split: float = 0.15
    # Max background tiles kept per split, as a multiple of that split's positives.
    neg_ratio: float = 2.0
    # Split regions are square blocks of this many (coarsest-scale) tiles per side.
    region_tiles: int = 3


def seed_from(fingerprint: str) -> int:
    return int.from_bytes(hashlib.sha256(fingerprint.encode()).digest()[:8], "big")


def compute_data_fingerprint(
    *,
    annotation_versions: list[tuple[str, str]],
    set_ids: list[str],
    aoi_bbox: list | None,  # one bbox, or (several AOIs) a list of bboxes
    aoi_geometry: dict | list | None,
    task: str,
) -> str:
    """Hash of everything about the *data* a build depends on. Build
    parameters (scale, crop size, splits) are deliberately excluded: changing
    them is always an explicit rebuild, not staleness."""
    payload = {
        "annotations": sorted(f"{aid}:{ts}" for aid, ts in annotation_versions),
        "sets": sorted(set_ids),
        "aoi_bbox": (
            [[round(float(v), 9) for v in bb] for bb in aoi_bbox] if aoi_bbox and isinstance(aoi_bbox[0], list)
            else [round(float(v), 9) for v in aoi_bbox] if aoi_bbox else None
        ),
        "aoi_geometry": aoi_geometry,
        "task": task,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# ── Tiling ───────────────────────────────────────────────────────────────────


def build_aoi_tiles(
    *,
    item_id: str,
    item_bbox: list[float],
    aoi_bbox: list[float],
    aoi_geometry: dict | None,
    rows: list[dict],
    tile_scale_m: float,
    all_rows: list[dict] | None = None,
) -> tuple[list[dict], int]:
    """Tile one (dataset item, AOI) pair at ``tile_scale_m`` metres.

    Each of ``rows`` (``{"geometry": geojson, ...}`` — the objects whose home
    scale this is) is assigned to the one tile containing its centroid, so an
    object is never duplicated across tiles. A tile with no assigned row is a
    *negative* only if it also does not touch any annotation in ``all_rows``
    (every object of the class, including those homed at other scales) — an
    object straddling a tile edge, or too big for this scale, would otherwise
    look like unlabeled foreground, i.e. a false negative. Those ambiguous
    tiles are dropped and counted.

    Returns ``(tiles, ambiguous_dropped)`` with
    ``tiles = [{"crop_bbox", "rows", "col", "row", "item_id", "scale_m", "tile_key"}]``.
    """
    clip = [
        max(item_bbox[0], aoi_bbox[0]),
        max(item_bbox[1], aoi_bbox[1]),
        min(item_bbox[2], aoi_bbox[2]),
        min(item_bbox[3], aoi_bbox[3]),
    ]
    if clip[0] >= clip[2] or clip[1] >= clip[3]:
        return [], 0

    grid = canonical_tiles(item_bbox, 0, clip_bbox=clip, tiers=[tile_scale_m])
    if len(grid) > MAX_GRID_TILES:
        raise ValueError(
            f"Tile scale {tile_scale_m:g} m yields {len(grid)} tiles for this AOI "
            f"(limit {MAX_GRID_TILES}) — pick a larger scale."
        )

    aoi_poly = shape(aoi_geometry) if aoi_geometry else shapely_box(*aoi_bbox)
    shapes = [shape(r["geometry"]) for r in rows]
    guard_shapes = shapes if all_rows is None else [shape(r["geometry"]) for r in all_rows]

    tiles: list[dict] = []
    ambiguous = 0
    for col, row_idx, tile_bbox in grid:
        tile_poly = shapely_box(*tile_bbox)
        if not tile_poly.intersects(aoi_poly):
            continue
        own = [rows[i] for i, s in enumerate(shapes) if tile_poly.contains(s.centroid)]
        if not own and any(tile_poly.intersects(g) for g in guard_shapes):
            ambiguous += 1
            continue
        tiles.append(
            {
                "crop_bbox": tile_bbox,
                "rows": own,
                "col": col,
                "row": row_idx,
                "item_id": item_id,
                "scale_m": float(tile_scale_m),
                "tile_key": f"{item_id}:{tile_scale_m:g}:{col}:{row_idx}",
            }
        )
    return tiles, ambiguous


# ── Splitting ────────────────────────────────────────────────────────────────


def allocate_regions(
    weights: dict[tuple, float], eval_split: float, test_split: float, rng: random.Random
) -> dict[tuple, str]:
    """Assign whole regions to splits so each split's share of the total
    *weight* (annotations, or tiles for background regions) lands near its
    target fraction. Assigning by region *count* instead lets a few dense
    regions fall into one small split and starve training of data.

    Largest-first: each region goes to the split currently furthest below its
    target. Then, when there are enough regions, test (>=2 regions) and eval
    (>=3) are guaranteed at least one, taken from the biggest split — train
    always keeps one.
    """
    if not weights:
        return {}
    keys = sorted(weights, key=str)
    rng.shuffle(keys)  # seeded tie-break between equal weights
    keys.sort(key=lambda k: -weights[k])
    total = float(sum(weights.values())) or 1.0
    targets = {"train": 1.0 - eval_split - test_split, "eval": eval_split, "test": test_split}
    got = {s: 0.0 for s in targets}
    out: dict[tuple, str] = {}
    for i, k in enumerate(keys):
        # The largest region always trains, so train can never end up empty.
        s = "train" if i == 0 else max(targets, key=lambda name: targets[name] * total - got[name])
        out[k] = s
        got[s] += weights[k]

    n = len(keys)
    for split, needs in (("test", 2), ("eval", 3)):
        if n >= needs and split not in out.values():
            counts = {name: sum(1 for v in out.values() if v == name) for name in targets}
            # A donor may be emptied unless it is train; train must keep one.
            donors = [name for name in targets if counts[name] >= 2 or (name != "train" and counts[name] >= 1)]
            donor = max(donors, key=lambda name: counts[name])
            smallest = min((k for k, v in out.items() if v == donor), key=lambda k: weights[k])
            out[smallest] = split
    return out


def assign_splits(tiles: list[dict], params: BuildParams, seed: int) -> list[dict]:
    """Annotate every tile with ``split`` and drop surplus negatives.

    Tiles are grouped into square regions of ``params.region_tiles`` tiles
    (of the coarsest scale, shared by all scales) and each region goes wholly
    to one split, so neighbouring tiles never straddle train/test. Regions containing a
    positive and regions with only negatives are allocated separately (see
    ``allocate_regions``), which keeps positives represented in every split
    whenever there are enough positive regions. Returns the kept tiles in a
    stable order.
    """
    # Regions are blocks of a fixed size in *metres*, shared by every scale, so
    # a place is in the same split whichever scale's tile covers it.
    block_m = max(1, params.region_tiles) * max(params.scales_m)
    mid_lat = sum((t["crop_bbox"][1] + t["crop_bbox"][3]) / 2 for t in tiles) / len(tiles) if tiles else 0.0
    block_lon, block_lat = meters_to_degrees(block_m, mid_lat)
    regions: dict[tuple, list[dict]] = {}
    for t in tiles:
        cx = (t["crop_bbox"][0] + t["crop_bbox"][2]) / 2
        cy = (t["crop_bbox"][1] + t["crop_bbox"][3]) / 2
        regions.setdefault((t["item_id"], math.floor(cx / block_lon), math.floor(cy / block_lat)), []).append(t)

    # A tile's ``role`` says where its annotations may be used: "train", "test", or "both" (the default).
    # A region holding train-only objects goes wholly to train, else one holding test-only objects goes to
    # test; those are decided by the user, and only the remaining regions are shared out by the split sizes.
    forced: dict[tuple, str] = {}
    for k, v in regions.items():
        roles = {t.get("role", "both") for t in v}
        if "train" in roles:
            forced[k] = "train"
        elif "test" in roles:
            forced[k] = "test"
    free = {k: v for k, v in regions.items() if k not in forced}
    pos_w = {k: float(sum(len(t["rows"]) for t in v)) for k, v in free.items() if any(t["rows"] for t in v)}
    neg_w = {k: float(len(v)) for k, v in free.items() if not any(t["rows"] for t in v)}
    rng = random.Random(seed)
    region_split = {
        **forced,
        **allocate_regions(pos_w, params.eval_split, params.test_split, rng),
        **allocate_regions(neg_w, params.eval_split, params.test_split, rng),
    }

    by_split: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}
    for key in sorted(regions, key=str):
        for t in sorted(regions[key], key=lambda x: (x["col"], x["row"])):
            by_split[region_split[key]].append({**t, "split": region_split[key]})

    kept: list[dict] = []
    for split, group in by_split.items():
        pos = [t for t in group if t["rows"]]
        neg = [t for t in group if not t["rows"]]
        cap = int(max(len(pos) * params.neg_ratio, MIN_NEGATIVE_TILES))
        if len(neg) > cap:
            neg = random.Random(seed ^ hash_split(split)).sample(neg, cap)
        kept += pos + neg
    return sorted(kept, key=lambda t: (t["split"], t["tile_key"]))


def hash_split(split: str) -> int:
    return int.from_bytes(hashlib.sha256(split.encode()).digest()[:4], "big")


def split_stats(tiles: list[dict], params: BuildParams) -> dict[str, Any]:
    """Counts (per split and per scale) + human-readable warnings."""
    counts = {s: {"tiles": 0, "positives": 0, "negatives": 0, "annotations": 0} for s in ("train", "eval", "test")}
    by_scale: dict[str, dict[str, int]] = {}
    for t in tiles:
        c = counts[t["split"]]
        sc = by_scale.setdefault(f"{t['scale_m']:g}", {"tiles": 0, "positives": 0, "annotations": 0})
        c["tiles"] += 1
        sc["tiles"] += 1
        if t["rows"]:
            c["positives"] += 1
            c["annotations"] += len(t["rows"])
            sc["positives"] += 1
            sc["annotations"] += len(t["rows"])
        else:
            c["negatives"] += 1
    warnings: list[str] = []
    if counts["train"]["positives"] == 0:
        warnings.append("Train split has no positive tiles.")
    if counts["test"]["positives"] == 0:
        warnings.append("Test split has no positive tiles — test metrics will be empty.")
    if params.eval_split > 0 and counts["eval"]["positives"] == 0:
        warnings.append("Eval split has no positive tiles — validation loss will only see background.")
    return {"splits": counts, "by_scale": by_scale, "warnings": warnings}
