"""Pure-logic tests for geoops.dino_dataset (tiling, region split, fingerprint)."""
from geoops.dino_dataset import (
    BuildParams,
    allocate_regions,
    assign_splits,
    build_aoi_tiles,
    compute_data_fingerprint,
    seed_from,
    split_stats,
)

from geoops.scale import home_scale, meters_to_degrees, suggest_scales, SCALE_LADDER_M

PARAMS = BuildParams(scales_m=(10.0,), region_tiles=2)
_LON, _LAT = meters_to_degrees(10.0, 0.0)  # one 10 m tile at the equator, in degrees


def _tiles(n_cols: int, n_rows: int, positive_cells: set[tuple[int, int]], scale: float = 10.0) -> list[dict]:
    return [
        {
            "crop_bbox": [c * _LON, r * _LAT, (c + 1) * _LON, (r + 1) * _LAT],
            "rows": [{"geometry": {}}] if (c, r) in positive_cells else [],
            "col": c,
            "row": r,
            "item_id": "item",
            "scale_m": scale,
            "tile_key": f"item:{scale:g}:{c}:{r}",
        }
        for c in range(n_cols)
        for r in range(n_rows)
    ]


def test_allocate_regions_balances_weight_not_region_count():
    import random

    # Two dense regions and many light ones: by count the dense ones could
    # both land in a small split; by weight train must keep most of the mass.
    weights = {("i", 0, 0): 40.0, ("i", 1, 0): 35.0, **{("i", c, 1): 2.0 for c in range(10)}}
    out = allocate_regions(weights, 0.15, 0.15, random.Random(1))
    totals = {s: sum(w for k, w in weights.items() if out[k] == s) for s in ("train", "eval", "test")}
    assert totals["train"] >= 0.6 * sum(weights.values())


def test_allocate_regions_guarantees_representation_and_train():
    import random

    for n in range(1, 40):
        weights = {("i", c, 0): float(1 + (c * 7) % 5) for c in range(n)}
        out = allocate_regions(weights, 0.4, 0.4, random.Random(n))
        splits = list(out.values())
        assert "train" in splits
        if n >= 2:
            assert "test" in splits
        if n >= 3:
            assert "eval" in splits


def test_split_is_deterministic_for_same_seed():
    tiles = _tiles(12, 12, {(1, 1), (5, 5), (9, 2), (3, 10), (10, 10), (7, 7)})
    a = assign_splits(tiles, PARAMS, seed=7)
    b = assign_splits(tiles, PARAMS, seed=7)
    assert [(t["tile_key"], t["split"]) for t in a] == [(t["tile_key"], t["split"]) for t in b]


def test_regions_never_straddle_splits():
    tiles = _tiles(12, 12, {(1, 1), (5, 5), (9, 2), (3, 10), (10, 10), (7, 7)})
    out = assign_splits(tiles, PARAMS, seed=1)
    lon, lat = meters_to_degrees(2 * 10.0, 0.0)  # region = region_tiles x coarsest scale
    by_region: dict[tuple, set[str]] = {}
    for t in out:
        cx = (t["crop_bbox"][0] + t["crop_bbox"][2]) / 2
        cy = (t["crop_bbox"][1] + t["crop_bbox"][3]) / 2
        by_region.setdefault((int(cx // lon), int(cy // lat)), set()).add(t["split"])
    assert all(len(s) == 1 for s in by_region.values())


def test_positives_land_in_every_split_when_enough_regions():
    cells = {(c * 2, r * 2) for c in range(6) for r in range(6)}  # 36 positive regions
    out = assign_splits(_tiles(12, 12, cells), PARAMS, seed=3)
    stats = split_stats(out, PARAMS)
    assert all(stats["splits"][s]["positives"] > 0 for s in ("train", "eval", "test"))
    assert stats["warnings"] == []


def test_few_positives_warns_instead_of_silently_empty_test():
    out = assign_splits(_tiles(8, 8, {(0, 0)}), PARAMS, seed=1)
    stats = split_stats(out, PARAMS)
    assert stats["splits"]["train"]["positives"] == 1
    assert any("Test split has no positive" in w for w in stats["warnings"])


def test_negatives_are_capped_per_split():
    params = BuildParams(scales_m=(10.0,), region_tiles=1, neg_ratio=1.0)
    out = assign_splits(_tiles(20, 20, {(0, 0), (19, 19)}), params, seed=5)
    for split in ("train", "eval", "test"):
        group = [t for t in out if t["split"] == split]
        pos = sum(1 for t in group if t["rows"])
        neg = len(group) - pos
        assert neg <= max(pos * 1.0, 5)


def test_fingerprint_changes_with_data_not_order():
    base = dict(set_ids=["s1"], aoi_bbox=[0, 0, 1, 1], aoi_geometry=None, task="detect")
    a = compute_data_fingerprint(annotation_versions=[("a", "t1"), ("b", "t1")], **base)
    b = compute_data_fingerprint(annotation_versions=[("b", "t1"), ("a", "t1")], **base)
    c = compute_data_fingerprint(annotation_versions=[("a", "t2"), ("b", "t1")], **base)
    assert a == b and a != c
    assert seed_from(a) == seed_from(b)


def test_straddling_object_makes_neighbour_ambiguous_not_negative():
    # ~10 m tiles near the equator: 1e-4 deg ≈ 11 m. Object centred on a tile
    # edge belongs to one tile; its neighbour must not become a "negative".
    item_bbox = [0.0, 0.0, 0.001, 0.001]
    geom = {
        "type": "Polygon",
        "coordinates": [[[0.00008, 0.00001], [0.00012, 0.00001], [0.00012, 0.00003], [0.00008, 0.00003], [0.00008, 0.00001]]],
    }
    tiles, ambiguous = build_aoi_tiles(
        item_id="i", item_bbox=item_bbox, aoi_bbox=item_bbox, aoi_geometry=None,
        rows=[{"geometry": geom}], tile_scale_m=10.0,
    )
    assert sum(1 for t in tiles if t["rows"]) == 1
    assert ambiguous >= 1


def test_home_scale_makes_objects_a_similar_fraction_of_the_tile():
    fracs = [span / home_scale(span, SCALE_LADDER_M) for span in (2, 3, 6, 12, 33, 90, 400)]
    assert all(0.15 <= f <= 0.45 for f in fracs), fracs


def test_suggest_scales_multi_only_when_sizes_spread():
    narrow = suggest_scales([15, 16, 18, 19, 20, 21])
    assert narrow["mode"] == "single" and len(narrow["scales_m"]) == 1
    wide = suggest_scales([3] * 40 + [12] * 30 + [40] * 30)
    assert wide["mode"] == "multi" and len(wide["scales_m"]) >= 2
    assert wide["scales_m"] == sorted(wide["scales_m"])
    assert suggest_scales([])["mode"] == "single"


def test_other_scale_object_makes_background_tile_ambiguous_not_negative():
    item_bbox = [0.0, 0.0, 0.002, 0.002]
    big = {"type": "Polygon", "coordinates": [[[0.0004, 0.0004], [0.0010, 0.0004], [0.0010, 0.0010], [0.0004, 0.0010], [0.0004, 0.0004]]]}
    # No own-scale rows at this (fine) scale, but the big object is known:
    tiles, ambiguous = build_aoi_tiles(
        item_id="i", item_bbox=item_bbox, aoi_bbox=item_bbox, aoi_geometry=None,
        rows=[], all_rows=[{"geometry": big}], tile_scale_m=20.0,
    )
    assert ambiguous > 0
    assert all(not _touches(t, big) for t in tiles)


def _touches(tile, geom) -> bool:
    from shapely.geometry import box, shape

    return box(*tile["crop_bbox"]).intersects(shape(geom))


def test_fingerprint_handles_several_aois_and_differs_from_one():
    base = dict(annotation_versions=[("a", "t1")], set_ids=["s1", "s2"], aoi_geometry=None, task="detect")
    one = compute_data_fingerprint(aoi_bbox=[0, 0, 1, 1], **base)
    two = compute_data_fingerprint(aoi_bbox=[[0, 0, 1, 1], [2, 2, 3, 3]], **base)
    assert one != two
    assert two == compute_data_fingerprint(aoi_bbox=[[0, 0, 1, 1], [2, 2, 3, 3]], **base)


def _with_role(tiles, role_by_cell):
    return [{**t, "role": role_by_cell.get((t["col"], t["row"]), "both")} for t in tiles]


def test_train_only_and_test_only_regions_are_forced_and_the_rest_still_split():
    cells = {(c * 2, r * 2) for c in range(6) for r in range(6)}
    tiles = _tiles(12, 12, cells)
    roles = {(0, 0): "train", (10, 10): "test"}
    out = assign_splits(_with_role(tiles, roles), PARAMS, seed=2)
    by = {(t["col"], t["row"]): t["split"] for t in out}
    assert by[(0, 0)] == "train"
    assert by[(10, 10)] == "test"
    # Every other region is still shared out by the split sizes, and each split still gets positives.
    free = [t for t in out if (t["col"], t["row"]) not in roles]
    assert {t["split"] for t in free if t["rows"]} == {"train", "eval", "test"}


def test_a_whole_region_follows_the_strictest_role():
    tiles = _tiles(4, 4, {(0, 0), (1, 1)})  # both positives sit in one 2x2 region
    out = assign_splits(_with_role(tiles, {(1, 1): "test", (0, 0): "train"}), PARAMS, seed=1)
    assert {t["split"] for t in out if (t["col"], t["row"]) in {(0, 0), (1, 0), (0, 1), (1, 1)}} == {"train"}


def test_no_roles_behaves_as_before():
    tiles = _tiles(12, 12, {(1, 1), (5, 5), (9, 2)})
    assert [(t["tile_key"], t["split"]) for t in assign_splits(tiles, PARAMS, seed=7)] == [
        (t["tile_key"], t["split"]) for t in assign_splits(_with_role(tiles, {}), PARAMS, seed=7)
    ]
