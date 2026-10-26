"""Pure-logic tests for geoops.interactive_cluster."""
import numpy as np

from geoops.interactive_cluster import cluster, group, normalize, outlier_z, similarity


def _blobs(per: int = 40, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = np.eye(dim)[:3] * 5
    x = np.vstack([c + rng.normal(0, 0.3, (per, dim)) for c in centres])
    truth = np.repeat([0, 1, 2], per)
    return normalize(x), truth


def test_cluster_recovers_obvious_groups_and_orders_by_size():
    x, truth = _blobs()
    labels = cluster(x, 3)
    # each true blob lands wholly in one label
    for t in (0, 1, 2):
        assert len(set(labels[truth == t])) == 1
    assert len(set(labels)) == 3
    assert list(np.bincount(labels)) == sorted(np.bincount(labels), reverse=True)


def test_cluster_clamps_k_and_handles_tiny_inputs():
    x, _ = _blobs(per=2)
    assert len(set(cluster(x, 999))) == len(x)
    assert len(set(cluster(x, 0))) == 1
    assert list(cluster(x[:1], 5)) == [0]


def test_finer_granularity_only_splits_never_regroups():
    x, _ = _blobs()
    coarse, fine = cluster(x, 3), cluster(x, 6)
    for lab in set(fine):
        assert len(set(coarse[fine == lab])) == 1  # every fine group sits inside one coarse group


def test_outlier_is_flagged_inside_its_group():
    x, truth = _blobs()
    odd = normalize(np.eye(16)[10:11] * 5 + 1.5)  # a point far from every blob
    x = np.vstack([x, odd])
    labels = np.concatenate([cluster(x[:-1], 3), [0]])
    z = outlier_z(x, labels)
    assert z[-1] > 2 and np.median(z[:-1]) < 1


def test_similarity_ranks_same_blob_first():
    x, truth = _blobs()
    s = similarity(x, [0, 1, 2])
    assert s[truth == 0].min() > s[truth != 0].max()


def test_group_payload_shape():
    x, _ = _blobs(per=5)
    ids = [f"m{i}" for i in range(len(x))]
    out = group(x, ids, 3)
    assert out["n_clusters"] == 3 and sum(out["sizes"]) == len(ids) and set(out["labels"]) == set(ids)
