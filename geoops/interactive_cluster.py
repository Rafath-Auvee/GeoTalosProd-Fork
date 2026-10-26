"""Grouping of proposed masks by visual similarity (Interactive annotation mode).

Pure numpy/scipy — no DB, S3 or torch — so it is unit-testable and cheap to
re-run. Embeddings are L2-normalised, so Euclidean distance on them is a
monotonic function of cosine distance. Clustering is hierarchical (Ward): the
tree is built once per embedding set and cached, so moving the UI's
"fewer groups ↔ finer groups" slider only re-cuts the tree.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

_LINKAGE_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
_CACHE_MAX = 4


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def _tree(xn: np.ndarray, cache_key: str | None) -> np.ndarray:
    if cache_key and cache_key in _LINKAGE_CACHE:
        _LINKAGE_CACHE.move_to_end(cache_key)
        return _LINKAGE_CACHE[cache_key]
    z = linkage(xn, method="ward")
    if cache_key:
        _LINKAGE_CACHE[cache_key] = z
        while len(_LINKAGE_CACHE) > _CACHE_MAX:
            _LINKAGE_CACHE.popitem(last=False)
    return z


def cluster(xn: np.ndarray, n_clusters: int, cache_key: str | None = None) -> np.ndarray:
    """Labels 0..k-1 with 0 the largest group. ``k`` is clamped to [1, n]."""
    n = len(xn)
    if n <= 1:
        return np.zeros(n, dtype=int)
    k = int(min(max(1, n_clusters), n))
    labels = fcluster(_tree(xn, cache_key), t=k, criterion="maxclust") - 1
    # Re-number by size, biggest first, so "group A" is the dominant look.
    order = np.argsort(-np.bincount(labels))
    remap = {int(old): new for new, old in enumerate(order) if np.any(labels == old)}
    return np.array([remap[int(label)] for label in labels], dtype=int)


def outlier_z(xn: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """How unlike its own group each mask is: z-score of its cosine distance to
    the group centroid, within the group (0 = typical, >2 = suspicious)."""
    z = np.zeros(len(xn), dtype=np.float32)
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        if len(idx) < 3:
            continue
        centroid = normalize(xn[idx].mean(axis=0, keepdims=True))[0]
        d = 1.0 - xn[idx] @ centroid
        z[idx] = (d - d.mean()) / (d.std() + 1e-6)
    return z


def similarity(xn: np.ndarray, query_idx: list[int]) -> np.ndarray:
    """Cosine similarity of every mask to the mean of the query masks."""
    q = normalize(xn[query_idx].mean(axis=0, keepdims=True))[0]
    return xn @ q


def group(xn: np.ndarray, ids: list[str], n_clusters: int, cache_key: str | None = None) -> dict[str, Any]:
    labels = cluster(xn, n_clusters, cache_key)
    z = outlier_z(xn, labels)
    sizes = np.bincount(labels).tolist()
    return {
        "n_clusters": len(sizes),
        "sizes": sizes,
        "labels": {mid: int(lab) for mid, lab in zip(ids, labels, strict=True)},
        "outlier_z": {mid: round(float(v), 3) for mid, v in zip(ids, z, strict=True)},
    }
