"""Multi-class YOLO detector on the platform's generated splits.

The splits (tiles + ``targets/*.json`` with ``bbox`` and ``class``) are copied to local disk in YOLO layout,
then trained in two stages with Ultralytics:

  1. the backbone frozen, the neck and head adapt to the classes (pretrained features untouched)
  2. everything unfrozen at a lower learning rate, to refine the backbone for the imagery

The head has as many outputs as the dataset has classes (not 80). When training starts from an earlier
model, Ultralytics matches class rows by name, so the classes it already knew keep their weights and a new
class starts fresh. The result is the whole model (one ``.pt``), saved beside the other trained models.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.config import settings
from app.device import DEVICE
from app.storage import get_bytes, get_json, put_bytes

logger = logging.getLogger(__name__)

DEFAULT_BASE = "yolo26l.pt"
_FOLDER = {"train": "train", "eval": "val", "test": "test"}
_models: dict[tuple[str, str], Any] = {}


def _device() -> str | int:
    return 0 if DEVICE.type == "cuda" else "cpu"


def list_bases() -> list[dict]:
    """Weights files available to start from (the service's models/yolo folder)."""
    root = Path(settings.YOLO_MODELS_DIR)
    return [{"id": p.name, "name": p.stem, "size_bytes": p.stat().st_size} for p in sorted(root.glob("*.pt"))] if root.is_dir() else []


def _resolve_base(org_id: str, base: str | None, work: Path) -> str:
    """A weights file path: a name in the models folder, or an earlier model's storage key (to add classes to it)."""
    base = base or DEFAULT_BASE
    local = Path(settings.YOLO_MODELS_DIR) / Path(base).name
    if "/" not in base:
        if not local.is_file():
            raise ValueError(f"Base weights '{base}' not found in {settings.YOLO_MODELS_DIR}.")
        return str(local)
    path = work / "start.pt"
    path.write_bytes(get_bytes(org_id, base))
    return str(path)


def _prepare_dataset(org_id: str, s3_prefix: str, work: Path) -> dict:
    """Copy the splits to local disk in YOLO layout and write data.yaml. Returns the class list and counts."""
    manifest = get_json(org_id, f"{s3_prefix}/manifest.json")
    classes = sorted(manifest.get("classes") or [{"index": 0, "name": manifest.get("class_name") or "object"}], key=lambda c: c["index"])
    names = [c["name"] for c in classes]
    counts = {"train": 0, "val": 0, "test": 0}
    boxes = {"train": 0, "val": 0, "test": 0}
    for sp in _FOLDER.values():
        (work / "images" / sp).mkdir(parents=True, exist_ok=True)
        (work / "labels" / sp).mkdir(parents=True, exist_ok=True)

    def fetch(item: dict) -> tuple[str, int]:
        sp = _FOLDER[item["split"]]
        stem = Path(item["image"]).stem
        (work / "images" / sp / f"{stem}.png").write_bytes(get_bytes(org_id, item["image"]))
        lines = []
        for t in get_json(org_id, item["targets"]).get("targets", []) if item.get("targets") else []:
            cx, cy, w, h = t["bbox"]
            if w > 0 and h > 0:
                lines.append(f"{int(t.get('class', 0))} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (work / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines))
        return sp, len(lines)

    items = [it for it in manifest["items"] if it.get("split") in _FOLDER]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for sp, n in pool.map(fetch, items):
            counts[sp] += 1
            boxes[sp] += n
    if counts["val"] == 0:  # no validation tiles: Ultralytics needs some, so it validates on the training tiles
        logger.warning("yolo: no eval tiles, validating on the training tiles")
        shutil.copytree(work / "images" / "train", work / "images" / "val", dirs_exist_ok=True)
        shutil.copytree(work / "labels" / "train", work / "labels" / "val", dirs_exist_ok=True)
        counts["val"], boxes["val"] = counts["train"], boxes["train"]
    yaml = (
        f"path: {work}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {n!r}\n" for i, n in enumerate(names))
    )
    (work / "data.yaml").write_text(yaml)
    return {"yaml": str(work / "data.yaml"), "names": names, "images": counts, "boxes": boxes, "scale_m": manifest.get("params", {}).get("tile_scale_m")}


def _read_results(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return []
    with csv_path.open() as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    out = []
    for r in rows:
        def num(k: str) -> float:
            try:
                return float(r.get(k) or 0)
            except ValueError:
                return 0.0
        out.append({
            "train": sum(num(k) for k in r if k.startswith("train/") and k.endswith("loss")),
            "val": sum(num(k) for k in r if k.startswith("val/") and k.endswith("loss")),
            "fitness": 0.1 * num("metrics/mAP50(B)") + 0.9 * num("metrics/mAP50-95(B)"),
        })
    return out


def run_yolo_training(
    *, org_id: str, s3_prefix: str, base_weights: str | None, stage1_epochs: int, stage2_epochs: int,
    artifact_prefix: str, imgsz: int = 640, batch: int = 8, lr1: float = 0.002, lr2: float = 0.0001,
    seed: int | None = None, dry_run: bool = False,  # dry_run: tuning trial, no artifact
) -> dict:
    from ultralytics import YOLO  # noqa: PLC0415 — slow import, kept off /health

    work = Path(tempfile.mkdtemp(prefix="yolo-"))
    try:
        data = _prepare_dataset(org_id, s3_prefix, work)
        if data["boxes"]["train"] == 0:
            raise ValueError("No boxes in the training tiles.")
        start = _resolve_base(org_id, base_weights, work)
        common = dict(
            data=data["yaml"], imgsz=imgsz, batch=batch, device=_device(), project=str(work / "runs"), exist_ok=True,
            workers=2, plots=False, verbose=False, flipud=0.5, fliplr=0.5,  # overhead imagery has no "up"
        )
        if seed is not None:
            common.update(seed=seed, deterministic=True)
        curves: list[dict] = []
        best = start
        freeze = 0
        if stage1_epochs > 0:
            model = YOLO(start)
            freeze = len(model.model.yaml.get("backbone", [])) or 10      # the whole backbone
            model.train(epochs=stage1_epochs, freeze=freeze, optimizer="AdamW", lr0=lr1, name="stage1", **common)
            curves += _read_results(work / "runs" / "stage1" / "results.csv")
            best = str(work / "runs" / "stage1" / "weights" / "best.pt")
        if stage2_epochs > 0:
            model = YOLO(best)
            model.train(epochs=stage2_epochs, optimizer="AdamW", lr0=lr2, name="stage2", **common)
            offset = len(curves)
            last = _read_results(work / "runs" / "stage2" / "results.csv")
            curves += last
            best = str(work / "runs" / "stage2" / "weights" / "best.pt")
        else:
            offset = 0
            last = curves
        if not curves:
            raise ValueError("Both stages have zero epochs.")

        best_i = max(range(len(last)), key=lambda i: last[i]["fitness"]) if last else 0
        artifact_key = None
        if not dry_run:
            artifact_key = f"{artifact_prefix}/head_artifact.pt"
            put_bytes(org_id, artifact_key, Path(best).read_bytes(), "application/octet-stream")
        return {
            "artifact_key": artifact_key,
            "train_examples": data["images"]["train"], "eval_examples": data["images"]["val"],
            "epoch_losses": [c["train"] for c in curves], "val_losses": [c["val"] for c in curves],
            "best_epoch": offset + best_i + 1, "best_val_loss": last[best_i]["val"] if last else None,
            "final_loss": curves[-1]["train"],
            "classes": data["names"], "imgsz": imgsz, "boxes": data["boxes"],
            "stages": {"stage1_epochs": stage1_epochs, "stage2_epochs": stage2_epochs, "frozen_layers": freeze, "stage1_lr": lr1, "stage2_lr": lr2},
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def load_trained_yolo_model(org_id: str, artifact_key: str):
    from ultralytics import YOLO  # noqa: PLC0415

    key = (str(org_id), artifact_key)
    if key not in _models:
        cache = Path(tempfile.gettempdir()) / "yolo-cache"
        cache.mkdir(exist_ok=True)
        path = cache / (hashlib.sha1(f"{org_id}/{artifact_key}".encode()).hexdigest() + ".pt")  # noqa: S324 — cache name
        if not path.is_file():
            path.write_bytes(get_bytes(org_id, artifact_key))
        _models[key] = YOLO(str(path))
    return _models[key]


def evaluate_yolo(org_id: str, artifact_key: str, s3_prefix: str, split: str = "test", iou_thresh: float = 0.5) -> dict:
    work = Path(tempfile.mkdtemp(prefix="yolo-eval-"))
    try:
        data = _prepare_dataset(org_id, s3_prefix, work)
        sp = _FOLDER[split]
        n_images = data["images"][sp]
        base = {"num_images": n_images, "iou_thresh": iou_thresh}
        if n_images == 0 or data["boxes"][sp] == 0:
            return {**base, "num_ground_truth": 0, "ap50": None, "note": "No ground-truth boxes in this split."}
        model = load_trained_yolo_model(org_id, artifact_key)
        m = model.val(data=data["yaml"], split=sp, imgsz=int(model.overrides.get("imgsz") or 640), batch=8, iou=0.6, device=_device(), plots=False, verbose=False, project=str(work / "runs"), exist_ok=True)
        b = m.box
        per_class = []
        for i, c in enumerate(b.ap_class_index):
            per_class.append({
                "name": data["names"][int(c)], "ap50": round(float(b.ap50[i]), 4), "precision": round(float(b.p[i]), 4),
                "recall": round(float(b.r[i]), 4), "instances": int(m.nt_per_class[int(c)]),
            })
        return {
            **base, "num_ground_truth": int(sum(m.nt_per_class)), "ap50": round(float(b.map50), 4), "map": round(float(b.map), 4),
            "precision": round(float(b.mp), 4), "recall": round(float(b.mr), 4), "mean_iou_tp": None, "per_class": per_class,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_yolo_inference(model, image_base64: str, georef_metadata: dict[str, Any], *, score_thresh: float = 0.25) -> list[dict[str, Any]]:
    """One box per detection, labelled with the class name, in the platform's prediction shape."""
    from PIL import Image  # noqa: PLC0415

    bbox = georef_metadata.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("georef_metadata.bbox [minx, miny, maxx, maxy] is required")
    img = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
    w, h = img.size
    res = model.predict(img, conf=score_thresh, verbose=False, device=_device())[0]
    ix0, iy0, ix1, iy1 = bbox
    out = []
    for (x1, y1, x2, y2), conf, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist(), strict=True):
        lon1, lon2 = ix0 + x1 / w * (ix1 - ix0), ix0 + x2 / w * (ix1 - ix0)
        lat2, lat1 = iy1 - y1 / h * (iy1 - iy0), iy1 - y2 / h * (iy1 - iy0)
        out.append({
            "label": str(model.names[int(cls)]), "confidence": float(conf),
            "geometry": {"type": "Polygon", "coordinates": [[[lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2], [lon1, lat1]]]},
        })
    return out


def export_yolo_onnx(org_id: str, artifact_key: str, imgsz: int | None = None) -> bytes:
    model = load_trained_yolo_model(org_id, artifact_key)
    path = model.export(format="onnx", imgsz=imgsz or int(model.overrides.get("imgsz") or 640), device="cpu", verbose=False)
    return Path(path).read_bytes()
