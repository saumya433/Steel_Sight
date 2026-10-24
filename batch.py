"""Batch inspection: many images (or a zip of images) -> one summary table + CSV.

Kept free of Streamlit so it can be tested and reused from scripts.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import PurePosixPath
from typing import Callable, Iterable, Iterator, Optional

import cv2
import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
MAX_IMAGES = 300

CSV_FIELDS = ["file", "decision", "n_defects", "max_conf", "classes", "anomaly_score", "ms", "note"]


def decode_image(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None


def iter_images(files: Iterable[tuple[str, bytes]], limit: int = MAX_IMAGES) -> Iterator[tuple[str, Optional[np.ndarray]]]:
    """Yields (name, image or None). Zips are opened and their images yielded; at most `limit` images."""
    n = 0
    for name, data in files:
        if n >= limit:
            return
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in sorted(z.infolist(), key=lambda i: i.filename):
                    p = PurePosixPath(info.filename)
                    if info.is_dir() or p.name.startswith(".") or "__MACOSX" in p.parts or p.suffix.lower() not in IMAGE_EXT:
                        continue
                    if n >= limit:
                        return
                    yield info.filename, decode_image(z.read(info))
                    n += 1
        elif PurePosixPath(name).suffix.lower() in IMAGE_EXT:
            yield name, decode_image(data)
            n += 1


def run_batch(images: Iterable[tuple[str, Optional[np.ndarray]]], inspect: Callable, check: Optional[Callable] = None,
              prepare: Optional[Callable] = None, on_progress: Optional[Callable[[int, str], None]] = None) -> list[dict]:
    """inspect(frame) -> InspectionResult; check(frame) -> SceneVerdict (optional); prepare(frame) -> frame (optional)."""
    rows: list[dict] = []
    for i, (name, im) in enumerate(images):
        if on_progress:
            on_progress(i, name)
        if im is None:
            rows.append({"file": name, "decision": "UNREADABLE", "n_defects": 0, "max_conf": None, "classes": "",
                         "anomaly_score": None, "ms": None, "note": "could not decode image", "_record": None})
            continue
        if check is not None:
            v = check(im)
            if not v.ok:
                rows.append({"file": name, "decision": "REJECTED", "n_defects": 0, "max_conf": None, "classes": "",
                             "anomaly_score": None, "ms": None, "note": f"{v.label}: {v.reason}", "_record": None})
                continue
        res = inspect(prepare(im) if prepare else im)
        g = res.gate
        rows.append({
            "file": name, "decision": res.decision.value, "n_defects": len(g.kept),
            "max_conf": round(float(g.max_conf), 3) if g.kept else None,
            "classes": ", ".join(sorted({d.name for d in g.kept})),
            "anomaly_score": None if res.anomaly_score is None else round(float(res.anomaly_score), 3),
            "ms": round(float(res.timings_ms.get("total_ms", 0.0)), 1), "note": g.reason,
            "_record": res.to_record(), "_per_class": _count(d.name for d in g.kept),
        })
    return rows


def _count(names: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in names:
        out[n] = out.get(n, 0) + 1
    return out


def summarize(rows: list[dict]) -> dict:
    inspected = [r for r in rows if r["decision"] not in ("REJECTED", "UNREADABLE")]
    decisions = _count(r["decision"] for r in rows)
    per_class: dict[str, int] = {}
    for r in inspected:
        for k, v in (r.get("_per_class") or {}).items():
            per_class[k] = per_class.get(k, 0) + v
    flagged = sum(1 for r in inspected if r["decision"] in ("AUTO_FLAG", "REVIEW", "UNKNOWN"))
    ms = [r["ms"] for r in inspected if r["ms"] is not None]
    return {
        "images": len(rows), "inspected": len(inspected), "rejected": decisions.get("REJECTED", 0),
        "unreadable": decisions.get("UNREADABLE", 0), "decisions": decisions, "per_class": per_class,
        "flagged_fraction": round(flagged / len(inspected), 4) if inspected else 0.0,
        "avg_ms": round(sum(ms) / len(ms), 1) if ms else None,
    }


def to_csv(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")
