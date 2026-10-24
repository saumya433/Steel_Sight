"""Batch inspection, coil PDF report and KPI computation (no model, no network)."""

import io
import json
import time
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np

from steelsight.batch import iter_images, run_batch, summarize, to_csv
from steelsight.gate import Decision
from steelsight.kpi import compute_kpis
from steelsight.report import coil_report_pdf
from steelsight.store import ResultStore

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def png_bytes(value=120, size=64):
    rng = np.random.default_rng(value)
    im = rng.normal(value, 12, (size, size, 3)).clip(0, 255).astype(np.uint8)
    return cv2.imencode(".png", im)[1].tobytes()


def fake_result(decision, names):
    kept = [SimpleNamespace(name=n, conf=0.8) for n in names]
    gate = SimpleNamespace(kept=kept, max_conf=0.8 if kept else 0.0, reason=f"{decision} test")
    rec = {"decision": decision, "n_detections": len(kept), "detections": [{"name": n, "conf": 0.8, "xyxy": [0, 0, 10, 10]} for n in names]}
    return SimpleNamespace(decision=Decision(decision), gate=gate, anomaly_score=None,
                           timings_ms={"total_ms": 12.0}, to_record=lambda: rec)


def test_iter_images_handles_files_zips_junk_and_limit():
    z = io.BytesIO()
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("set/a.png", png_bytes(100))
        f.writestr("set/b.jpg", png_bytes(110))
        f.writestr("__MACOSX/set/._a.png", b"junk")
        f.writestr("set/readme.txt", b"not an image")
        f.writestr("set/broken.png", b"not really a png")
    files = [("one.png", png_bytes(90)), ("pack.zip", z.getvalue()), ("notes.pdf", b"%PDF")]
    out = list(iter_images(files))
    names = [n for n, _ in out]
    assert names == ["one.png", "set/a.png", "set/b.jpg", "set/broken.png"]
    assert out[3][1] is None and all(im is not None for _, im in out[:3])
    assert len(list(iter_images(files, limit=2))) == 2


def test_run_batch_summary_and_csv():
    seq = iter([fake_result("AUTO_FLAG", ["scratches", "scratches"]), fake_result("PASS", []),
                fake_result("REVIEW", ["crazing"])])
    images = [("a.png", np.zeros((8, 8, 3), np.uint8) + 100), ("face.png", np.zeros((8, 8, 3), np.uint8) + 1),
              ("b.png", np.zeros((8, 8, 3), np.uint8) + 100), ("bad.png", None), ("c.png", np.zeros((8, 8, 3), np.uint8) + 100)]

    def check(im):
        bad = im.mean() < 5
        return SimpleNamespace(ok=not bad, label="face" if bad else "steel", reason="face found" if bad else "ok")

    rows = run_batch(images, inspect=lambda im: next(seq), check=check)
    assert [r["decision"] for r in rows] == ["AUTO_FLAG", "REJECTED", "PASS", "UNREADABLE", "REVIEW"]
    s = summarize(rows)
    assert s["images"] == 5 and s["inspected"] == 3 and s["rejected"] == 1 and s["unreadable"] == 1
    assert s["per_class"] == {"scratches": 2, "crazing": 1}
    assert s["flagged_fraction"] == round(2 / 3, 4) and s["avg_ms"] == 12.0
    csv_text = to_csv(rows).decode()
    assert csv_text.splitlines()[0] == "file,decision,n_defects,max_conf,classes,anomaly_score,ms,note"
    assert "face.png,REJECTED" in csv_text and "_record" not in csv_text


def test_coil_report_is_a_valid_pdf():
    summary = {"coil_id": "COIL-T-01", "length_m": 12.5, "frames": 25, "flagged_fraction": 0.2, "defects_per_100m": 40.0,
               "decisions": {"PASS": 20, "REVIEW": 3, "AUTO_FLAG": 2}, "per_class": {"scratches": 4, "patches": 1},
               "grade_hint": "DOWNGRADE"}
    events = [{"position_m": 1.0 + i, "name": "scratches", "conf": 0.6 + i / 20, "decision": "AUTO_FLAG", "frame": i} for i in range(4)]
    events.append({"position_m": 9.0, "name": "patches", "conf": 0.9, "decision": "REVIEW", "frame": 18})
    pdf = coil_report_pdf(summary, events, CLASSES, {"scratches": "#008300", "patches": "#1baf7a"},
                          ["PASS", "REVIEW", "AUTO_FLAG", "UNKNOWN"], roll_text="no repeating pitch", model_info="best_v1.pt @ 224 px")
    assert pdf[:5] == b"%PDF-" and len(pdf) > 5000
    empty = coil_report_pdf({**summary, "per_class": {}, "decisions": {"PASS": 25}}, [], CLASSES, {}, ["PASS", "REVIEW", "AUTO_FLAG", "UNKNOWN"])
    assert empty[:5] == b"%PDF-"


def test_kpis_from_store(tmp_path):
    store = ResultStore(tmp_path / "t.db")
    for i, (dec, names, coil) in enumerate([("PASS", [], "C1"), ("AUTO_FLAG", ["scratches"], "C1"),
                                             ("REVIEW", ["crazing", "scratches"], "C2"), ("PASS", [], None)]):
        store.log_inspection({"decision": dec, "n_detections": len(names), "classes": names,
                              "detections": [{"name": n, "conf": 0.7, "xyxy": [0, 0, 1, 1]} for n in names],
                              "timings_ms": {"total_ms": 10.0 + i}}, coil_id=coil)
    k = compute_kpis(store.all_inspections())
    assert k["total"] == 4 and k["today"] == 4
    assert k["flagged_fraction"] == 0.5 and k["avg_ms"] == 11.5
    assert k["per_class"] == {"scratches": 2, "crazing": 1}
    assert k["hourly"][23] == 4 and k["hourly_flagged"][23] == 2 and sum(k["hourly"]) == 4
    coils = {c["coil"]: c for c in k["coils"]}
    assert coils["C1"]["frames"] == 2 and coils["C1"]["flagged %"] == 50.0 and coils["C1"]["grade"] == "DOWNGRADE"
    assert coils["C2"]["top defect"] in ("crazing", "scratches")


def test_kpis_empty_and_old_rows():
    assert compute_kpis([])["total"] == 0
    old = {"ts": time.time() - 3 * 86400, "decision": "PASS", "detections": json.dumps([]), "coil_id": None, "total_ms": None}
    k = compute_kpis([old])
    assert k["today"] == 0 and sum(k["hourly"]) == 0 and k["avg_ms"] is None
