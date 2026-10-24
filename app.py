"""SteelSight AI - inspection dashboard.

    streamlit run app.py

Tabs
    Inspect        upload a frame -> Stage 1 + Stage 2 -> gate decision, boxes, heatmap, timings
    Live line      replay a folder of frames as a moving coil, build the defect map, roll diagnosis
    Operator review  confirm / reject / relabel REVIEW and UNKNOWN frames -> export training labels
    Model card     measured metrics, per-class thresholds, confusion matrix, curves
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from steelsight.batch import MAX_IMAGES, iter_images, run_batch, summarize, to_csv
from steelsight.coil import DEFAULT_ROLL_TABLE_MM, CoilEvent, CoilTracker, best_diagnosis, diagnose_by_class
from steelsight.kpi import compute_kpis
from steelsight.report import coil_report_pdf
from steelsight.data import list_images, load_classes
from steelsight.drift import DriftMonitor
from steelsight.frames import prepare_frame
from steelsight.gate import DECISION_COLOURS_HEX, DECISION_ICONS, Decision, GateConfig
from steelsight.pipeline import InspectionPipeline
from steelsight.scene_gate import SceneGate
from steelsight.store import ResultStore

# ----------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parent
WEIGHTS_DEFAULT = os.environ.get("STEELSIGHT_WEIGHTS", "models/best_v1.pt")
BANK_DEFAULT = os.environ.get("STEELSIGHT_BANK", "models/anomaly_bank.pt")
THRESHOLDS = ROOT / "config" / "thresholds.json"
DB_PATH = os.environ.get("STEELSIGHT_DB", "data/steelsight.db")
# Must equal the size the detector was TRAINED at (Ultralytics rounds up to a multiple of 32:
# trained with imgsz=200 -> 224; trained with imgsz=640 -> 640).
IMGSZ_DEFAULT = int(os.environ.get("STEELSIGHT_IMGSZ", "640"))
CLASSES = load_classes(ROOT / "config" / "data.yaml")

# Categorical palette in fixed order: colour follows the class, never its rank.
CLASS_COLOURS = dict(zip(CLASSES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]))
DECISIONS = [Decision.PASS, Decision.REVIEW, Decision.AUTO_FLAG, Decision.UNKNOWN]
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13),
    margin=dict(l=10, r=10, t=36, b=10), hoverlabel=dict(font_size=13),
)
GRID = dict(gridcolor="rgba(137,135,129,0.25)", zerolinecolor="rgba(137,135,129,0.35)")

st.set_page_config(page_title="SteelSight AI", page_icon="🔩", layout="wide")

# Streamlit renamed use_container_width -> width="stretch" in 1.49; support both.
_ver = tuple(int(x) for x in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if _ver >= (1, 49) else {"use_container_width": True}


# ----------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading models...")
def load_pipeline(weights: str, bank: str, thresholds_mtime: float, device: str | None, imgsz: int):
    gate = GateConfig.load(THRESHOLDS) if THRESHOLDS.exists() else GateConfig()
    return InspectionPipeline(weights, gate, anomaly_bank=bank if Path(bank).exists() else None,
                              imgsz=imgsz, device=device)


@st.cache_resource(show_spinner="Loading scene check (the first CLIP run downloads ~600 MB)...")
def load_scene_gate(use_face: bool, use_clip: bool, min_steel: float, device: str | None) -> SceneGate:
    return SceneGate(use_face=use_face, use_clip=use_clip, clip_min_steel=min_steel, device=device)


@st.cache_resource
def get_store(path: str) -> ResultStore:
    return ResultStore(path)


@st.cache_data(show_spinner="Building the coil report...")
def make_coil_report(summary_json: str, events_json: str, roll_text: str | None, model_info: str, line_info: str) -> bytes:
    return coil_report_pdf(json.loads(summary_json), json.loads(events_json), CLASSES, CLASS_COLOURS,
                           [d.value for d in DECISIONS], roll_text=roll_text, model_info=model_info, line_info=line_info)


def bgr_to_rgb(im: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im


def decision_badge(d: Decision) -> str:
    return f"{DECISION_ICONS[d]} **{d.value}**"


def sample_images() -> list[Path]:
    for folder in (ROOT / "samples", ROOT / "YOLO" / "val" / "images"):
        if folder.exists():
            imgs = list_images(folder)
            if imgs:
                return imgs
    return []


def read_upload(uploaded) -> np.ndarray:
    data = np.frombuffer(uploaded.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def heatmap_rgb(im_bgr: np.ndarray, heat: np.ndarray) -> np.ndarray:
    h = heat - heat.min()
    h = (255 * h / (h.max() + 1e-6)).astype(np.uint8)
    cmap = cv2.applyColorMap(h, cv2.COLORMAP_INFERNO)
    base = im_bgr if im_bgr.ndim == 3 else cv2.cvtColor(im_bgr, cv2.COLOR_GRAY2BGR)
    return bgr_to_rgb(cv2.addWeighted(base, 0.5, cmap, 0.5, 0))


def draw_from_record(image_path: str, detections_json: str) -> np.ndarray | None:
    im = cv2.imread(image_path)
    if im is None:
        return None
    for d in json.loads(detections_json or "[]"):
        x1, y1, x2, y2 = (int(v) for v in d["xyxy"])
        cv2.rectangle(im, (x1, y1), (x2, y2), (25, 178, 250), 2)
        cv2.putText(im, f"{d['name']} {d['conf']:.2f}", (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr_to_rgb(im)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("SteelSight AI")
    st.caption("Two-stage surface defect inspection · human in the loop")
    weights = st.text_input("Detector weights", WEIGHTS_DEFAULT)
    bank = st.text_input("Anomaly bank (Stage 1)", BANK_DEFAULT)
    device = st.selectbox("Device", ["auto", "cpu", "0"], index=0)
    device_arg = None if device == "auto" else device
    imgsz = st.number_input("Inference size (px)", 160, 1280, IMGSZ_DEFAULT, 32,
                            help="Must equal the size the model was trained at. Trained with imgsz=200 -> use 224. "
                                 "Trained with imgsz=640 -> 640. A mismatch silently lowers accuracy.")
    st.divider()
    st.subheader("Is it steel? (input check)")
    scene_on = st.toggle("Reject faces / blank frames", value=True,
                         help="The detector only knows 6 defect types, so on its own it will draw boxes on anything. "
                              "This check runs first. A rejected image is never inspected or logged.")
    clip_on = st.toggle("Also reject other non-steel scenes (CLIP)", value=False,
                        help="Zero-shot check against 'metal surface' vs face / room / desk / screen ... "
                             "Downloads a ~600 MB model on first use.")
    clip_min = st.slider("Steel match needed", 0.10, 0.95, 0.50, 0.05, disabled=not clip_on)
    prep_on = st.toggle("Match training look (square crop + grayscale)", value=True,
                        help="Phone photos are large colour rectangles; the model was trained on small grayscale squares.")
    st.divider()
    st.subheader("Confidence gate")
    use_per_class = st.toggle("Per-class thresholds from config", value=True,
                              help="Each class gets its own REVIEW / AUTO floor from the validation F1 curve.")
    review_min = st.slider("Review floor (default)", 0.05, 0.95, 0.25, 0.01)
    auto_flag = st.slider("Auto-flag floor (default)", 0.05, 0.99, 0.60, 0.01)
    stage1_on = st.toggle("Stage 1 anomaly screen", value=True)
    st.divider()
    st.subheader("Line")
    line_speed = st.number_input("Line speed (m/min)", 50.0, 1200.0, 300.0, 10.0)
    fps = st.number_input("Camera frames / s", 1.0, 200.0, 20.0, 1.0)
    st.caption(f"Each frame covers {line_speed / 60 / fps:.3f} m of strip. Budget per frame: {1000 / fps:.0f} ms.")

weights_ok = Path(weights).exists()
pipe = None
if weights_ok:
    mtime = THRESHOLDS.stat().st_mtime if THRESHOLDS.exists() else 0.0
    pipe = load_pipeline(weights, bank, mtime, device_arg, int(imgsz))
    cfg = pipe.gate_cfg
    cfg.review_min, cfg.auto_flag = review_min, auto_flag
    if not use_per_class:
        cfg.per_class_review, cfg.per_class_auto = {}, {}
    elif not cfg.per_class_review and THRESHOLDS.exists():
        base = GateConfig.load(THRESHOLDS)
        cfg.per_class_review, cfg.per_class_auto = base.per_class_review, base.per_class_auto
    if not stage1_on:
        cfg.anomaly_threshold = None
    elif pipe.anomaly is not None and cfg.anomaly_threshold is None:
        cfg.anomaly_threshold = pipe.anomaly.threshold
store = get_store(DB_PATH)

tab_overview, tab_inspect, tab_batch, tab_line, tab_review, tab_card = st.tabs(
    ["Overview", "Inspect", "Batch", "Live line", "Operator review", "Model card"])

# ============================================================================= INSPECT
with tab_inspect:
    st.subheader("Upload a frame, get a decision")
    if not weights_ok:
        st.error(f"Detector weights not found at `{weights}`. Copy your trained `best.pt` there (see README) or set STEELSIGHT_WEIGHTS.")
    scene_gate = load_scene_gate(True, clip_on, float(clip_min), device_arg) if scene_on else None
    if scene_gate is not None:
        st.caption("Input check: " + scene_gate.status)
    else:
        st.caption("Input check is OFF - the detector will draw boxes on any image, including faces.")

    samples = sample_images()
    left, right = st.columns([1, 1])
    with left:
        source = st.radio("Source", ["Upload / phone camera", "Browser camera", "Sample"], horizontal=True,
                          help="On a phone, 'Upload / phone camera' -> Browse files -> Camera opens the native camera app "
                               "(rear camera). 'Browser camera' uses the browser's default camera (front on most phones).")
        image = None
        if source == "Upload / phone camera":
            up = st.file_uploader("Image (jpg/png) - on a phone choose Camera", type=["jpg", "jpeg", "png", "bmp"])
            image = read_upload(up) if up is not None else None
        elif source == "Browser camera":
            shot = st.camera_input("Take a photo")
            image = read_upload(shot) if shot is not None else None
        else:
            pick = st.selectbox("Pick a sample", ["(none)"] + [p.name for p in samples]) if samples else "(none)"
            if pick != "(none)":
                image = cv2.imread(str(next(p for p in samples if p.name == pick)))

        verdict = scene_gate.check(image) if (scene_gate is not None and image is not None) else None
        if image is not None:
            if verdict is not None and verdict.label == "face":
                st.warning("A face was detected, so the image is not previewed here and will not be inspected or logged.")
            else:
                st.image(bgr_to_rgb(image), caption="input frame", **WIDE)

    with right:
        if image is not None and pipe is not None and verdict is not None and not verdict.ok:
            titles = {"face": "FACE - NOT INSPECTED", "not_steel": "NOT A STEEL SURFACE - NOT INSPECTED",
                      "unusable": "UNUSABLE IMAGE - NOT INSPECTED"}
            st.markdown(f"## ⛔ {titles.get(verdict.label, 'NOT INSPECTED')}")
            st.error(verdict.reason)
            st.caption("The defect detector was not run, nothing was logged. Capture a close-up of the steel strip "
                       "under even light and try again.")
        elif image is not None and pipe is not None:
            if verdict is not None:
                st.caption("Input check: " + verdict.reason)
                for w in verdict.warnings:
                    st.warning(w)
            frame = prepare_frame(image) if prep_on else image
            res = pipe.inspect(frame, draw_heatmap=True)
            g = res.gate
            st.markdown(f"## {decision_badge(res.decision)}")
            st.write(g.reason)
            c1, c2, c3 = st.columns(3)
            c1.metric("Detections kept", len(g.kept))
            c2.metric("Max confidence", f"{g.max_conf:.2f}")
            if res.anomaly_score is not None:
                thr = pipe.gate_cfg.anomaly_threshold
                c3.metric("Anomaly score", f"{res.anomaly_score:.2f}", delta=f"threshold {thr:.2f}" if thr else None, delta_color="off")
            else:
                c3.metric("Anomaly score", "off")
            st.image(res.annotated[:, :, ::-1], caption="Stage 2 detections + gate decision", **WIDE)
            if res.heatmap is not None:
                with st.expander("Stage 1 anomaly heatmap (where the surface looks abnormal)"):
                    st.image(heatmap_rgb(frame, res.heatmap), **WIDE)
            t = res.timings_ms
            st.caption(
                f"Stage 1 {t.get('stage1_ms', 0):.0f} ms · Stage 2 {t.get('stage2_ms', 0):.0f} ms "
                f"(inference {t.get('stage2_inference_ms', float('nan')):.1f} ms) · total {t['total_ms']:.0f} ms"
            )
            if g.kept:
                st.dataframe(pd.DataFrame([{"class": d.name, "conf": round(d.conf, 3),
                                            "floor": pipe.gate_cfg.review_floor(d.name),
                                            "auto": "yes" if d in g.auto else "", "box": [round(v) for v in d.xyxy]} for d in g.kept]),
                             hide_index=True, **WIDE)
            if st.button("Log this inspection", type="primary"):
                store.log_inspection(res.to_record(), coil_id="MANUAL", image_path=None)
                st.success("logged")
        elif image is None:
            st.info("Upload a frame or pick a sample to inspect it.")

# ============================================================================= BATCH
with tab_batch:
    st.subheader("Inspect a batch of images")
    st.caption(f"Upload up to {MAX_IMAGES} images, or a .zip of images. Each image goes through the input check, "
               "the detector and the confidence gate. Download the results as a CSV.")
    ups = st.file_uploader("Images or a .zip", type=["jpg", "jpeg", "png", "bmp", "zip"], accept_multiple_files=True,
                           key="batch_files")
    bA, bB = st.columns([1, 1])
    batch_id = bA.text_input("Batch ID", f"BATCH-{time.strftime('%Y%m%d-%H%M')}")
    log_batch = bB.toggle("Log results to the audit store", value=True,
                          help="Logged results appear on the Overview tab, and REVIEW / UNKNOWN images join the Operator review queue.")
    if st.button("▶ Run batch", type="primary", disabled=(pipe is None or not ups)):
        imgs = list(iter_images([(u.name, u.getvalue()) for u in ups]))
        if not imgs:
            st.error("No images found in the upload.")
        else:
            prog = st.progress(0.0)
            rows = run_batch(
                imgs, inspect=lambda f: pipe.inspect(f, draw_heatmap=False),
                check=scene_gate.check if scene_gate is not None else None,
                prepare=prepare_frame if prep_on else None,
                on_progress=lambda i, n: prog.progress((i + 1) / len(imgs), text=f"{i + 1} / {len(imgs)} · {n}"))
            if log_batch:
                for i, r in enumerate(rows):
                    if r["_record"] is not None:
                        store.log_inspection(r["_record"], coil_id=batch_id, frame=i, image_path=None)
            st.session_state["batch_run"] = {"id": batch_id, "rows": rows, "summary": summarize(rows)}

    br = st.session_state.get("batch_run")
    if br:
        s = br["summary"]
        st.markdown(f"#### {br['id']}")
        n1, n2, n3, n4, n5 = st.columns(5)
        n1.metric("Images", s["images"])
        n2.metric("Inspected", s["inspected"])
        n3.metric("Flagged", f"{100 * s['flagged_fraction']:.0f}%")
        n4.metric("Rejected by input check", s["rejected"] + s["unreadable"])
        n5.metric("Avg time / image", f"{s['avg_ms']:.0f} ms" if s["avg_ms"] is not None else "–")
        g1, g2 = st.columns([1, 1])
        with g1:
            order = [d.value for d in DECISIONS]
            counts = [s["decisions"].get(d, 0) for d in order]
            fig = go.Figure(go.Bar(x=[d.replace("_", " ") for d in order], y=counts, text=counts, textposition="outside",
                                   marker=dict(color=[DECISION_COLOURS_HEX[d] for d in DECISIONS], cornerradius=4),
                                   hovertemplate="%{x}: %{y} images<extra></extra>"))
            fig.update_layout(title="Decisions", height=320, showlegend=False, **PLOT_LAYOUT)
            fig.update_yaxes(**GRID, title="images", range=[0, max(counts + [1]) * 1.2]); fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, **WIDE)
        with g2:
            present = [c for c in CLASSES if s["per_class"].get(c)]
            if present:
                vals = [s["per_class"][c] for c in present]
                fig = go.Figure(go.Bar(x=vals, y=present, orientation="h", text=vals, textposition="outside",
                                       marker=dict(color=[CLASS_COLOURS[c] for c in present], cornerradius=4),
                                       hovertemplate="%{y}: %{x} defects<extra></extra>"))
                fig.update_layout(title="Defects by class", height=320, showlegend=False, **PLOT_LAYOUT)
                fig.update_xaxes(**GRID, title="defects", range=[0, max(vals) * 1.2])   # room for the value labels
                fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=present[::-1])
                st.plotly_chart(fig, **WIDE)
            else:
                st.info("No defects kept by the gate in this batch.")
        table = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in br["rows"]])
        st.dataframe(table, hide_index=True, **WIDE)
        st.download_button("⬇ Download results (CSV)", to_csv(br["rows"]), file_name=f"{br['id']}.csv", mime="text/csv")

# ============================================================================= LIVE LINE
with tab_line:
    st.subheader("Replay frames as a moving coil")
    colA, colB, colC = st.columns([2, 1, 1])
    folder = colA.text_input("Frames folder", "samples" if (ROOT / "samples").exists() else "YOLO/val/images")
    coil_id = colB.text_input("Coil ID", f"COIL-{time.strftime('%Y%m%d')}-01")
    n_frames = colC.number_input("Frames", 5, 2000, 60, 5)
    run = st.button("▶ Run line", type="primary", disabled=pipe is None)

    if run and pipe is not None:
        frames = list_images(folder)[: int(n_frames)]
        if not frames:
            st.error(f"no images in {folder}")
        else:
            tracker = CoilTracker(coil_id, line_speed_m_per_min=line_speed, fps=fps)
            drift = DriftMonitor(baseline_n=max(10, len(frames) // 3), window=max(10, len(frames) // 5))
            prog = st.progress(0.0)
            live_img, live_txt = st.columns([1, 1])
            img_slot, txt_slot = live_img.empty(), live_txt.empty()
            records = []
            for i, p in enumerate(frames):
                im = cv2.imread(str(p))
                res = pipe.inspect(im, draw_heatmap=False)
                pos = tracker.add_frame(i, res.decision.value, res.gate.kept, image_h=im.shape[0])
                store.log_inspection(res.to_record(), coil_id=coil_id, frame=i, position_m=pos, image_path=str(p))
                drift.update(anomaly=res.anomaly_score, confidence=res.gate.max_conf or None)
                records.append({"frame": i, "position_m": pos, "decision": res.decision.value,
                                "anomaly": res.anomaly_score, "max_conf": res.gate.max_conf, "ms": res.timings_ms["total_ms"]})
                if i % 2 == 0 or i == len(frames) - 1:
                    img_slot.image(res.annotated[:, :, ::-1], caption=f"frame {i} · {pos:.1f} m", **WIDE)
                    txt_slot.markdown(f"### {decision_badge(res.decision)}\n{res.gate.reason}\n\n`{res.timings_ms['total_ms']:.0f} ms`")
                prog.progress((i + 1) / len(frames))
            st.session_state["line_run"] = {"coil": coil_id, "summary": tracker.summary(),
                                            "events": [e.__dict__ for e in tracker.events], "records": records,
                                            "drift": {m: (s.message if (s := drift.status(m)) else "collecting baseline") for m in ("anomaly", "confidence")}}

    run_data = st.session_state.get("line_run")
    if run_data:
        s = run_data["summary"]
        st.markdown(f"#### Coil {run_data['coil']}")
        m1, m2, m3, m4, m5 = st.columns([1, 0.8, 1, 1.1, 1.5])
        m1.metric("Length inspected", f"{s['length_m']:.1f} m")
        m2.metric("Frames", s["frames"])
        m3.metric("Flagged frames", f"{100 * s['flagged_fraction']:.0f}%")
        m4.metric("Defects / 100 m", f"{s['defects_per_100m']:,.0f}")
        m5.metric("Grade hint", s["grade_hint"])

        ev = pd.DataFrame(run_data["events"])
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = go.Figure()
            if len(ev):
                for name in CLASSES:
                    sub = ev[ev["name"] == name]
                    if len(sub):
                        fig.add_trace(go.Scatter(
                            x=sub["position_m"], y=[name] * len(sub), mode="markers", name=name,
                            marker=dict(size=10, color=CLASS_COLOURS.get(name, "#898781"), line=dict(width=2, color="rgba(252,252,251,0.9)")),
                            customdata=np.stack([sub["conf"], sub["decision"]], axis=1),
                            hovertemplate="%{y}<br>%{x:.2f} m<br>conf %{customdata[0]:.2f} · %{customdata[1]}<extra></extra>"))
            fig.update_layout(title="Defect map along the coil", xaxis_title="position along strip (m)", height=360,
                              legend=dict(orientation="h", y=-0.25), **PLOT_LAYOUT)
            fig.update_xaxes(**GRID, range=[0, max(1.0, s["length_m"])])
            fig.update_yaxes(**GRID, categoryorder="array", categoryarray=CLASSES[::-1])
            st.plotly_chart(fig, **WIDE)
        with c2:
            counts = [s["decisions"].get(d.value, 0) for d in DECISIONS]
            fig2 = go.Figure(go.Bar(x=[d.value.replace("_", " ") for d in DECISIONS], y=counts,
                                    marker=dict(color=[DECISION_COLOURS_HEX[d] for d in DECISIONS], cornerradius=4),
                                    text=counts, textposition="outside", hovertemplate="%{x}: %{y} frames<extra></extra>"))
            fig2.update_layout(title="Decisions", height=360, showlegend=False, **PLOT_LAYOUT)
            fig2.update_yaxes(**GRID, title="frames")
            fig2.update_xaxes(showgrid=False, tickangle=0, tickfont=dict(size=11))
            st.plotly_chart(fig2, **WIDE)

        st.markdown("#### Roll diagnosis")
        rc1, rc2 = st.columns([1, 1])
        with rc1:
            plant = st.number_input("Demo: plant a synthetic repeating defect every N m (0 = off)", 0.0, 20.0, 0.0, 0.1,
                                    help="Shows the mechanism. Clearly labelled synthetic; never present as a measurement.")
            events = [CoilEvent(**e) for e in run_data["events"]]
            planted = [k * plant for k in range(1, int(s["length_m"] / plant) + 1)] if plant > 0 else None
            diags = diagnose_by_class(events, DEFAULT_ROLL_TABLE_MM, extra_positions=planted, n_frames=s["frames"])
            best = best_diagnosis(diags)
            if best:
                name, diag = best
                tag = " · **SYNTHETIC DEMO**" if name == "synthetic" else ""
                st.success(f"**{name}**: {diag.message}{tag}")
                st.caption("Per class, confident detections only. pitch = one roll circumference → diameter = pitch / π → nearest roll in the roll table")
            else:
                st.info("No repeating pitch among confident detections on this coil (that is the normal case: most defects are not periodic).")
            with st.expander("All classes"):
                for name, d in diags.items():
                    st.write(f"• {name}: {d.message}")
        with rc2:
            st.caption("Drift monitor (recent window vs baseline)")
            for m, msg in run_data["drift"].items():
                st.write(f"• {msg}")

        roll_text = None
        if best:
            bname, bdiag = best
            roll_text = f"{bname}: {bdiag.message}" + (" (SYNTHETIC DEMO, not a measurement)" if bname == "synthetic" else "")
        pdf_bytes = make_coil_report(json.dumps(s, default=str), json.dumps(run_data["events"], default=str), roll_text,
                                     f"{Path(weights).name} @ {int(imgsz)} px",
                                     f"line speed {line_speed:.0f} m/min · {fps:.0f} frames/s")
        st.download_button("⬇ Download coil quality report (PDF)", pdf_bytes, file_name=f"{run_data['coil']}_report.pdf",
                           mime="application/pdf", type="primary")

        rec = pd.DataFrame(run_data["records"])
        if rec["anomaly"].notna().any():
            fig3 = go.Figure(go.Scatter(x=rec["frame"], y=rec["anomaly"], mode="lines", line=dict(width=2, color="#2a78d6"),
                                        hovertemplate="frame %{x}<br>anomaly %{y:.3f}<extra></extra>", name="anomaly score"))
            thr = pipe.gate_cfg.anomaly_threshold if pipe else None
            if thr:
                fig3.add_hline(y=thr, line=dict(color="#ec835a", width=2, dash="dot"), annotation_text="Stage 1 threshold")
            fig3.update_layout(title="Stage 1 anomaly score per frame", height=260, **PLOT_LAYOUT)
            fig3.update_xaxes(**GRID, title="frame"); fig3.update_yaxes(**GRID)
            st.plotly_chart(fig3, **WIDE)
        with st.expander("Frame log"):
            st.dataframe(rec, hide_index=True, **WIDE)

# ============================================================================= OPERATOR REVIEW
with tab_review:
    st.subheader("Operator review queue")
    st.caption("REVIEW and UNKNOWN frames wait here. Every verdict is stored for audit and becomes training data.")
    stats = store.review_stats()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Pending", len(store.pending_reviews(1000)))
    k2.metric("Confirmed", stats.get("CONFIRM", 0) + stats.get("RELABEL", 0))
    k3.metric("Rejected", stats.get("REJECT", 0))
    k4.metric("Precision on reviewed", f"{stats['precision_on_reviewed']:.0%}" if stats.get("precision_on_reviewed") is not None else "–")

    pending = store.pending_reviews(12)
    if not pending:
        st.info("Queue is empty. Run the Live line tab to generate frames.")
    for row in pending:
        with st.container(border=True):
            a, b = st.columns([1, 2])
            with a:
                img = draw_from_record(row["image_path"], row["detections"]) if row["image_path"] else None
                if img is not None:
                    st.image(img, **WIDE)
                else:
                    st.caption("(image not available)")
            with b:
                d = Decision(row["decision"])
                st.markdown(f"{decision_badge(d)} · coil `{row['coil_id']}` · frame {row['frame']} · {row['position_m'] or 0:.1f} m")
                st.write(row["reason"])
                cls = st.selectbox("Correct class", CLASSES, key=f"cls{row['id']}")
                bc1, bc2, bc3 = st.columns(3)
                if bc1.button("Confirm", key=f"c{row['id']}", type="primary"):
                    store.log_review(row["id"], "CONFIRM"); st.rerun()
                if bc2.button("Relabel", key=f"l{row['id']}"):
                    store.log_review(row["id"], "RELABEL", corrected_class=cls); st.rerun()
                if bc3.button("Reject", key=f"r{row['id']}"):
                    store.log_review(row["id"], "REJECT"); st.rerun()
    st.divider()
    if st.button("Export confirmed verdicts as YOLO labels (data/retrain/labels)"):
        n = store.export_training_labels("data/retrain/labels", CLASSES, image_hw=(200, 200))
        st.success(f"wrote {n} label files for the next fine-tune")

# ============================================================================= MODEL CARD
with tab_card:
    st.subheader("Model card: measured, not claimed")
    rep = ROOT / "reports" / "eval.json"
    an = ROOT / "reports" / "anomaly.json"
    if rep.exists():
        r = json.loads(rep.read_text())
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("mAP@50", f"{100 * r['mAP50']:.1f}%")
        c2.metric("mAP@50-95", f"{100 * r['mAP50_95']:.1f}%")
        c3.metric("Precision", f"{100 * r['precision']:.1f}%")
        c4.metric("Recall", f"{100 * r['recall']:.1f}%")
        c5.metric("Inference", f"{r['speed_ms'].get('inference', float('nan')):.1f} ms")
        st.caption(f"weights `{r['weights']}` · {r['val_images']} validation images")
        per = r.get("per_class", {})
        if per:
            df = pd.DataFrame(per).T.reset_index().rename(columns={"index": "class"})
            fig = go.Figure(go.Bar(x=df["class"], y=df["ap50"], marker=dict(color="#2a78d6", cornerradius=4),
                                   text=[f"{v:.2f}" for v in df["ap50"]], textposition="outside",
                                   hovertemplate="%{x}: AP50 %{y:.3f}<extra></extra>"))
            fig.update_layout(title="AP@50 per class", height=320, showlegend=False, **PLOT_LAYOUT)
            fig.update_yaxes(**GRID, range=[0, 1.08]); fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, **WIDE)
            st.dataframe(df, hide_index=True, **WIDE)
    else:
        st.info("No reports/eval.json yet. Run `python scripts/evaluate.py --weights models/best_v1.pt`.")
        if THRESHOLDS.exists():
            st.json(json.loads(THRESHOLDS.read_text()))
    if an.exists():
        a = json.loads(an.read_text())
        st.markdown("#### Stage 1 anomaly screen")
        d1, d2, d3 = st.columns(3)
        d1.metric("Patch AUROC (defect vs normal)", f"{a.get('patch_auroc_defect_vs_normal', 0):.3f}")
        d2.metric("Defective images flagged", f"{100 * a.get('image_recall_at_budget', 0):.0f}%")
        d3.metric("Normal-patch false-alarm budget", a.get("normal_patch_false_alarm_budget", "–"))
    plots = [p for p in (ROOT / "reports").glob("*.png")] if (ROOT / "reports").exists() else []
    if plots:
        st.markdown("#### Validation plots")
        cols = st.columns(3)
        for i, p in enumerate(sorted(plots)):
            cols[i % 3].image(str(p), caption=p.stem, **WIDE)
    st.markdown("""
#### Known limits (say them before the jury does)
- NEU-DET is hot-rolled carbon steel at 200×200 px. Every number here is a benchmark result, not a stainless-line result.
- No defect-free images exist in NEU-DET, so Stage 1 is trained on background regions and its clean-strip false-alarm rate is a design budget until shadow mode.
- Crazing (fine hairline cracks) is the weak class; it needs a per-class threshold and higher camera resolution.
""")

# ============================================================================= OVERVIEW
# Written last so it includes anything logged earlier in this same run.
with tab_overview:
    st.subheader("Plant overview")
    st.caption("Everything logged by the Inspect, Batch and Live line tabs, from the audit store.")
    k = compute_kpis(store.all_inspections())
    if k["total"] == 0:
        st.info("Nothing logged yet. Run the **Live line** or a **Batch**, or log an **Inspect** result, and this page fills in.")
    else:
        o1, o2, o3, o4, o5 = st.columns(5)
        o1.metric("Inspections", f"{k['total']:,}")
        o2.metric("Today", f"{k['today']:,}")
        o3.metric("Flagged", f"{100 * k['flagged_fraction']:.1f}%")
        o4.metric("Avg time / image", f"{k['avg_ms']:.0f} ms" if k["avg_ms"] is not None else "–")
        o5.metric("Coils / batches", len(k["coils"]))

        p1, p2 = st.columns([1, 1])
        with p1:
            order = [d.value for d in DECISIONS]
            counts = [k["decisions"].get(d, 0) for d in order]
            fig = go.Figure(go.Bar(x=[d.replace("_", " ") for d in order], y=counts, text=counts, textposition="outside",
                                   marker=dict(color=[DECISION_COLOURS_HEX[d] for d in DECISIONS], cornerradius=4),
                                   hovertemplate="%{x}: %{y} frames<extra></extra>"))
            fig.update_layout(title="Decision mix", height=320, showlegend=False, **PLOT_LAYOUT)
            fig.update_yaxes(**GRID, title="frames", range=[0, max(counts + [1]) * 1.2]); fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, **WIDE)
        with p2:
            present = [c for c in CLASSES if k["per_class"].get(c)]
            if present:
                vals = [k["per_class"][c] for c in present]
                fig = go.Figure(go.Bar(x=vals, y=present, orientation="h", text=vals, textposition="outside",
                                       marker=dict(color=[CLASS_COLOURS[c] for c in present], cornerradius=4),
                                       hovertemplate="%{y}: %{x} defects<extra></extra>"))
                fig.update_layout(title="Defects by class", height=320, showlegend=False, **PLOT_LAYOUT)
                fig.update_xaxes(**GRID, title="defects", range=[0, max(vals) * 1.2])   # room for the value labels
                fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=present[::-1])
                st.plotly_chart(fig, **WIDE)
            else:
                st.info("No defects logged yet.")

        hours = [time.strftime("%H:00", time.localtime(time.time() - (23 - i) * 3600)) for i in range(24)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=k["hourly"], mode="lines+markers", name="inspected",
                                 line=dict(width=2, color="#2a78d6"), marker=dict(size=8),
                                 hovertemplate="%{x}<br>%{y} inspected<extra></extra>"))
        fig.add_trace(go.Scatter(x=hours, y=k["hourly_flagged"], mode="lines+markers", name="flagged",
                                 line=dict(width=2, color="#eb6834"), marker=dict(size=8),
                                 hovertemplate="%{x}<br>%{y} flagged<extra></extra>"))
        fig.update_layout(title="Last 24 hours", height=300, legend=dict(orientation="h", y=-0.3), **PLOT_LAYOUT)
        fig.update_xaxes(**GRID); fig.update_yaxes(**GRID, title="frames", rangemode="tozero")
        st.plotly_chart(fig, **WIDE)

        st.markdown("#### Coils and batches")
        if k["coils"]:
            g = k["grades"]
            q1, q2, q3 = st.columns(3)
            q1.metric("PRIME", g.get("PRIME", 0))
            q2.metric("SECOND", g.get("SECOND", 0))
            q3.metric("DOWNGRADE", g.get("DOWNGRADE", 0))
            coil_df = pd.DataFrame(k["coils"])
            st.dataframe(coil_df, hide_index=True, **WIDE)
            st.download_button("⬇ Download coil summary (CSV)", coil_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"coil_summary_{time.strftime('%Y%m%d')}.csv", mime="text/csv")
            st.caption("Grades are an indicative rule on flagged-frame density (thresholds to be calibrated against QA).")
        else:
            st.caption("Single inspections logged without a coil ID are counted above but not graded.")
