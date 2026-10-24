"""Coil quality report as a PDF: the document a QA engineer files and a customer can be shown.

Uses reportlab for layout and matplotlib (Agg) for the defect map, so it works headless.
"""

from __future__ import annotations

import io
import time
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

INK = colors.HexColor("#1b2a4a")
MUTED = colors.HexColor("#6b7280")
RULE = colors.HexColor("#d6dbe3")
HEAD_BG = colors.HexColor("#eef2f7")
GRADE_COLOURS = {"PRIME": "#0b8a5b", "SECOND": "#b26b00", "DOWNGRADE": "#c62828"}


def _defect_map_png(events: Sequence[dict], length_m: float, classes: Sequence[str], class_colours: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=160)
    rows = list(classes)[::-1]
    for name in classes:
        xs = [e["position_m"] for e in events if e["name"] == name]
        if xs:
            ax.scatter(xs, [rows.index(name)] * len(xs), s=26, color=class_colours.get(name, "#898781"),
                       edgecolors="white", linewidths=0.8, zorder=3, label=name)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.set_xlim(0, max(1.0, length_m))
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("position along strip (m)", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _table(data: list[list], col_widths: list[float], header: bool = True) -> Table:
    t = Table(data, colWidths=col_widths)
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5)]
    t.setStyle(TableStyle(style))
    return t


def coil_report_pdf(summary: dict, events: Sequence[dict], classes: Sequence[str], class_colours: dict,
                    decision_order: Sequence[str], roll_text: Optional[str] = None, model_info: str = "",
                    line_info: str = "", generated_at: Optional[float] = None) -> bytes:
    """summary = CoilTracker.summary(); events = [CoilEvent.__dict__, ...]."""
    ts = time.strftime("%d %b %Y, %H:%M", time.localtime(generated_at or time.time()))
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=18, textColor=INK, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11, textColor=INK, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.5, textColor=INK, leading=11)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, textColor=MUTED)

    grade = summary.get("grade_hint", "-")
    gcol = GRADE_COLOURS.get(grade, "#1b2a4a")
    story = [
        Paragraph("SteelSight AI · Coil Quality Report", h1),
        Paragraph(f"Coil <b>{summary.get('coil_id', '-')}</b> · generated {ts}" + (f" · {line_info}" if line_info else ""), small),
        Spacer(1, 8),
    ]

    kpi = [["Length inspected", "Frames", "Flagged frames", "Defects / 100 m", "Grade (indicative)"],
           [f"{summary.get('length_m', 0):.1f} m", str(summary.get("frames", 0)),
            f"{100 * summary.get('flagged_fraction', 0):.1f}%", f"{summary.get('defects_per_100m', 0):,.1f}",
            Paragraph(f"<font color='{gcol}'><b>{grade}</b></font>", body)]]
    story.append(_table(kpi, [34 * mm] * 5))

    story.append(Paragraph("Decisions per frame", h2))
    dec = summary.get("decisions", {})
    story.append(_table([list(d.replace("_", " ") for d in decision_order), [str(dec.get(d, 0)) for d in decision_order]],
                        [42 * mm] * len(decision_order)))

    story.append(Paragraph("Defects by class", h2))
    rows = [["Class", "Count", "Max confidence"]]
    for name in classes:
        confs = [e["conf"] for e in events if e["name"] == name]
        if confs:
            rows.append([name, str(len(confs)), f"{max(confs):.2f}"])
    if len(rows) == 1:
        rows.append(["no defects detected", "0", "-"])
    story.append(_table(rows, [70 * mm, 30 * mm, 40 * mm]))

    story.append(Paragraph("Defect map along the coil", h2))
    png = _defect_map_png(events, summary.get("length_m", 0.0), classes, class_colours)
    story.append(Image(io.BytesIO(png), width=172 * mm, height=62 * mm))

    story.append(Paragraph("Most confident defects", h2))
    top = sorted(events, key=lambda e: -e["conf"])[:12]
    rows = [["Position (m)", "Class", "Confidence", "Decision", "Frame"]]
    rows += [[f"{e['position_m']:.2f}", e["name"], f"{e['conf']:.2f}", e["decision"].replace("_", " "), str(e["frame"])] for e in top]
    if len(rows) == 1:
        rows.append(["-", "-", "-", "-", "-"])
    story.append(_table(rows, [30 * mm, 50 * mm, 28 * mm, 34 * mm, 28 * mm]))

    story.append(Paragraph("Roll diagnosis", h2))
    story.append(Paragraph(roll_text or "No repeating pitch among confident detections on this coil.", body))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Notes: the grade is an indicative rule based on flagged-frame density (thresholds to be calibrated against QA). "
        "Detections are model outputs; the QA engineer's verdict is final." + (f" Model: {model_info}." if model_info else ""), small))
    story.append(Spacer(1, 18))
    sig = Table([["Inspected by: ____________________", "QA approved by: ____________________", "Date: ____________"]],
                colWidths=[62 * mm, 66 * mm, 42 * mm])
    sig.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 8.5), ("TEXTCOLOR", (0, 0), (-1, -1), INK)]))
    story.append(sig)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"Coil report {summary.get('coil_id', '')}", author="SteelSight AI")
    doc.build(story)
    return buf.getvalue()
