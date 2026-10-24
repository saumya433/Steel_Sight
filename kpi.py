"""Plant-level KPIs computed from the audit store (no Streamlit here, so it is testable)."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from typing import Optional

from .coil import grade_hint

FLAGGED = ("AUTO_FLAG", "REVIEW", "UNKNOWN")


def _classes(row: dict) -> list[str]:
    try:
        dets = json.loads(row.get("detections") or "[]")
        return [d["name"] for d in dets if "name" in d]
    except (ValueError, TypeError):
        return []


def compute_kpis(rows: list[dict], now: Optional[float] = None) -> dict:
    """rows = ResultStore.all_inspections(). Returns totals, mixes, a 24 h hourly trend and a per-coil table."""
    now = now or time.time()
    start_today = time.mktime(time.localtime(now)[:3] + (0, 0, 0, 0, 0, -1))
    total = len(rows)
    decisions = Counter(r["decision"] for r in rows)
    per_class: Counter = Counter()
    for r in rows:
        per_class.update(_classes(r))
    flagged = sum(decisions.get(d, 0) for d in FLAGGED)
    ms = [r["total_ms"] for r in rows if r.get("total_ms") is not None]

    hourly = [0] * 24                     # index 23 = the current hour
    hourly_flagged = [0] * 24
    for r in rows:
        age_h = int((now - r["ts"]) // 3600)
        if 0 <= age_h < 24:
            hourly[23 - age_h] += 1
            if r["decision"] in FLAGGED:
                hourly_flagged[23 - age_h] += 1

    coils: dict[str, dict] = defaultdict(lambda: {"frames": 0, "flagged": 0, "classes": Counter(), "last_ts": 0.0})
    for r in rows:
        cid = r.get("coil_id")
        if not cid:
            continue
        c = coils[cid]
        c["frames"] += 1
        c["flagged"] += r["decision"] in FLAGGED
        c["classes"].update(_classes(r))
        c["last_ts"] = max(c["last_ts"], r["ts"])
    coil_table = []
    for cid, c in sorted(coils.items(), key=lambda kv: -kv[1]["last_ts"]):
        frac = c["flagged"] / c["frames"] if c["frames"] else 0.0
        top = c["classes"].most_common(1)
        coil_table.append({"coil": cid, "frames": c["frames"], "flagged %": round(100 * frac, 1),
                           "grade": grade_hint(frac, dict(c["classes"])), "top defect": top[0][0] if top else "-",
                           "last seen": time.strftime("%d %b %H:%M", time.localtime(c["last_ts"]))})
    grades = Counter(c["grade"] for c in coil_table)

    return {
        "total": total,
        "today": sum(1 for r in rows if r["ts"] >= start_today),
        "flagged_fraction": round(flagged / total, 4) if total else 0.0,
        "avg_ms": round(sum(ms) / len(ms), 1) if ms else None,
        "decisions": dict(decisions),
        "per_class": dict(per_class.most_common()),
        "hourly": hourly,
        "hourly_flagged": hourly_flagged,
        "coils": coil_table,
        "grades": dict(grades),
    }
