"""Shareable alert text (WhatsApp/SMS) and CAP 1.2 XML for official alerting systems.

CAP (Common Alerting Protocol, OASIS standard) is the format used by systems such as
India's NDMA SACHET and US IPAWS, so an officer can forward the alert without retyping.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
from xml.etree import ElementTree as ET

from .i18n import actions, t

CFS_PER_CMS = 35.3147
CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
CAP_LANG = {"en": "en-US", "hi": "hi-IN", "kn": "kn-IN"}
CAP_SEVERITY = {"normal": "Minor", "elevated": "Minor", "warning": "Moderate", "danger": "Severe",
                "extreme": "Extreme", "unknown": "Unknown"}
CAP_URGENCY = {"normal": "Future", "elevated": "Expected", "warning": "Expected", "danger": "Immediate",
               "extreme": "Immediate", "unknown": "Unknown"}
ALERT_VALID_HOURS = 24
QUIET_LEVELS = ("normal", "unknown")


def fmt_cfs(value: float) -> str:
    return f"{value:,.0f}"


def _headline_body(river: str, level: str, peak_date: str, peak_cfs: float, lang: str, action: str) -> tuple[str, str]:
    """'Normal' and 'no record' get their own wording: a flood headline would mislead there."""
    suffix = f"_{level}" if level in QUIET_LEVELS else ""
    fields = {"level": t(f"level.{level}", lang), "river": river, "date": peak_date, "cfs": fmt_cfs(peak_cfs),
              "cms": fmt_cfs(peak_cfs / CFS_PER_CMS), "action": action}
    return t(f"alert.headline{suffix}", lang, **fields), t(f"alert.body{suffix}", lang, **fields).strip()


def alert_text(river: str, level: str, peak_date: str, peak_cfs: float, lang: str) -> str:
    """Short plain-language alert, ready to paste into WhatsApp/SMS."""
    head, body = _headline_body(river, level, peak_date, peak_cfs, lang, actions(level, lang)[0])
    return f"{head}\n{body}\n{t('alert.source', lang)}"


def whatsapp_link(text: str) -> str:
    return "https://wa.me/?text=" + quote(text, safe="")


def cap_xml(station_id: str, river: str, lat: float, lon: float, level: str, peak_date: str,
            peak_cfs: float, lang: str, now: datetime | None = None) -> str:
    """Build a CAP 1.2 alert document (UTF-8 XML string)."""
    now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    ET.register_namespace("", CAP_NS)
    q = lambda tag: f"{{{CAP_NS}}}{tag}"  # noqa: E731
    root = ET.Element(q("alert"))
    for tag, value in (("identifier", f"floodcast-{station_id}-{uuid.uuid4().hex[:12]}"),
                       ("sender", "FLOODCAST-MidnightCowens"), ("sent", now.isoformat()),
                       ("status", "Exercise"), ("msgType", "Alert"), ("scope", "Public"),
                       ("note", "Research prototype output (FLOODCAST hackathon); not an official warning.")):
        ET.SubElement(root, q(tag)).text = value
    info = ET.SubElement(root, q("info"))
    headline, body = _headline_body(river, level, peak_date, peak_cfs, lang, "")
    fields = (("language", CAP_LANG.get(lang, "en-US")), ("category", "Met"), ("event", "River flood"),
              ("responseType", "Prepare" if level in ("elevated", "warning") else "Evacuate" if level in ("danger", "extreme") else "Monitor"),
              ("urgency", CAP_URGENCY.get(level, "Unknown")), ("severity", CAP_SEVERITY.get(level, "Unknown")),
              ("certainty", "Likely" if level in ("warning", "danger", "extreme") else "Possible"),
              ("effective", now.isoformat()), ("expires", (now + timedelta(hours=ALERT_VALID_HOURS)).isoformat()),
              ("senderName", "FLOODCAST — Team Midnight Cowens"),
              ("headline", headline), ("description", f"{body} {t(f'meaning.{level}', lang)}"),
              ("instruction", " ".join(actions(level, lang))))
    for tag, value in fields:
        ET.SubElement(info, q(tag)).text = value
    param = ET.SubElement(info, q("parameter"))
    ET.SubElement(param, q("valueName")).text = "peak_discharge_cfs"
    ET.SubElement(param, q("value")).text = f"{peak_cfs:.0f}"
    area = ET.SubElement(info, q("area"))
    ET.SubElement(area, q("areaDesc")).text = f"{river} (USGS {station_id})"
    ET.SubElement(area, q("circle")).text = f"{lat:.5f},{lon:.5f} 10"
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")
