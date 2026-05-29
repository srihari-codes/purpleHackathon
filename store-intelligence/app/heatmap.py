"""
heatmap.py — Zone heatmap computation.
Returns normalised 0-100 scores for visit frequency + avg dwell per zone.
Flags data_confidence=False if fewer than 20 sessions in the window.
"""

from datetime import datetime, timezone, timedelta
from typing import List
import logging

from sqlalchemy.orm import Session
from sqlalchemy import func, distinct

from app.database import EventRecord
from app.models import HeatmapResponse, HeatmapZone

logger = logging.getLogger(__name__)

MIN_SESSIONS_FOR_CONFIDENCE = 20


def get_heatmap(store_id: str, db: Session, hours: int = 24) -> HeatmapResponse:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=hours)

    # Total sessions in window (for confidence flag)
    total_sessions = (
        db.query(distinct(EventRecord.visitor_id))
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "ENTRY",
            EventRecord.is_staff == False,  # noqa: E712
            EventRecord.timestamp >= window_start,
        )
        .count()
    )
    high_confidence = total_sessions >= MIN_SESSIONS_FOR_CONFIDENCE

    # Zone visit frequency + avg dwell
    rows = (
        db.query(
            EventRecord.zone_id,
            func.count(EventRecord.event_id).label("freq"),
            func.avg(EventRecord.dwell_ms).label("avg_dwell"),
        )
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= window_start,
            EventRecord.is_staff == False,  # noqa: E712
            EventRecord.zone_id != None,  # noqa: E711
            EventRecord.event_type.in_(["ZONE_ENTER", "ZONE_DWELL"]),
        )
        .group_by(EventRecord.zone_id)
        .all()
    )

    if not rows:
        return HeatmapResponse(store_id=store_id, zones=[])

    # Normalise frequency 0-100
    max_freq = max(r.freq for r in rows) or 1
    zones: List[HeatmapZone] = [
        HeatmapZone(
            zone_id=row.zone_id,
            visit_frequency=row.freq,
            avg_dwell_ms=round(row.avg_dwell or 0, 1),
            normalised_score=round((row.freq / max_freq) * 100, 1),
            data_confidence=high_confidence,
        )
        for row in rows
    ]

    return HeatmapResponse(store_id=store_id, zones=zones)
