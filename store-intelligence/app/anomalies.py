"""
anomalies.py — Real-time operational anomaly detection.
Anomaly types:
  - BILLING_QUEUE_SPIKE   : current queue depth > threshold
  - CONVERSION_DROP       : today's rate < 7-day rolling avg by margin
  - DEAD_ZONE             : no zone visits in last 30 min
  - STALE_FEED            : no events from any camera in >10 min (also in health)
"""

from datetime import datetime, timezone, timedelta
from typing import List
import logging

from sqlalchemy.orm import Session
from sqlalchemy import func, distinct

from app.database import EventRecord, POSTransaction
from app.models import AnomaliesResponse, Anomaly, AnomalySeverity

logger = logging.getLogger(__name__)

QUEUE_SPIKE_THRESHOLD = 5        # persons in queue
CONVERSION_DROP_MARGIN = 0.20    # 20% below 7-day avg → anomaly
DEAD_ZONE_MINUTES = 30


def _current_conversion_rate(store_id: str, db: Session, hours: int = 24) -> float:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=hours)
    from app.metrics import get_store_metrics
    metrics = get_store_metrics(store_id, db, hours=hours)
    return metrics.conversion_rate


def _seven_day_avg_conversion(store_id: str, db: Session) -> float:
    """Rough 7-day conversion avg using daily buckets."""
    rates = []
    for day_offset in range(1, 8):
        start = datetime.now(timezone.utc) - timedelta(days=day_offset + 1)
        end = datetime.now(timezone.utc) - timedelta(days=day_offset)

        visitors = (
            db.query(distinct(EventRecord.visitor_id))
            .filter(
                EventRecord.store_id == store_id,
                EventRecord.event_type == "ENTRY",
                EventRecord.is_staff == False,  # noqa: E712
                EventRecord.timestamp.between(start, end),
            )
            .count()
        )
        txn_count = (
            db.query(POSTransaction)
            .filter(
                POSTransaction.store_id == store_id,
                POSTransaction.timestamp.between(start, end),
            )
            .count()
        )
        if visitors > 0:
            rates.append(min(txn_count / visitors, 1.0))

    return sum(rates) / len(rates) if rates else 0.0


def get_anomalies(store_id: str, db: Session) -> AnomaliesResponse:
    now = datetime.now(timezone.utc)
    anomalies: List[Anomaly] = []

    # ── BILLING_QUEUE_SPIKE ──────────────────────────────────────
    latest_queue = (
        db.query(EventRecord)
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_JOIN",
        )
        .order_by(EventRecord.timestamp.desc())
        .first()
    )
    if latest_queue and latest_queue.metadata_json:
        qd = latest_queue.metadata_json.get("queue_depth", 0) or 0
        if qd >= QUEUE_SPIKE_THRESHOLD:
            severity = AnomalySeverity.CRITICAL if qd >= QUEUE_SPIKE_THRESHOLD * 2 else AnomalySeverity.WARN
            anomalies.append(Anomaly(
                anomaly_type="BILLING_QUEUE_SPIKE",
                severity=severity,
                description=f"Billing queue depth is {qd} (threshold: {QUEUE_SPIKE_THRESHOLD}).",
                suggested_action="Open additional billing counter or redirect customers.",
                detected_at=now,
                metadata={"queue_depth": qd},
            ))

    # ── CONVERSION_DROP ──────────────────────────────────────────
    today_rate = _current_conversion_rate(store_id, db)
    avg_rate = _seven_day_avg_conversion(store_id, db)
    if avg_rate > 0 and today_rate < avg_rate * (1 - CONVERSION_DROP_MARGIN):
        drop_pct = round((1 - today_rate / avg_rate) * 100, 1)
        anomalies.append(Anomaly(
            anomaly_type="CONVERSION_DROP",
            severity=AnomalySeverity.WARN,
            description=f"Conversion rate {today_rate:.1%} is {drop_pct}% below 7-day avg {avg_rate:.1%}.",
            suggested_action="Review floor staff deployment and product zone placement.",
            detected_at=now,
            metadata={"today_rate": today_rate, "seven_day_avg": avg_rate},
        ))

    # ── DEAD_ZONE ─────────────────────────────────────────────────
    dead_zone_cutoff = now - timedelta(minutes=DEAD_ZONE_MINUTES)
    all_zones = (
        db.query(distinct(EventRecord.zone_id))
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.zone_id != None,  # noqa: E711
            EventRecord.zone_id.notin_(["ENTRY_THRESHOLD", "BILLING_COUNTER", "BILLING_QUEUE"]),
            EventRecord.is_staff == False,  # noqa: E712
        )
        .all()
    )
    all_zone_ids = [z[0] for z in all_zones if z[0]]

    recently_active = (
        db.query(distinct(EventRecord.zone_id))
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= dead_zone_cutoff,
            EventRecord.zone_id != None,  # noqa: E711
            EventRecord.is_staff == False,  # noqa: E712
        )
        .all()
    )
    active_zone_ids = {z[0] for z in recently_active if z[0]}

    dead_zones = [z for z in all_zone_ids if z not in active_zone_ids]
    if dead_zones:
        anomalies.append(Anomaly(
            anomaly_type="DEAD_ZONE",
            severity=AnomalySeverity.INFO,
            description=f"No customer visits in {DEAD_ZONE_MINUTES} min: {', '.join(dead_zones)}.",
            suggested_action="Check camera feed for these zones; consider staff engagement.",
            detected_at=now,
            metadata={"dead_zones": dead_zones},
        ))

    return AnomaliesResponse(store_id=store_id, active_anomalies=anomalies)
