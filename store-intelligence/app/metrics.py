"""
metrics.py — Real-time store metrics computation.
All queries are live against the DB — no stale cache.
Staff events (is_staff=True) are excluded from all customer metrics.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional
import logging

from sqlalchemy.orm import Session
from sqlalchemy import func, distinct

from app.database import EventRecord, POSTransaction
from app.models import StoreMetrics, ZoneDwellMetric

logger = logging.getLogger(__name__)

# Conversion window: a visitor who was in billing zone within N minutes before a POS txn
CONVERSION_WINDOW_MINUTES = 5


def get_store_metrics(store_id: str, db: Session, hours: int = 24) -> StoreMetrics:
    """Compute today's metrics for a store over the last `hours` hours."""
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=hours)

    base_q = (
        db.query(EventRecord)
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= window_start,
            EventRecord.is_staff == False,  # noqa: E712
        )
    )

    # ── Unique visitors ──────────────────────────────────────────
    unique_visitors = (
        base_q.with_entities(distinct(EventRecord.visitor_id)).count()
    )

    # ── Conversion rate ──────────────────────────────────────────
    # Visitors who had a BILLING_COUNTER / BILLING_QUEUE zone event
    # in the 5-min window before any POS transaction
    transactions = (
        db.query(POSTransaction)
        .filter(
            POSTransaction.store_id == store_id,
            POSTransaction.timestamp >= window_start,
        )
        .all()
    )

    converted_visitors: set = set()
    for txn in transactions:
        txn_time = txn.timestamp
        if txn_time.tzinfo is None:
            txn_time = txn_time.replace(tzinfo=timezone.utc)
        billing_start = txn_time - timedelta(minutes=CONVERSION_WINDOW_MINUTES)

        billing_visitors = (
            db.query(EventRecord.visitor_id)
            .filter(
                EventRecord.store_id == store_id,
                EventRecord.is_staff == False,  # noqa: E712
                EventRecord.zone_id.in_(["BILLING_COUNTER", "BILLING_QUEUE"]),
                EventRecord.timestamp >= billing_start,
                EventRecord.timestamp <= txn_time,
            )
            .distinct()
            .all()
        )
        converted_visitors.update(v[0] for v in billing_visitors)

    conversion_rate = (
        round(len(converted_visitors) / unique_visitors, 4)
        if unique_visitors > 0
        else 0.0
    )

    # ── Avg dwell per zone ───────────────────────────────────────
    dwell_rows = (
        db.query(
            EventRecord.zone_id,
            func.avg(EventRecord.dwell_ms).label("avg_dwell"),
            func.count(EventRecord.event_id).label("cnt"),
        )
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.timestamp >= window_start,
            EventRecord.is_staff == False,  # noqa: E712
            EventRecord.zone_id != None,  # noqa: E711
            EventRecord.event_type.in_(["ZONE_DWELL", "ZONE_EXIT"]),
        )
        .group_by(EventRecord.zone_id)
        .all()
    )

    avg_dwell_per_zone: List[ZoneDwellMetric] = [
        ZoneDwellMetric(zone_id=row.zone_id, avg_dwell_ms=round(row.avg_dwell or 0, 1), visit_count=row.cnt)
        for row in dwell_rows
    ]

    # ── Queue depth (current) ────────────────────────────────────
    # Latest BILLING_QUEUE_JOIN metadata queue_depth value
    latest_queue = (
        db.query(EventRecord)
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "BILLING_QUEUE_JOIN",
        )
        .order_by(EventRecord.timestamp.desc())
        .first()
    )
    queue_depth = 0
    if latest_queue and latest_queue.metadata_json:
        queue_depth = latest_queue.metadata_json.get("queue_depth") or 0

    # ── Abandonment rate ─────────────────────────────────────────
    abandon_count = (
        base_q.filter(EventRecord.event_type == "BILLING_QUEUE_ABANDON").count()
    )
    queue_joins = (
        base_q.filter(EventRecord.event_type == "BILLING_QUEUE_JOIN").count()
    )
    abandonment_rate = (
        round(abandon_count / queue_joins, 4) if queue_joins > 0 else 0.0
    )

    return StoreMetrics(
        store_id=store_id,
        window_start=window_start,
        window_end=now,
        unique_visitors=unique_visitors,
        conversion_rate=conversion_rate,
        avg_dwell_per_zone=avg_dwell_per_zone,
        queue_depth=queue_depth,
        abandonment_rate=abandonment_rate,
    )
