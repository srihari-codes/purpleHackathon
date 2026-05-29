"""
funnel.py — Conversion funnel computation.
Unit of analysis = session (unique visitor_id), not raw events.
Re-entries do NOT double-count a visitor in the funnel.
Stages: Entry → Zone Visit → Billing Queue → Purchase
"""

from datetime import datetime, timezone, timedelta
from typing import List
import logging

from sqlalchemy.orm import Session
from sqlalchemy import distinct

from app.database import EventRecord, POSTransaction
from app.models import FunnelResponse, FunnelStage

logger = logging.getLogger(__name__)

CONVERSION_WINDOW_MINUTES = 5


def get_funnel(store_id: str, db: Session, hours: int = 24) -> FunnelResponse:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=hours)

    # Stage 1: unique customer visitors (ENTRY events, excluding staff, deduplicated)
    entry_visitors: set = set(
        v[0] for v in db.query(distinct(EventRecord.visitor_id))
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.event_type == "ENTRY",
            EventRecord.is_staff == False,  # noqa: E712
            EventRecord.timestamp >= window_start,
        )
        .all()
    )
    total_entries = len(entry_visitors)

    # Stage 2: visitors who visited at least one named zone
    zone_visitors: set = set(
        v[0] for v in db.query(distinct(EventRecord.visitor_id))
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.event_type.in_(["ZONE_ENTER", "ZONE_DWELL"]),
            EventRecord.is_staff == False,  # noqa: E712
            EventRecord.timestamp >= window_start,
            EventRecord.zone_id.notin_(["ENTRY_THRESHOLD", "BILLING_COUNTER", "BILLING_QUEUE"]),
        )
        .all()
    ) & entry_visitors  # must have entered first
    total_zone_visits = len(zone_visitors)

    # Stage 3: visitors who reached billing queue
    billing_visitors: set = set(
        v[0] for v in db.query(distinct(EventRecord.visitor_id))
        .filter(
            EventRecord.store_id == store_id,
            EventRecord.zone_id.in_(["BILLING_COUNTER", "BILLING_QUEUE"]),
            EventRecord.is_staff == False,  # noqa: E712
            EventRecord.timestamp >= window_start,
        )
        .all()
    ) & entry_visitors
    total_billing = len(billing_visitors)

    # Stage 4: visitors who purchased (via POS correlation)
    transactions = (
        db.query(POSTransaction)
        .filter(
            POSTransaction.store_id == store_id,
            POSTransaction.timestamp >= window_start,
        )
        .all()
    )
    purchased_visitors: set = set()
    for txn in transactions:
        txn_time = txn.timestamp
        if txn_time.tzinfo is None:
            txn_time = txn_time.replace(tzinfo=timezone.utc)
        billing_start = txn_time - timedelta(minutes=CONVERSION_WINDOW_MINUTES)

        visitors_at_txn = set(
            v[0] for v in db.query(distinct(EventRecord.visitor_id))
            .filter(
                EventRecord.store_id == store_id,
                EventRecord.is_staff == False,  # noqa: E712
                EventRecord.zone_id.in_(["BILLING_COUNTER", "BILLING_QUEUE"]),
                EventRecord.timestamp >= billing_start,
                EventRecord.timestamp <= txn_time,
            )
            .all()
        )
        purchased_visitors.update(visitors_at_txn & entry_visitors)

    total_purchased = len(purchased_visitors)

    # ── Build stages with drop-off % ─────────────────────────────
    def drop_off(current: int, previous: int) -> float:
        if previous == 0:
            return 0.0
        return round((1 - current / previous) * 100, 2)

    stages: List[FunnelStage] = [
        FunnelStage(stage="Entry", count=total_entries, drop_off_pct=0.0),
        FunnelStage(stage="Zone Visit", count=total_zone_visits, drop_off_pct=drop_off(total_zone_visits, total_entries)),
        FunnelStage(stage="Billing Queue", count=total_billing, drop_off_pct=drop_off(total_billing, total_zone_visits)),
        FunnelStage(stage="Purchase", count=total_purchased, drop_off_pct=drop_off(total_purchased, total_billing)),
    ]

    return FunnelResponse(
        store_id=store_id,
        window_start=window_start,
        window_end=now,
        stages=stages,
    )
