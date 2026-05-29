"""
health.py — /health endpoint logic.
Returns service status, per-store last event timestamp,
and STALE_FEED warning if any store has >10 min lag.
"""

from datetime import datetime, timezone, timedelta
from typing import List
import time
import logging

from sqlalchemy.orm import Session
from sqlalchemy import func, distinct

from app.database import EventRecord
from app.models import HealthResponse, StoreFeedStatus

logger = logging.getLogger(__name__)

STALE_FEED_MINUTES = 10
_start_time = time.time()


def get_health(db: Session) -> HealthResponse:
    now = datetime.now(timezone.utc)

    # All stores that have ingested at least one event
    store_ids = [
        r[0] for r in db.query(distinct(EventRecord.store_id)).all()
    ]

    store_statuses: List[StoreFeedStatus] = []
    for store_id in store_ids:
        last_event = (
            db.query(func.max(EventRecord.timestamp))
            .filter(EventRecord.store_id == store_id)
            .scalar()
        )
        if last_event is None:
            status = "NO_DATA"
            lag_minutes = None
        else:
            if last_event.tzinfo is None:
                last_event = last_event.replace(tzinfo=timezone.utc)
            lag_minutes = round((now - last_event).total_seconds() / 60, 1)
            status = "STALE_FEED" if lag_minutes > STALE_FEED_MINUTES else "OK"

        store_statuses.append(StoreFeedStatus(
            store_id=store_id,
            last_event_timestamp=last_event,
            lag_minutes=lag_minutes,
            status=status,
        ))

    overall = (
        "degraded" if any(s.status == "STALE_FEED" for s in store_statuses)
        else "ok"
    )

    return HealthResponse(
        status=overall,
        version="1.0.0",
        uptime_seconds=round(time.time() - _start_time, 1),
        stores=store_statuses,
    )
