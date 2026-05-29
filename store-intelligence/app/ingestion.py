"""
ingestion.py — Ingest, validate, deduplicate events.
Idempotent: re-ingesting the same event_id is a no-op.
Supports partial success: malformed events are rejected but valid ones proceed.
"""

from datetime import datetime, timezone
from typing import List
import logging

from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.models import StoreEvent, IngestResult
from app.database import EventRecord

logger = logging.getLogger(__name__)


def ingest_events(events: List[StoreEvent], db: Session) -> IngestResult:
    """
    Validate and persist a batch of events.
    Returns counts of accepted / rejected / duplicate.
    """
    accepted = 0
    rejected = 0
    duplicate = 0
    errors = []

    for evt in events:
        try:
            # Check for existing event_id (dedup)
            existing = db.query(EventRecord).filter(
                EventRecord.event_id == evt.event_id
            ).first()

            if existing:
                duplicate += 1
                continue

            record = EventRecord(
                event_id=evt.event_id,
                store_id=evt.store_id,
                camera_id=evt.camera_id,
                visitor_id=evt.visitor_id,
                event_type=evt.event_type if isinstance(evt.event_type, str) else evt.event_type.value,
                timestamp=evt.timestamp,
                zone_id=evt.zone_id,
                dwell_ms=evt.dwell_ms,
                is_staff=evt.is_staff,
                confidence=evt.confidence,
                metadata_json=evt.metadata.model_dump() if evt.metadata else None,
                ingested_at=datetime.now(timezone.utc),
            )
            db.add(record)
            accepted += 1

        except Exception as exc:
            logger.warning("Failed to ingest event %s: %s", getattr(evt, "event_id", "?"), exc)
            errors.append({"event_id": getattr(evt, "event_id", "?"), "error": str(exc)})
            rejected += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("DB commit failed during ingest: %s", exc)
        raise

    logger.info(
        "Ingest complete accepted=%d rejected=%d duplicate=%d",
        accepted, rejected, duplicate,
    )
    return IngestResult(
        accepted=accepted,
        rejected=rejected,
        duplicate=duplicate,
        errors=errors,
    )
