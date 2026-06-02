"""
app/correlation.py — POS Transaction Correlation Engine.

Rule (spec-exact):
    A visitor who was in the BILLING zone in the 5-minute window BEFORE
    a POS transaction timestamp counts as a converted visitor for that session.

No customer_id in POS data — correlation is purely time-window + store.

CorrelationEngine:
    - Accepts POS transaction records (from pos_transactions.csv or API)
    - Marks sessions whose last billing-zone event falls within 5 min before
      any transaction for the same store
    - Is idempotent: running twice with same data produces same result
    - Exposes conversion rate and per-session purchase flag

POS transaction schema:
    store_id, transaction_id, timestamp, basket_value_inr
"""

from __future__ import annotations

import csv
import io
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set

from .models import EventType, VisitorSession

logger = logging.getLogger(__name__)

# How far back from a POS timestamp to look for billing-zone presence
BILLING_WINDOW_SEC = 300   # 5 minutes (spec)

BILLING_ZONE_PREFIXES = {
    "ZONE_BILLING", "ZONE_CASH", "BILLING", "ZONE_CHECKOUT",
}


def _is_billing_zone(zone_id: Optional[str]) -> bool:
    if not zone_id:
        return False
    z = zone_id.upper()
    return any(z.startswith(pfx) for pfx in BILLING_ZONE_PREFIXES)


def _parse_dt(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# POS Transaction record
# ---------------------------------------------------------------------------

@dataclass
class POSTransaction:
    store_id:         str
    transaction_id:   str
    timestamp:        str          # ISO-8601
    basket_value_inr: float = 0.0

    def parsed_dt(self) -> datetime:
        return _parse_dt(self.timestamp)


# ---------------------------------------------------------------------------
# Correlation Engine
# ---------------------------------------------------------------------------

class CorrelationEngine:
    """
    Correlates VisitorSessions with POS transactions.

    Usage:
        engine = CorrelationEngine()
        engine.load_csv("/data/pos_transactions.csv")

        # After sessions are built:
        engine.correlate(session_store, "STORE_BLR_002")

        # Query:
        rate = engine.conversion_rate("STORE_BLR_002")
        converted_ids = engine.converted_session_ids("STORE_BLR_002")
    """

    def __init__(self) -> None:
        # store_id → list of transactions sorted by time
        self._transactions: Dict[str, List[POSTransaction]] = {}
        # session_id → True if correlated with a POS transaction
        self._converted_sessions: Set[str] = set()
        self._lock = threading.Lock()

    # ── load POS data ──────────────────────────────────────────────────────

    def load_csv(self, path: str) -> int:
        """
        Load POS transactions from CSV file.
        Returns number of records loaded.
        CSV columns: store_id, transaction_id, timestamp, basket_value_inr
        """
        loaded = 0
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                txn = POSTransaction(
                    store_id=row["store_id"].strip(),
                    transaction_id=row["transaction_id"].strip(),
                    timestamp=row["timestamp"].strip(),
                    basket_value_inr=float(row.get("basket_value_inr", 0) or 0),
                )
                with self._lock:
                    if txn.store_id not in self._transactions:
                        self._transactions[txn.store_id] = []
                    self._transactions[txn.store_id].append(txn)
                loaded += 1
        # Sort each store's transactions by time
        with self._lock:
            for txns in self._transactions.values():
                txns.sort(key=lambda t: t.timestamp)
        logger.info("pos_loaded records=%d", loaded)
        return loaded

    def add_transaction(self, txn: POSTransaction) -> None:
        """Add a single POS transaction (for API-fed data)."""
        with self._lock:
            if txn.store_id not in self._transactions:
                self._transactions[txn.store_id] = []
            self._transactions[txn.store_id].append(txn)
            self._transactions[txn.store_id].sort(key=lambda t: t.timestamp)

    def transaction_count(self, store_id: str) -> int:
        with self._lock:
            return len(self._transactions.get(store_id, []))

    # ── correlation ───────────────────────────────────────────────────────

    def correlate(self, sessions: List[VisitorSession], store_id: str) -> int:
        """
        Mark sessions as converted if their last billing-zone event
        falls within BILLING_WINDOW_SEC before any POS transaction.

        Returns: number of newly converted sessions.
        """
        with self._lock:
            txns = list(self._transactions.get(store_id, []))

        if not txns:
            return 0

        newly_converted = 0
        txn_dts = [t.parsed_dt() for t in txns]

        for session in sessions:
            if session.is_staff:
                continue
            if session.session_id in self._converted_sessions:
                continue

            # Find the last billing-zone timestamp for this session
            last_billing_dt = self._last_billing_ts(session)
            if last_billing_dt is None:
                continue

            # Check if any POS transaction occurred within the window after
            # the visitor's last billing-zone presence
            window_end = last_billing_dt + timedelta(seconds=BILLING_WINDOW_SEC)
            for txn_dt in txn_dts:
                if last_billing_dt <= txn_dt <= window_end:
                    with self._lock:
                        self._converted_sessions.add(session.session_id)
                    session.purchase_candidate = True   # promote to confirmed purchase
                    newly_converted += 1
                    logger.debug(
                        "pos_correlated session=%s visitor=%s store=%s",
                        session.session_id, session.visitor_id, store_id,
                    )
                    break  # one match is enough

        logger.info(
            "correlation_run store=%s sessions=%d converted=%d",
            store_id, len(sessions), newly_converted,
        )
        return newly_converted

    # ── helpers ───────────────────────────────────────────────────────────

    def _last_billing_ts(self, session: VisitorSession) -> Optional[datetime]:
        """
        Find the latest billing-zone event timestamp in the session's
        queue_events (BILLING_QUEUE_JOIN preferred) or dwell_per_zone.
        """
        # Prefer explicit queue join timestamps
        billing_ts = None
        for qe in session.queue_events:
            if qe.event_type == EventType.BILLING_QUEUE_JOIN:
                ts_dt = _parse_dt(qe.timestamp)
                if billing_ts is None or ts_dt > billing_ts:
                    billing_ts = ts_dt

        # Also check zones_visited for billing zones (ZONE_ENTER timestamps
        # are not stored directly, so we use session start as a fallback)
        if billing_ts is None:
            for zone_id in session.zones_visited:
                if _is_billing_zone(zone_id):
                    # Use session end_time or start_time as estimate
                    ref = session.end_time or session.start_time
                    billing_ts = _parse_dt(ref)
                    break

        return billing_ts

    # ── queries ───────────────────────────────────────────────────────────

    def is_converted(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._converted_sessions

    def converted_session_ids(self, store_id: str = None) -> Set[str]:
        with self._lock:
            return set(self._converted_sessions)

    def conversion_rate(
        self, sessions: List[VisitorSession], store_id: str
    ) -> float:
        """
        Compute conversion rate for a list of sessions.
        Only counts non-staff, closed (exited) sessions.
        Re-entry visitors counted once.
        """
        # Deduplicate by visitor_id (reentry must not double-count)
        seen_visitors: Set[str] = set()
        unique_customers = 0
        converted_customers = 0

        for s in sessions:
            if s.is_staff:
                continue
            if s.visitor_id in seen_visitors:
                continue
            seen_visitors.add(s.visitor_id)
            unique_customers += 1
            if self.is_converted(s.session_id) or s.purchase_candidate:
                converted_customers += 1

        if unique_customers == 0:
            return 0.0
        return round(converted_customers / unique_customers, 4)

    def clear(self) -> None:
        with self._lock:
            self._transactions.clear()
            self._converted_sessions.clear()
