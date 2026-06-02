"""
app/projections.py — Materialized read-model projections.

The API never reads raw events or sessions directly.
It reads these pre-computed projections instead.

Projections:
    MetricsProjection   — unique visitors, conversion rate, avg dwell/zone, queue stats
    FunnelProjection    — Entry → Zone → Billing → Purchase counts + drop-off %
    HeatmapProjection   — Zone visit frequency + avg dwell, normalised 0-100
    AnomalyProjection   — Active anomalies with severity and suggested actions
    HealthProjection    — Service status, last event per store, stale feed warnings

Each projection is rebuilt on demand from the SessionStore + VerifierEngine.
Projections are NOT cached between requests (real-time per spec).

data_confidence flag is set on Heatmap when session count < 20.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set

from .models import EventType, VisitorSession
from .sessionizer import SessionStore

logger = logging.getLogger(__name__)

STALE_FEED_THRESHOLD_SEC = 600   # 10 minutes
DATA_CONFIDENCE_MIN_SESSIONS = 20


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_dt(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# MetricsProjection
# ---------------------------------------------------------------------------

class MetricsProjection:
    """
    GET /stores/{id}/metrics

    unique_visitors    — deduplicated by visitor_id (reentry ≠ new visitor)
    conversion_rate    — converted sessions / unique customer sessions
    avg_dwell_per_zone — zone_id → average dwell ms across all sessions
    queue_depth        — max queue_depth seen in active/recent sessions
    abandonment_rate   — sessions with ABANDON but no JOIN / total join sessions
    as_of              — projection timestamp
    """

    @staticmethod
    def build(
        sessions: List[VisitorSession],
        store_id: str,
        correlation_engine=None,
    ) -> Dict[str, Any]:
        customer_sessions = [s for s in sessions if not s.is_staff]

        # Unique visitors (visitor_id dedup — reentry must not double count)
        visitor_converted: Dict[str, bool] = {}
        for s in customer_sessions:
            is_conv = (
                (correlation_engine and correlation_engine.is_converted(s.session_id))
                or s.purchase_candidate
            )
            if is_conv:
                visitor_converted[s.visitor_id] = True
            else:
                visitor_converted.setdefault(s.visitor_id, False)

        unique_visitors = len(visitor_converted)
        converted = sum(1 for v in visitor_converted.values() if v)
        conversion_rate = round(converted / unique_visitors, 4) if unique_visitors else 0.0

        # Avg dwell per zone
        zone_total:  Dict[str, int] = {}
        zone_count:  Dict[str, int] = {}
        for s in customer_sessions:
            for zone_id, dwell_ms in s.dwell_per_zone.items():
                zone_total[zone_id] = zone_total.get(zone_id, 0) + dwell_ms
                zone_count[zone_id] = zone_count.get(zone_id, 0) + 1
        avg_dwell_per_zone = {
            z: round(zone_total[z] / zone_count[z])
            for z in zone_total
        }

        # Queue stats
        join_sessions   = sum(
            1 for s in customer_sessions
            if any(qe.event_type == EventType.BILLING_QUEUE_JOIN for qe in s.queue_events)
        )
        abandon_sessions = sum(
            1 for s in customer_sessions
            if any(qe.event_type == EventType.BILLING_QUEUE_ABANDON for qe in s.queue_events)
        )
        abandonment_rate = round(abandon_sessions / join_sessions, 4) if join_sessions else 0.0

        max_queue_depth = 0
        for s in customer_sessions:
            for qe in s.queue_events:
                if qe.queue_depth and qe.queue_depth > max_queue_depth:
                    max_queue_depth = qe.queue_depth

        return {
            "store_id":          store_id,
            "unique_visitors":   unique_visitors,
            "conversion_rate":   conversion_rate,
            "converted_count":   converted,
            "avg_dwell_per_zone": avg_dwell_per_zone,
            "current_queue_depth": max_queue_depth,
            "abandonment_rate":  abandonment_rate,
            "total_sessions":    len(customer_sessions),
            "active_sessions":   sum(1 for s in customer_sessions if s.is_active),
            "as_of":             _iso(_now_utc()),
        }


# ---------------------------------------------------------------------------
# FunnelProjection
# ---------------------------------------------------------------------------

class FunnelProjection:
    """
    GET /stores/{id}/funnel

    Session is the unit — not raw events.
    Re-entries must not double-count a visitor.

    Funnel stages:
        1. ENTRY      — total unique customer visitors
        2. ZONE_VISIT — visited at least one zone
        3. BILLING    — joined billing queue (purchase_candidate)
        4. PURCHASE   — correlated with a POS transaction
    """

    @staticmethod
    def build(
        sessions: List[VisitorSession],
        store_id: str,
        correlation_engine=None,
    ) -> Dict[str, Any]:
        customer_sessions = [s for s in sessions if not s.is_staff]

        # Track stages per visitor across all their sessions to handle re-entry correctly
        visitor_stages: Dict[str, Dict[str, bool]] = {}
        for s in customer_sessions:
            v_id = s.visitor_id
            if v_id not in visitor_stages:
                visitor_stages[v_id] = {
                    "entry": True,
                    "zone": False,
                    "billing": False,
                    "purchase": False,
                }
            if s.zones_visited:
                visitor_stages[v_id]["zone"] = True
            if s.purchase_candidate:
                visitor_stages[v_id]["billing"] = True
            is_conv = (
                (correlation_engine and correlation_engine.is_converted(s.session_id))
                or s.purchase_candidate
            )
            if is_conv:
                visitor_stages[v_id]["purchase"] = True
                visitor_stages[v_id]["billing"] = True  # purchase implies they were at billing

        stage_entry = len(visitor_stages)
        stage_zone = sum(1 for v in visitor_stages.values() if v["zone"])
        stage_billing = sum(1 for v in visitor_stages.values() if v["billing"])
        stage_purchase = sum(1 for v in visitor_stages.values() if v["purchase"])

        def pct(num: int, denom: int) -> float:
            return round(100.0 * num / denom, 1) if denom else 0.0

        def drop(stage_n: int, stage_prev: int) -> float:
            if stage_prev == 0:
                return 0.0
            return round(100.0 * (stage_prev - stage_n) / stage_prev, 1)

        return {
            "store_id": store_id,
            "funnel": [
                {
                    "stage":      "ENTRY",
                    "count":      stage_entry,
                    "pct_of_top": 100.0,
                    "drop_off_pct": 0.0,
                },
                {
                    "stage":      "ZONE_VISIT",
                    "count":      stage_zone,
                    "pct_of_top": pct(stage_zone, stage_entry),
                    "drop_off_pct": drop(stage_zone, stage_entry),
                },
                {
                    "stage":      "BILLING_QUEUE",
                    "count":      stage_billing,
                    "pct_of_top": pct(stage_billing, stage_entry),
                    "drop_off_pct": drop(stage_billing, stage_zone),
                },
                {
                    "stage":      "PURCHASE",
                    "count":      stage_purchase,
                    "pct_of_top": pct(stage_purchase, stage_entry),
                    "drop_off_pct": drop(stage_purchase, stage_billing),
                },
            ],
            "as_of": _iso(_now_utc()),
        }


# ---------------------------------------------------------------------------
# HeatmapProjection
# ---------------------------------------------------------------------------

class HeatmapProjection:
    """
    GET /stores/{id}/heatmap

    Zone visit frequency + avg dwell, normalised 0-100.
    data_confidence=False if fewer than 20 sessions.
    """

    @staticmethod
    def build(
        sessions: List[VisitorSession],
        store_id: str,
    ) -> Dict[str, Any]:
        customer_sessions = [s for s in sessions if not s.is_staff]
        session_count = len(customer_sessions)
        data_confidence = session_count >= DATA_CONFIDENCE_MIN_SESSIONS

        zone_visits: Dict[str, int] = {}
        zone_dwell_total: Dict[str, int] = {}
        zone_dwell_count: Dict[str, int] = {}

        for s in customer_sessions:
            for zone_id in s.zones_visited:
                zone_visits[zone_id] = zone_visits.get(zone_id, 0) + 1
            for zone_id, dwell_ms in s.dwell_per_zone.items():
                zone_dwell_total[zone_id] = zone_dwell_total.get(zone_id, 0) + dwell_ms
                zone_dwell_count[zone_id] = zone_dwell_count.get(zone_id, 0) + 1

        all_zones = set(zone_visits) | set(zone_dwell_total)
        if not all_zones:
            return {
                "store_id": store_id,
                "zones": [],
                "data_confidence": data_confidence,
                "session_count": session_count,
                "as_of": _iso(_now_utc()),
            }

        max_visits = max(zone_visits.values(), default=1)
        max_dwell  = max(zone_dwell_total.values(), default=1)

        zones = []
        for zone_id in sorted(all_zones):
            visits     = zone_visits.get(zone_id, 0)
            total_ms   = zone_dwell_total.get(zone_id, 0)
            count      = zone_dwell_count.get(zone_id, 1)
            avg_dwell  = round(total_ms / count) if count else 0
            norm_freq  = round(100.0 * visits / max_visits) if max_visits else 0
            norm_dwell = round(100.0 * total_ms / max_dwell) if max_dwell else 0

            zones.append({
                "zone_id":          zone_id,
                "visit_count":      visits,
                "avg_dwell_ms":     avg_dwell,
                "freq_score":       norm_freq,    # 0-100 normalised visit frequency
                "dwell_score":      norm_dwell,   # 0-100 normalised total dwell
                "combined_score":   round((norm_freq + norm_dwell) / 2),
            })

        # Sort by combined_score descending (hottest zone first)
        zones.sort(key=lambda z: z["combined_score"], reverse=True)

        return {
            "store_id":        store_id,
            "zones":           zones,
            "data_confidence": data_confidence,
            "session_count":   session_count,
            "as_of":           _iso(_now_utc()),
        }


# ---------------------------------------------------------------------------
# AnomalyProjection
# ---------------------------------------------------------------------------

class AnomalyProjection:
    """
    GET /stores/{id}/anomalies

    Active anomalies from the VerifierEngine + session-derived checks:
        BILLING_QUEUE_SPIKE  — queue depth > threshold
        CONVERSION_DROP      — current rate vs 7-day avg
        DEAD_ZONE            — no zone visits in >30 min
    """

    QUEUE_SPIKE_THRESHOLD = 5
    DEAD_ZONE_WINDOW_MIN  = 30

    @staticmethod
    def build(
        sessions: List[VisitorSession],
        store_id: str,
        verifier_warnings=None,
        conversion_rate: float = 0.0,
        historical_avg_rate: float = 0.0,
    ) -> Dict[str, Any]:
        anomalies = []
        now = _now_utc()

        # 1. Verifier warnings
        if verifier_warnings:
            for w in verifier_warnings:
                anomalies.append({
                    "type":             w.code,
                    "severity":         w.severity,
                    "message":          w.message,
                    "visitor_ids":      w.visitor_ids,
                    "timestamp":        w.timestamp,
                    "suggested_action": w.suggested_action,
                    "metadata":         w.metadata,
                })

        # 2. Queue depth spike from active sessions
        max_depth = 0
        for s in sessions:
            for qe in s.queue_events:
                if qe.queue_depth and qe.queue_depth > max_depth:
                    max_depth = qe.queue_depth
        if max_depth >= AnomalyProjection.QUEUE_SPIKE_THRESHOLD:
            anomalies.append({
                "type":     "BILLING_QUEUE_SPIKE",
                "severity": "CRITICAL" if max_depth >= 8 else "WARN",
                "message":  f"Billing queue depth reached {max_depth} — customers may abandon",
                "visitor_ids": [],
                "timestamp": _iso(now),
                "suggested_action": "Open additional billing counter or call for staff support.",
                "metadata": {"max_queue_depth": max_depth},
            })

        # 3. Conversion drop vs historical average
        if historical_avg_rate > 0 and conversion_rate < historical_avg_rate * 0.7:
            drop_pct = round(100 * (historical_avg_rate - conversion_rate) / historical_avg_rate, 1)
            anomalies.append({
                "type":     "CONVERSION_DROP",
                "severity": "WARN",
                "message":  (
                    f"Conversion rate {conversion_rate:.1%} is {drop_pct}% below "
                    f"7-day avg {historical_avg_rate:.1%}"
                ),
                "visitor_ids": [],
                "timestamp":  _iso(now),
                "suggested_action": "Check for product availability, staff engagement, or queue issues.",
                "metadata": {
                    "current_rate": conversion_rate,
                    "historical_avg": historical_avg_rate,
                    "drop_pct": drop_pct,
                },
            })

        # 4. Dead zone detection (no visits in DEAD_ZONE_WINDOW_MIN)
        zone_last_seen: Dict[str, datetime] = {}
        for s in sessions:
            for zone_id in s.zones_visited:
                ref = _parse_dt(s.end_time or s.start_time)
                if zone_id not in zone_last_seen or ref > zone_last_seen[zone_id]:
                    zone_last_seen[zone_id] = ref

        dead_window = timedelta(minutes=AnomalyProjection.DEAD_ZONE_WINDOW_MIN)
        for zone_id, last_dt in zone_last_seen.items():
            if (now - last_dt) > dead_window:
                anomalies.append({
                    "type":     "DEAD_ZONE",
                    "severity": "INFO",
                    "message":  f"Zone {zone_id} has had no visits in >{AnomalyProjection.DEAD_ZONE_WINDOW_MIN} min",
                    "visitor_ids": [],
                    "timestamp":  _iso(now),
                    "suggested_action": f"Check zone {zone_id} signage and product placement.",
                    "metadata": {
                        "zone_id": zone_id,
                        "last_seen": _iso(last_dt),
                        "minutes_idle": round((now - last_dt).total_seconds() / 60, 1),
                    },
                })

        return {
            "store_id":      store_id,
            "anomaly_count": len(anomalies),
            "anomalies":     anomalies,
            "as_of":         _iso(now),
        }


# ---------------------------------------------------------------------------
# HealthProjection
# ---------------------------------------------------------------------------

class HealthProjection:
    """
    GET /health

    Service status, last event timestamp per store, STALE_FEED warning.
    """

    @staticmethod
    def build(
        event_store,
        session_store: SessionStore,
        started_at: str,
    ) -> Dict[str, Any]:
        now = _now_utc()
        store_ids = set(event_store.get_all_store_ids()) | set(session_store.get_all_store_ids())

        stores_health = []
        overall_stale = False

        for store_id in sorted(store_ids):
            last_ts = event_store.last_event_timestamp(store_id)
            session_count = len(session_store.get_all_sessions(store_id))
            active_count  = session_store.get_active_count(store_id)

            stale = False
            lag_sec = None
            if last_ts:
                lag_sec = (now - _parse_dt(last_ts)).total_seconds()
                stale   = lag_sec > STALE_FEED_THRESHOLD_SEC
                if stale:
                    overall_stale = True
            else:
                stale = True
                overall_stale = True

            stores_health.append({
                "store_id":       store_id,
                "last_event_at":  last_ts,
                "lag_sec":        round(lag_sec, 1) if lag_sec is not None else None,
                "stale_feed":     stale,
                "session_count":  session_count,
                "active_sessions": active_count,
                "status":         "STALE_FEED" if stale else "OK",
            })

        return {
            "status":       "DEGRADED" if overall_stale else "OK",
            "started_at":   started_at,
            "checked_at":   _iso(now),
            "stores":       stores_health,
            "store_count":  len(stores_health),
        }
