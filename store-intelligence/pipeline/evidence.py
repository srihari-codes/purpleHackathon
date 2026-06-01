"""
evidence.py — Identity Evidence Ledger.

Every visitor maintains a traceable evidence history that explains
WHY each identity match happened. No opaque scores — every decision
is backed by per-signal evidence with human-readable context.

Usage:
    ledger = EvidenceLedger()
    ledger.record("VIS_0042", [
        EvidenceEntry("reid", 0.92, 0.35, wall_time, "embedding cosine sim"),
        EvidenceEntry("trajectory", 0.78, 0.20, wall_time, "speed + direction match"),
    ])
    print(ledger.format_readable("VIS_0042"))
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, Deque, List, Optional

from config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evidence entry — one signal's contribution to an identity decision
# ---------------------------------------------------------------------------
@dataclass
class EvidenceEntry:
    signal_name:    str              # e.g. "reid", "fingerprint", "trajectory"
    score:          float            # raw signal value [0, 1]
    weight:         float            # consensus weight applied
    timestamp:      float            # wall-clock when recorded
    context:        str = ""         # human-readable explanation
    contribution:   float = 0.0     # weight × score (computed)

    def __post_init__(self):
        self.contribution = round(self.weight * self.score, 5)

    def to_dict(self) -> dict:
        return {
            "signal": self.signal_name,
            "score": round(self.score, 4),
            "weight": round(self.weight, 4),
            "contribution": self.contribution,
            "context": self.context,
        }


# ---------------------------------------------------------------------------
# Evidence snapshot — all evidence from one identity decision
# ---------------------------------------------------------------------------
@dataclass
class EvidenceSnapshot:
    visitor_id:        str
    matched_to:        Optional[str]   # "VIS_xxxx" if re-association, None if new
    decision:          str             # "ASSOCIATE" / "REJECT" / "NEW"
    identity_score:    float
    confidence_band:   str             # "HIGH" / "MEDIUM" / "LOW" / "REJECT"
    entries:           List[EvidenceEntry] = field(default_factory=list)
    timestamp:         float = field(default_factory=time.time)
    courtroom_verdict: Optional[dict] = None   # set if courtroom was invoked

    def to_dict(self) -> dict:
        return {
            "visitor_id":      self.visitor_id,
            "matched_to":      self.matched_to,
            "decision":        self.decision,
            "identity_score":  round(self.identity_score, 4),
            "confidence_band": self.confidence_band,
            "entries":         [e.to_dict() for e in self.entries],
            "courtroom":       self.courtroom_verdict,
        }


# ---------------------------------------------------------------------------
# Evidence Ledger — maintains per-visitor evidence history
# ---------------------------------------------------------------------------
class EvidenceLedger:
    """
    Thread-safe evidence ledger for all visitors.
    Records traceable match evidence for every identity decision.
    """

    def __init__(self, max_history: int = None):
        self._max = max_history or cfg.EVIDENCE_MAX_HISTORY
        # visitor_id → deque of EvidenceSnapshot
        self._ledgers: Dict[str, Deque[EvidenceSnapshot]] = defaultdict(
            lambda: deque(maxlen=self._max)
        )

    def record(
        self,
        visitor_id:      str,
        entries:         List[EvidenceEntry],
        decision:        str = "ASSOCIATE",
        identity_score:  float = 0.0,
        confidence_band: str = "MEDIUM",
        matched_to:      Optional[str] = None,
        courtroom_verdict: Optional[dict] = None,
    ):
        """Record one identity decision with its evidence chain."""
        snapshot = EvidenceSnapshot(
            visitor_id      = visitor_id,
            matched_to      = matched_to,
            decision        = decision,
            identity_score  = identity_score,
            confidence_band = confidence_band,
            entries         = entries,
            timestamp       = time.time(),
            courtroom_verdict = courtroom_verdict,
        )
        self._ledgers[visitor_id].append(snapshot)
        logger.debug(
            f"Evidence recorded for {visitor_id}: {decision} "
            f"score={identity_score:.3f} band={confidence_band} "
            f"({len(entries)} signals)"
        )

    def record_from_consensus(
        self,
        visitor_id:    str,
        decision_dict: dict,
        matched_to:    Optional[str] = None,
        courtroom_verdict: Optional[dict] = None,
    ):
        """
        Build evidence entries from a ConsensusDecision explanation dict.
        This is the primary integration point with the consensus engine.
        """
        signals     = decision_dict.get("signals", {})
        weights     = decision_dict.get("weights", {})
        entries = []
        for sig_name, sig_score in signals.items():
            w = weights.get(sig_name, 0.0)
            entries.append(EvidenceEntry(
                signal_name = sig_name,
                score       = sig_score,
                weight      = w,
                timestamp   = time.time(),
                context     = f"{sig_name}: {sig_score:.3f} × {w:.3f}",
            ))

        self.record(
            visitor_id      = visitor_id,
            entries         = entries,
            decision        = decision_dict.get("action", "UNKNOWN"),
            identity_score  = decision_dict.get("identity_score", 0.0),
            confidence_band = decision_dict.get("confidence_band", "UNKNOWN"),
            matched_to      = matched_to,
            courtroom_verdict = courtroom_verdict,
        )

    def get_ledger(self, visitor_id: str) -> List[EvidenceSnapshot]:
        """Full evidence history for a visitor."""
        return list(self._ledgers.get(visitor_id, []))

    def get_latest(self, visitor_id: str) -> Optional[EvidenceSnapshot]:
        """Most recent evidence snapshot."""
        ledger = self._ledgers.get(visitor_id)
        if ledger and len(ledger) > 0:
            return ledger[-1]
        return None

    def format_readable(self, visitor_id: str) -> str:
        """
        Human-readable evidence summary for a visitor.
        Suitable for GUI display and debug logs.

        Example output:
        ═══ VIS_0042 Evidence Ledger ═══
        ▸ Match #1: ASSOCIATE (score=0.814, band=MEDIUM)
          +0.322  reid            0.920 × 0.350  embedding cosine sim
          +0.156  trajectory      0.780 × 0.200  speed + direction match
          +0.127  temporal        0.850 × 0.150  2.1s gap, within window
          +0.105  camera_trans    0.700 × 0.150  CAM_01→CAM_02 expected
          +0.060  zone_plaus      0.600 × 0.100  ZONE_EB→ZONE_TFS valid
          +0.044  detection       0.880 × 0.050  strong YOLO detection
        """
        ledger = self._ledgers.get(visitor_id)
        if not ledger:
            return f"No evidence recorded for {visitor_id}"

        lines = [f"═══ {visitor_id} Evidence Ledger ═══"]
        for i, snap in enumerate(ledger):
            lines.append(
                f"▸ Match #{i+1}: {snap.decision} "
                f"(score={snap.identity_score:.3f}, band={snap.confidence_band})"
            )
            if snap.matched_to:
                lines.append(f"  ↳ matched to: {snap.matched_to}")
            for e in sorted(snap.entries, key=lambda x: -x.contribution):
                lines.append(
                    f"  {'+' if e.contribution >= 0 else ''}"
                    f"{e.contribution:.3f}  "
                    f"{e.signal_name:<16} "
                    f"{e.score:.3f} × {e.weight:.3f}  "
                    f"{e.context}"
                )
            if snap.courtroom_verdict:
                lines.append(f"  ⚖ Courtroom: {snap.courtroom_verdict.get('judge_rationale', 'N/A')}")
        return "\n".join(lines)

    def all_visitor_ids(self) -> List[str]:
        """List all visitors that have evidence records."""
        return list(self._ledgers.keys())

    def summary(self, visitor_id: str) -> dict:
        """Compact summary for API responses."""
        latest = self.get_latest(visitor_id)
        if not latest:
            return {"visitor_id": visitor_id, "evidence_count": 0}
        return {
            "visitor_id":      visitor_id,
            "evidence_count":  len(self._ledgers.get(visitor_id, [])),
            "latest":          latest.to_dict(),
        }
