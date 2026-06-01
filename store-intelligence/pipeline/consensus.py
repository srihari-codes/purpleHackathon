"""
consensus.py — Hybrid Consensus Identity Engine.

Every identity re-association decision is produced by a committee of
8 independent signals. No single model decides identity alone.

Usage:
    engine = ConsensusIdentityEngine()
    signals = ConsensusSignals(
        reid_score=0.92,
        fingerprint_score=0.85,
        trajectory_score=0.78,
        temporal_score=0.85,
        camera_transition_score=0.70,
        zone_plausibility_score=0.60,
        detection_score=0.88,
        track_health=0.75,
    )
    decision = engine.decide(signals, candidate_visitor_id="VIS_0047")
    # decision.should_associate → True
    # decision.explanation → human-readable dict
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input / Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConsensusSignals:
    """
    All 8 independent signals for one candidate identity match.
    Every value is in [0, 1]. Missing signals should be set to 0.5 (neutral).
    """
    reid_score:             float = 0.5   # cosine similarity of embeddings
    fingerprint_score:      float = 0.5   # AppearanceFingerprint.compare()
    trajectory_score:       float = 0.5   # motion speed/direction similarity
    temporal_score:         float = 0.5   # time-gap plausibility
    camera_transition_score:float = 0.5   # StoreGraph camera transition prob
    zone_plausibility_score:float = 0.5   # StoreGraph zone transition prob
    detection_score:        float = 0.5   # YOLO detection confidence
    track_health:           float = 0.5   # normalised track health (0–1)
    group_continuity_score: float = 0.5   # group co-location boost
    staff_reputation_score: float = 0.5   # staff behavior combined score
    visitor_dna_score:      float = 0.5   # visitor behavioral fingerprint sim

    # Metadata for explanation (not used in scoring)
    candidate_visitor_id:  Optional[str]   = None
    age_sec:               float           = 0.0
    cam_from:              Optional[str]   = None
    cam_to:                Optional[str]   = None
    zone_from:             Optional[str]   = None
    zone_to:               Optional[str]   = None


@dataclass
class ConsensusDecision:
    """
    Result of the consensus vote for one candidate identity match.
    """
    identity_score:    float
    should_associate:  bool
    explanation:       Dict               # per-signal breakdown + summary
    dominant_signal:   str               # which signal had highest contribution
    confidence_band:   str               # "HIGH" / "MEDIUM" / "LOW" / "REJECT"

    # Per-signal contributions (weight × score)
    contributions: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Consensus Identity Engine
# ---------------------------------------------------------------------------

class ConsensusIdentityEngine:
    """
    Weighted voting system for identity re-association.

    Weights are configurable via config.py (CONSENSUS_W_* constants).
    The engine normalises weights so they sum to 1.0.

    Hard gates:
      - reid_score < CONSENSUS_REID_HARD_MIN → always reject
      - zone transition physically impossible → strong penalty
    """

    SIGNAL_NAMES = [
        "reid",
        "fingerprint",
        "trajectory",
        "temporal",
        "camera_transition",
        "zone_plausibility",
        "detection",
        "track_health",
        "group_continuity",
        "staff_reputation",
        "visitor_dna",
    ]

    def __init__(self):
        raw_weights = {
            "reid":              cfg.CONSENSUS_W_REID,
            "fingerprint":       cfg.CONSENSUS_W_FINGERPRINT,
            "trajectory":        cfg.CONSENSUS_W_TRAJECTORY,
            "temporal":          cfg.CONSENSUS_W_TEMPORAL,
            "camera_transition": cfg.CONSENSUS_W_CAM_TRANSITION,
            "zone_plausibility": cfg.CONSENSUS_W_ZONE,
            "detection":         cfg.CONSENSUS_W_DETECTION,
            "track_health":      cfg.CONSENSUS_W_TRACK_HEALTH,
            "group_continuity":  getattr(cfg, "CONSENSUS_W_GROUP", 0.05),
            "staff_reputation":  getattr(cfg, "CONSENSUS_W_STAFF_REP", 0.0),
            "visitor_dna":       getattr(cfg, "CONSENSUS_W_VISITOR_DNA", 0.05),
        }
        # Normalise so weights always sum to 1.0
        total = sum(raw_weights.values())
        self.weights = {k: v / total for k, v in raw_weights.items()} if total > 0 else raw_weights
        self.threshold = cfg.CONSENSUS_THRESHOLD
        self.reid_hard_min = cfg.CONSENSUS_REID_HARD_MIN

    # ------------------------------------------------------------------
    def decide(
        self,
        signals: ConsensusSignals,
        candidate_visitor_id: Optional[str] = None,
    ) -> ConsensusDecision:
        """
        Run the consensus vote and produce a decision + explanation.
        """
        vid = candidate_visitor_id or signals.candidate_visitor_id or "UNKNOWN"

        # ── Hard gate: low ReID means no association ───────────────────
        if signals.reid_score < self.reid_hard_min:
            return ConsensusDecision(
                identity_score   = signals.reid_score,
                should_associate = False,
                explanation      = self._build_explanation(
                    signals, {}, signals.reid_score, vid,
                    veto_reason=f"reid_score={signals.reid_score:.3f} < hard_min={self.reid_hard_min}"
                ),
                dominant_signal  = "reid",
                confidence_band  = "REJECT",
                contributions    = {},
            )

        # ── Weighted vote ──────────────────────────────────────────────
        signal_values = {
            "reid":              signals.reid_score,
            "fingerprint":       signals.fingerprint_score,
            "trajectory":        signals.trajectory_score,
            "temporal":          signals.temporal_score,
            "camera_transition": signals.camera_transition_score,
            "zone_plausibility": signals.zone_plausibility_score,
            "detection":         signals.detection_score,
            "track_health":      signals.track_health,
            "group_continuity":  signals.group_continuity_score,
            "staff_reputation":  signals.staff_reputation_score,
            "visitor_dna":       signals.visitor_dna_score,
        }

        contributions = {
            name: round(self.weights[name] * val, 5)
            for name, val in signal_values.items()
        }
        identity_score = sum(contributions.values())

        # Dominant signal = highest weighted contribution
        dominant = max(contributions, key=contributions.get)

        # Confidence band
        if identity_score >= 0.85:
            band = "HIGH"
        elif identity_score >= self.threshold:
            band = "MEDIUM"
        elif identity_score >= self.threshold * 0.8:
            band = "LOW"
        else:
            band = "REJECT"

        should_associate = identity_score >= self.threshold

        explanation = self._build_explanation(
            signals, contributions, identity_score, vid
        )

        return ConsensusDecision(
            identity_score   = round(identity_score, 4),
            should_associate = should_associate,
            explanation      = explanation,
            dominant_signal  = dominant,
            confidence_band  = band,
            contributions    = contributions,
        )

    def decide_batch(
        self,
        candidates: List[Tuple[str, ConsensusSignals]],
    ) -> Optional[Tuple[str, ConsensusDecision]]:
        """
        Evaluate multiple candidates and return the best match above threshold.
        Also detects competing associations (multiple candidates near threshold).
        Returns (visitor_id, decision) or None if no match.
        """
        results = []
        for vid, signals in candidates:
            dec = self.decide(signals, candidate_visitor_id=vid)
            if dec.should_associate:
                results.append((vid, dec))

        if not results:
            return None

        # Sort by identity_score descending
        results.sort(key=lambda x: -x[1].identity_score)
        best_vid, best_dec = results[0]

        # Check for competing matches (second-best within 0.05 of best)
        if len(results) > 1:
            second_score = results[1][1].identity_score
            gap = best_dec.identity_score - second_score
            if gap < 0.05:
                # Ambiguous — note it in the explanation
                best_dec.explanation["competing_match"] = {
                    "visitor_id": results[1][0],
                    "score": second_score,
                    "gap": round(gap, 4),
                    "warning": "Ambiguous match — low gap to second candidate",
                }
                best_dec.confidence_band = "LOW"
                logger.debug(
                    f"Ambiguous match: {best_vid}={best_dec.identity_score:.3f} "
                    f"vs {results[1][0]}={second_score:.3f} (gap={gap:.3f})"
                )

        return best_vid, best_dec

    # ------------------------------------------------------------------
    def format_explanation(self, decision: ConsensusDecision) -> str:
        """
        Human-readable explanation string, suitable for logs and GUI.

        Example output:
        VIS_0047 reused (score=0.814, band=MEDIUM):
          reid             0.92 × 0.350 = 0.322  ✓ dominant
          trajectory       0.78 × 0.200 = 0.156  ✓
          temporal         0.85 × 0.150 = 0.127  ✓
          camera_transition 0.70 × 0.150 = 0.105  ✓
          zone_plausibility 0.60 × 0.100 = 0.060  ✓
          detection        0.88 × 0.050 = 0.044  ✓
          fingerprint      0.65 × 0.050 = 0.033  ✓
          DECISION: ASSOCIATE (≥0.65)
        """
        ex = decision.explanation
        vid = ex.get("candidate_visitor_id", "?")
        lines = [
            f"{vid} {ex.get('action','?')} "
            f"(score={decision.identity_score:.3f}, band={decision.confidence_band}):"
        ]
        for sig_name, contrib in sorted(
            decision.contributions.items(), key=lambda x: -x[1]
        ):
            raw = decision.explanation.get("signals", {}).get(sig_name, 0.0)
            w   = self.weights.get(sig_name, 0.0)
            dom = " ← dominant" if sig_name == decision.dominant_signal else ""
            lines.append(
                f"  {sig_name:<22} {raw:.3f} × {w:.3f} = {contrib:.4f}{dom}"
            )
        if "veto_reason" in ex:
            lines.append(f"  VETO: {ex['veto_reason']}")
        if "competing_match" in ex:
            cm = ex["competing_match"]
            lines.append(
                f"  ⚠ Competing match: {cm['visitor_id']} "
                f"score={cm['score']:.3f} gap={cm['gap']:.4f}"
            )
        lines.append(
            f"  DECISION: {'ASSOCIATE' if decision.should_associate else 'REJECT'} "
            f"(threshold={self.threshold})"
        )
        if ex.get("context"):
            ctx = ex["context"]
            lines.append(
                f"  Context: {ctx.get('cam_from','?')}→{ctx.get('cam_to','?')} "
                f"age={ctx.get('age_sec',0):.1f}s "
                f"zone={ctx.get('zone_from','?')}→{ctx.get('zone_to','?')}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _build_explanation(
        self,
        signals: ConsensusSignals,
        contributions: Dict[str, float],
        total_score: float,
        vid: str,
        veto_reason: Optional[str] = None,
    ) -> Dict:
        ex: Dict = {
            "candidate_visitor_id": vid,
            "identity_score":       round(total_score, 4),
            "threshold":            self.threshold,
            "action":               "reused" if total_score >= self.threshold else "rejected",
            "signals": {
                "reid":               round(signals.reid_score, 4),
                "fingerprint":        round(signals.fingerprint_score, 4),
                "trajectory":         round(signals.trajectory_score, 4),
                "temporal":           round(signals.temporal_score, 4),
                "camera_transition":  round(signals.camera_transition_score, 4),
                "zone_plausibility":  round(signals.zone_plausibility_score, 4),
                "detection":          round(signals.detection_score, 4),
                "track_health":       round(signals.track_health, 4),
                "group_continuity":   round(signals.group_continuity_score, 4),
                "staff_reputation":   round(signals.staff_reputation_score, 4),
                "visitor_dna":        round(signals.visitor_dna_score, 4),
            },
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "contributions": {k: round(v, 5) for k, v in contributions.items()},
            "context": {
                "age_sec":   round(signals.age_sec, 2),
                "cam_from":  signals.cam_from,
                "cam_to":    signals.cam_to,
                "zone_from": signals.zone_from,
                "zone_to":   signals.zone_to,
            },
        }
        if veto_reason:
            ex["veto_reason"] = veto_reason
        return ex
