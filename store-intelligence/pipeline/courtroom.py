"""
courtroom.py — Identity Courtroom Framework.

Adversarial reasoning for difficult identity matches.
DEFENDER argues for matching.
PROSECUTOR argues against matching.
JUDGE decides.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class CourtroomVerdict:
    should_match:          bool
    confidence:            float
    defender_arguments:    List[str]
    prosecutor_arguments:  List[str]
    judge_rationale:       str

    def to_dict(self) -> dict:
        return {
            "should_match": self.should_match,
            "confidence": round(self.confidence, 4),
            "defender": self.defender_arguments,
            "prosecutor": self.prosecutor_arguments,
            "judge_rationale": self.judge_rationale,
        }


class IdentityCourtroom:
    """
    Adjudicates ambiguous matches.
    """
    
    def __init__(self):
        # Activation bands configuration. e.g. "LOW" means activate for LOW or REJECT.
        self._activation_bands = ["LOW", "REJECT"] if cfg.COURTROOM_ACTIVATION_BAND == "LOW" else \
                                 ["MEDIUM", "LOW", "REJECT"] if cfg.COURTROOM_ACTIVATION_BAND == "MEDIUM" else \
                                 ["HIGH", "MEDIUM", "LOW", "REJECT"]

    def adjudicate(
        self,
        candidate_signals: Dict[str, float],
        base_score: float,
        confidence_band: str,
        context: dict = None
    ) -> Optional[CourtroomVerdict]:
        """
        Adjudicate an identity match if it falls within the activation band.
        Returns CourtroomVerdict or None if not activated.
        """
        if confidence_band not in self._activation_bands:
            return None
            
        defender_args = []
        prosecutor_args = []
        
        def_score = 0.0
        pros_score = 0.0
        
        # Context extraction
        ctx = context or {}
        
        # 1. Appearance (ReID + Fingerprint)
        reid_score = candidate_signals.get("reid", 0.0)
        fingerprint_score = candidate_signals.get("fingerprint", 0.0)
        
        if reid_score > 0.8:
            defender_args.append(f"Strong ReID match ({reid_score:.2f})")
            def_score += 0.4
        elif reid_score < 0.4 and reid_score > 0:
            prosecutor_args.append(f"Poor ReID match ({reid_score:.2f})")
            pros_score += 0.3
            
        if fingerprint_score > 0.8:
            defender_args.append(f"Strong fingerprint match ({fingerprint_score:.2f})")
            def_score += 0.3
        elif fingerprint_score < 0.4 and fingerprint_score > 0:
            prosecutor_args.append(f"Poor fingerprint match ({fingerprint_score:.2f})")
            pros_score += 0.2

        # 2. Trajectory & Physics
        traj_score = candidate_signals.get("trajectory", 0.0)
        physics_score = candidate_signals.get("physics", candidate_signals.get("store_graph", 0.0))
        
        if physics_score < 0.1:
            prosecutor_args.append("Physics violation: teleportation or impossible speed")
            pros_score += 0.8  # Major veto power
        elif physics_score > 0.8:
            defender_args.append(f"Highly plausible physical transition ({physics_score:.2f})")
            def_score += 0.2
            
        if traj_score > 0.7:
            defender_args.append(f"Trajectory alignment is good ({traj_score:.2f})")
            def_score += 0.2
            
        # 3. Time & Temporal
        temp_score = candidate_signals.get("temporal", 0.0)
        if temp_score < 0.2:
            prosecutor_args.append("Temporal gap is too large")
            pros_score += 0.2
            
        # 4. Group & DNA
        group_score = candidate_signals.get("group_continuity", 0.0)
        if group_score > 0.5:
            defender_args.append("Group members are present nearby")
            def_score += 0.3
            
        dna_score = candidate_signals.get("visitor_dna", 0.0)
        if dna_score > 0.7:
            defender_args.append(f"Behavioral DNA is very similar ({dna_score:.2f})")
            def_score += 0.2
            
        # The Judge decides
        judge_rationale = ""
        should_match = False
        final_conf = base_score
        
        if pros_score >= cfg.COURTROOM_VETO_THRESHOLD:
            judge_rationale = "Prosecutor established reasonable doubt (Veto threshold reached)."
            should_match = False
            final_conf *= 0.5
        elif def_score < cfg.COURTROOM_MIN_DEFENDER_SCORE:
            judge_rationale = "Defender failed to build a strong enough case for association."
            should_match = False
            final_conf *= 0.8
        elif def_score > pros_score * 1.5:
            judge_rationale = f"Defender successfully argued for match despite {confidence_band} base score."
            should_match = True
            final_conf = min(0.99, base_score + 0.15)
        else:
            judge_rationale = "Evidence is inconclusive; defaulting to safe rejection."
            should_match = False
            
        verdict = CourtroomVerdict(
            should_match=should_match,
            confidence=final_conf,
            defender_arguments=defender_args,
            prosecutor_arguments=prosecutor_args,
            judge_rationale=judge_rationale,
        )
        
        logger.debug(f"Courtroom convened: {judge_rationale}")
        return verdict
