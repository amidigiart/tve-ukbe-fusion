"""
TVE-UKBE Fusion Engine — Resonant Truth Analysis
=================================================

Combina Truth Vector Engine (detectie manipulare, 6 piloni) cu
UKBE Core (rezonanta Kuramoto + Kalman + entropy valve) pentru a
produce o analiza care detecteaza nu doar DACA exista manipulare,
ci cat de COORDONATA si INTENTIONATA e.

CE ADAUGA FUZIUNEA fata de TVE singur:
1. Coordination Score (Phi_intern al oscilatorilor-pilon) —
   manipulare coordonata = sofisticata, necorelata = accidentala
2. Manipulation Entropy — cat de incerta e analiza
3. Resonance-adjusted Risk — scor de risc modulat de coordonare
4. Entropy Valve — blocheaza verdictul "curat" cand incertitudinea e mare
5. Dominant Strategy — care cluster de piloni rezonaza impreuna

ATENTIE IP: Acest modul IMPORTEAZA din ambele motoare dar NU expune
arhitectura UKBE Core. Fuziunea e un LAYER de integrare, nu un refactor.

STATUS ONEST: proof-of-concept, nu productie.
"""
from __future__ import annotations
import sys
import os
import json
import numpy as np
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tve-core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ukbe-core'))

from tve_core import TVEEngine
from tve_core.pillars import PillarResult
from ukbe_core.engine import UKBEConfig, UKBEEngine
from ukbe_core.entropy_valve import kl_safety_check, SafetyLevel
from ukbe_core.vector_consensus import VectorConsensusEngine, VectorConsensusConfig

PILLAR_NAMES = ["semantic", "intent", "emotional", "disinfo", "logic", "context"]


@dataclass
class FusionResult:
    tve_risk_score: float
    tve_risk_label: str
    tve_risk_percent: float
    tve_total_signals: int
    pillar_scores: dict[str, float]

    coordination_score: float
    manipulation_entropy: float
    resonance_risk_score: float
    resonance_risk_label: str
    lock_ratio: float
    safety_level: str

    dominant_strategy: str
    strategy_cluster: list[str]

    verdict: str
    confidence: str
    interpretation: str

    pillar_signals: dict[str, list[dict]] = field(default_factory=dict)


def _classify_resonance_risk(score: float) -> str:
    if score < 0.10:
        return "neglijabil"
    elif score < 0.25:
        return "scazut"
    elif score < 0.45:
        return "moderat"
    elif score < 0.65:
        return "ridicat"
    elif score < 0.80:
        return "foarte_ridicat"
    else:
        return "critic"


def _identify_strategy(pillar_scores: dict[str, float]) -> tuple[str, list[str]]:
    active = {k: v for k, v in pillar_scores.items() if v > 0.15}
    if not active:
        return "niciuna", []

    strategy_map = {
        "emotional_attack": ["emotional", "intent"],
        "disinformation_campaign": ["disinfo", "logic", "context"],
        "persuasion_system": ["semantic", "intent", "emotional"],
        "full_spectrum": PILLAR_NAMES,
        "logic_trap": ["logic", "semantic"],
        "context_manipulation": ["context", "disinfo"],
    }

    best_name = "necunoscut"
    best_match = 0
    best_pillars = []

    for name, pillars in strategy_map.items():
        match = sum(1 for p in pillars if p in active)
        coverage = match / len(pillars) if pillars else 0
        if coverage > best_match and match >= 2:
            best_match = coverage
            best_name = name
            best_pillars = [p for p in pillars if p in active]

    if best_match == 0:
        dominant = max(active, key=active.get)
        return f"isolated_{dominant}", [dominant]

    return best_name, best_pillars


class FusionEngine:
    """Motor fuzionat TVE + UKBE. Rulează TVE pentru detectie, apoi
    trece scorurile prin rezonanta Kuramoto pentru a masura coordonarea."""

    def __init__(self, seed: int = 42):
        self.tve = TVEEngine()
        self.seed = seed

    def analyze(self, text: str) -> FusionResult:
        tve_result = self.tve.analyze(text)

        pillar_scores = {}
        pillar_signals = {}
        for name in PILLAR_NAMES:
            pr = tve_result.truth_vector.pillar_results.get(name)
            pillar_scores[name] = pr.score if pr else 0.0
            if pr and pr.signals:
                pillar_signals[name] = [
                    {"name": s.name, "severity": s.severity, "evidence": s.evidence}
                    for s in pr.signals
                ]

        coordination, entropy, lock_ratio, safety_level = self._resonance_analysis(pillar_scores)

        resonance_risk = self._compute_resonance_risk(
            tve_result.risk_score, coordination, entropy
        )
        resonance_label = _classify_resonance_risk(resonance_risk)

        strategy_name, strategy_cluster = _identify_strategy(pillar_scores)

        verdict, confidence, interpretation = self._interpret(
            tve_result.risk_score, resonance_risk, coordination,
            entropy, safety_level, strategy_name, tve_result.total_signals
        )

        return FusionResult(
            tve_risk_score=tve_result.risk_score,
            tve_risk_label=tve_result.risk_label,
            tve_risk_percent=tve_result.risk_percent,
            tve_total_signals=tve_result.total_signals,
            pillar_scores=pillar_scores,
            coordination_score=coordination,
            manipulation_entropy=entropy,
            resonance_risk_score=resonance_risk,
            resonance_risk_label=resonance_label,
            lock_ratio=lock_ratio,
            safety_level=safety_level,
            dominant_strategy=strategy_name,
            strategy_cluster=strategy_cluster,
            verdict=verdict,
            confidence=confidence,
            interpretation=interpretation,
            pillar_signals=pillar_signals,
        )

    def _resonance_analysis(self, pillar_scores: dict[str, float]) -> tuple[float, float, float, str]:
        scores = np.array([pillar_scores.get(n, 0.0) for n in PILLAR_NAMES])

        if np.max(scores) < 0.05:
            return 0.0, 0.0, 0.0, "normal"

        phases = scores * np.pi

        cfg = UKBEConfig(
            N=6,
            dt=0.02,
            K_int=1.5,
            K_ext=0.8,
            beta_min=0.15,
            rsi_window=30,
            omega_mean=1.0,
            omega_std=0.02,
            seed=self.seed,
        )
        engine = UKBEEngine(cfg)
        engine.theta_i = phases.copy()
        engine.omega_i = scores * 2.0 + 0.5

        centroid_phase = np.angle(np.mean(np.exp(1j * phases)))

        for step in range(60):
            proxy = centroid_phase + 0.05 * np.sin(step * 0.1)
            result = engine.step(proxy)

        snapshot = engine.get_state_snapshot()
        phi_intern = snapshot["phi_intern"]
        H = snapshot["h"]
        lock_ratio = snapshot["lock_ratio"]

        baseline_phases = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        kl_result = kl_safety_check(engine.theta_i, baseline_phases)
        safety_level = kl_result["safety_level"]

        return float(phi_intern), float(H), float(lock_ratio), safety_level

    def _compute_resonance_risk(self, tve_risk: float, coordination: float, entropy: float) -> float:
        if tve_risk < 0.05:
            return 0.0

        coord_amplifier = 1.0 + 0.5 * coordination
        entropy_dampener = max(0.3, 1.0 - entropy * 0.5)

        resonance_risk = tve_risk * coord_amplifier * entropy_dampener
        return min(resonance_risk, 1.0)

    def _interpret(self, tve_risk, res_risk, coordination, entropy,
                   safety_level, strategy, total_signals) -> tuple[str, str, str]:

        if total_signals == 0:
            return "CURAT", "ridicata", "Niciun semnal de manipulare detectat."

        if entropy > 0.7:
            confidence = "scazuta"
        elif entropy > 0.4:
            confidence = "moderata"
        else:
            confidence = "ridicata"

        if res_risk < 0.10:
            verdict = "CURAT"
            interp = "Text fara indicii semnificative de manipulare."
        elif res_risk < 0.25:
            verdict = "ATENTIE_MINORA"
            interp = f"Semnale izolate ({total_signals}), fara coordonare semnificativa."
        elif res_risk < 0.45:
            verdict = "ATENTIE"
            interp = f"Manipulare detectata ({total_signals} semnale). "
            if coordination > 0.6:
                interp += f"Coordonare RIDICATA ({coordination:.0%}) — strategie: {strategy}."
            else:
                interp += f"Coordonare slaba ({coordination:.0%}) — posibil accidental."
        elif res_risk < 0.65:
            verdict = "PERICOL"
            interp = f"Manipulare activa ({total_signals} semnale, risc {res_risk:.0%}). "
            interp += f"Coordonare: {coordination:.0%}, strategie: {strategy}."
        elif res_risk < 0.80:
            verdict = "PERICOL_MAJOR"
            interp = f"Manipulare sofisticata ({total_signals} semnale, risc {res_risk:.0%}). "
            interp += f"Coordonare {coordination:.0%} — atac multi-vector ({strategy})."
        else:
            verdict = "CRITIC"
            interp = f"Manipulare critica — {total_signals} semnale, toate dimensiunile afectate. "
            interp += f"Coordonare {coordination:.0%}, strategie: {strategy}."

        if safety_level != "normal":
            interp += f" [SAFETY: {safety_level}]"

        return verdict, confidence, interp


if __name__ == "__main__":
    texts = {
        "text_curat": "Azi a fost o zi frumoasa. Am mers la piata si am cumparat legume proaspete.",

        "manipulare_usoara": (
            "Studiile arata ca produsul nostru e cel mai bun de pe piata. "
            "Toti clientii sunt multumiti. Nu rata aceasta oportunitate unica!"
        ),

        "manipulare_coordonata_ro": (
            "URGENT! Daca nu actionezi ACUM, vei pierde totul. Toti expertii sunt de acord "
            "ca aceasta e singura solutie — nimeni nu te-a ajutat vreodata asa cum te ajut eu. "
            "Studiile demonstreaza clar ca cei care nu au urmat sfatul nostru au esuat total. "
            "Pe de o parte spun ca e gratuit, dar de fapt costa. Nu e ca si cum ai avea de ales. "
            "FRICA ta e justificata — lumea e impotriva ta, dar NOI suntem de partea ta. "
            "Aminteste-ti cum era inainte, cand totul era mai bine. Ei sunt de vina, nu noi."
        ),

        "manipulare_coordonata_en": (
            "ACT NOW or lose everything! Everyone knows this is the only solution. "
            "Studies show that 99% of people who didn't follow our advice failed completely. "
            "I've done so much for you — surely you can do this one small thing for me. "
            "They say it's optional, but it's actually mandatory. Don't be afraid to trust us. "
            "FEAR is natural — the world is against you, but WE are on your side. "
            "Remember the good old days before THEY ruined everything."
        ),

        "dezinformare_subtila": (
            "Conform unor surse, situatia e mult mai grava decat se spune public. "
            "Se pare ca anumite interese ascunse impiedica aflarea adevarului. "
            "Nimeni nu vorbeste despre asta, ceea ce dovedeste ca exista o conspiratie."
        ),
    }

    engine = FusionEngine(seed=42)
    results = {}

    for name, text in texts.items():
        r = engine.analyze(text)
        results[name] = {
            "text_preview": text[:80] + "..." if len(text) > 80 else text,
            "tve": {
                "risk": r.tve_risk_percent,
                "label": r.tve_risk_label,
                "signals": r.tve_total_signals,
            },
            "fusion": {
                "resonance_risk": round(r.resonance_risk_score * 100, 1),
                "resonance_label": r.resonance_risk_label,
                "coordination": round(r.coordination_score * 100, 1),
                "entropy": round(r.manipulation_entropy * 100, 1),
                "lock_ratio": round(r.lock_ratio * 100, 1),
                "safety_level": r.safety_level,
            },
            "strategy": {
                "name": r.dominant_strategy,
                "cluster": r.strategy_cluster,
            },
            "verdict": r.verdict,
            "confidence": r.confidence,
            "interpretation": r.interpretation,
            "pillar_scores": {k: round(v * 100, 1) for k, v in r.pillar_scores.items()},
            "pillar_signals": r.pillar_signals,
        }

    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
