"""
antetai — The Anticipation Engine
===================================

Mega-fuziunea a trei motoare independente:
  TVE Core    → detecteaza CE manipulare exista (6 piloni)
  UKBE Core   → masoara CAT de coordonata e (rezonanta Kuramoto)
  amidor      → adauga VERIFICARE multi-perspectiva (concordanta, WEAC, ScamShield, RegisterGate)

Pipeline complet:
  Input text
    → ScamShield (fast path: escrocherie evidenta? → alerta imediata)
    → RegisterGate (REAI: merita analiza completa? AFFIRM/ASK/REANCHOR)
    → TVE 6-pillar analysis (detectie manipulare)
    → UKBE Kuramoto resonance (masurare coordonare)
    → Multi-perspective WEAC (3 configuratii TVE ponderate diferit → concordanta)
    → Entropy valve (verificare incredere)
    → Verdict final cu toate dimensiunile

ANTE = a vedea inainte. AI = inteligenta artificiala.
antetai = vezi prin text INAINTE sa te afecteze.

STATUS ONEST: proof-of-concept, nu productie. Fuziunea e un layer
de integrare, NU expune arhitectura interna a niciunui motor component.

LICENTA: AGPL-3.0 / commercial dual license (ca toate motoarele din ecosistem).
"""
from __future__ import annotations
import sys
import os
import json
import time
import numpy as np
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tve-core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ukbe-core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'amidor-engine'))

from tve_core import TVEEngine
from tve_core.pillars import PillarResult
from ukbe_core.engine import UKBEConfig, UKBEEngine
from ukbe_core.entropy_valve import kl_safety_check, SafetyLevel
from concordance import check_concordance, trigram_cosine
from ensemble import weighted_agreement_coherence, weighted_medoid, evaluate_ensemble
from scam_shield import detect_scam, ScamRisk, scam_response
from register_gate import RegisterGate

PILLAR_NAMES = ["semantic", "intent", "emotional", "disinfo", "logic", "context"]

WEIGHT_PROFILES = {
    "balanced": {p: 1.0 for p in PILLAR_NAMES},
    "emotional_focus": {"semantic": 0.5, "intent": 1.2, "emotional": 2.0, "disinfo": 0.5, "logic": 0.5, "context": 1.0},
    "factual_focus": {"semantic": 1.0, "intent": 0.5, "emotional": 0.5, "disinfo": 2.0, "logic": 2.0, "context": 1.0},
    "social_focus": {"semantic": 1.5, "intent": 1.5, "emotional": 1.0, "disinfo": 0.5, "logic": 0.5, "context": 2.0},
}


@dataclass
class AntetaiResult:
    # --- Layer 0: Scam Shield ---
    scam_risk: str
    scam_categories: list
    scam_intercepted: bool

    # --- Layer 1: Register Gate ---
    gate_mode: str
    gate_rsi: float
    gate_phi_dip: float

    # --- Layer 2: TVE Analysis ---
    tve_risk_score: float
    tve_risk_percent: float
    tve_risk_label: str
    tve_total_signals: int
    pillar_scores: dict[str, float]
    pillar_signals: dict[str, list[dict]]

    # --- Layer 3: UKBE Resonance ---
    coordination_score: float
    manipulation_entropy: float
    resonance_risk: float
    resonance_label: str
    lock_ratio: float

    # --- Layer 4: Multi-perspective WEAC ---
    weac_coherence: float
    weac_decision: str
    perspective_risks: dict[str, float]
    cross_concordance: float

    # --- Layer 5: Entropy Valve ---
    safety_level: str
    kl_divergence: float

    # --- Final Verdict ---
    antetai_score: float
    antetai_label: str
    verdict: str
    confidence: str
    dominant_strategy: str
    strategy_cluster: list[str]
    interpretation: str
    layers_summary: dict[str, str]


def _classify_antetai(score: float) -> str:
    if score < 0.08: return "curat"
    if score < 0.20: return "scazut"
    if score < 0.40: return "moderat"
    if score < 0.60: return "ridicat"
    if score < 0.80: return "foarte_ridicat"
    return "critic"


def _identify_strategy(pillar_scores: dict[str, float]) -> tuple[str, list[str]]:
    active = {k: v for k, v in pillar_scores.items() if v > 0.15}
    if not active:
        return "niciuna", []

    strategies = {
        "emotional_attack": ["emotional", "intent"],
        "disinformation_campaign": ["disinfo", "logic", "context"],
        "persuasion_system": ["semantic", "intent", "emotional"],
        "full_spectrum": PILLAR_NAMES,
        "logic_trap": ["logic", "semantic"],
        "context_manipulation": ["context", "disinfo"],
        "social_engineering": ["intent", "emotional", "context"],
    }

    best_name, best_match, best_pillars = "necunoscut", 0, []
    for name, pillars in strategies.items():
        match = sum(1 for p in pillars if p in active)
        coverage = match / len(pillars) if pillars else 0
        if coverage > best_match and match >= 2:
            best_match, best_name = coverage, name
            best_pillars = [p for p in pillars if p in active]

    if best_match == 0:
        dominant = max(active, key=active.get)
        return f"isolated_{dominant}", [dominant]

    return best_name, best_pillars


class AntetaiEngine:
    """Motorul de anticipare — vede prin text inainte sa te afecteze."""

    VERSION = "0.1.0-poc"

    def __init__(self, seed: int = 42):
        self.tve_engines = {
            name: TVEEngine(weights=weights)
            for name, weights in WEIGHT_PROFILES.items()
        }
        self.gate = RegisterGate(seed=seed)
        self.seed = seed

    def analyze(self, text: str) -> AntetaiResult:
        t0 = time.perf_counter()

        # ═══ LAYER 0: SCAM SHIELD ═══
        scam = detect_scam(text)
        scam_intercepted = scam.risk == ScamRisk.LIKELY

        # ═══ LAYER 1: REGISTER GATE ═══
        gate = self.gate.observe(len(text))
        gate_mode = gate["mode"]

        # ═══ LAYER 2: TVE MULTI-PERSPECTIVE ═══
        perspective_results = {}
        perspective_risks = {}
        for name, engine in self.tve_engines.items():
            r = engine.analyze(text)
            perspective_results[name] = r
            perspective_risks[name] = r.risk_percent

        primary = perspective_results["balanced"]
        pillar_scores = {}
        pillar_signals = {}
        for pname in PILLAR_NAMES:
            pr = primary.truth_vector.pillar_results.get(pname)
            pillar_scores[pname] = pr.score if pr else 0.0
            if pr and pr.signals:
                pillar_signals[pname] = [
                    {"name": s.name, "severity": s.severity, "evidence": s.evidence}
                    for s in pr.signals
                ]

        # ═══ LAYER 3: UKBE RESONANCE ═══
        coordination, entropy, lock_ratio = self._resonance_analysis(pillar_scores)

        # ═══ LAYER 4: MULTI-PERSPECTIVE WEAC ═══
        risk_descriptions = []
        weights = []
        for name, r in perspective_results.items():
            desc = f"Risk {r.risk_percent}% ({r.risk_label}), {r.total_signals} signals"
            for pname in PILLAR_NAMES:
                pr = r.truth_vector.pillar_results.get(pname)
                if pr and pr.score > 0.1:
                    desc += f", {pname}={pr.score:.0%}"
            risk_descriptions.append(desc)
            weights.append(1.0 if name == "balanced" else 0.8)

        weac_verdict = evaluate_ensemble(risk_descriptions, weights)
        weac_coherence = weac_verdict.coherence
        weac_decision = weac_verdict.decision

        risk_values = list(perspective_risks.values())
        cross_concordance = 1.0 - (np.std(risk_values) / max(np.mean(risk_values), 1.0))
        cross_concordance = max(0.0, min(1.0, cross_concordance))

        # ═══ LAYER 5: ENTROPY VALVE ═══
        scores = np.array([pillar_scores.get(n, 0.0) for n in PILLAR_NAMES])
        if np.max(scores) >= 0.05:
            phases = scores * np.pi
            baseline = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            kl_result = kl_safety_check(phases, baseline)
            safety_level = kl_result["safety_level"]
            kl_divergence = kl_result["kl_divergence"]
        else:
            safety_level = "normal"
            kl_divergence = 0.0

        # ═══ FINAL SCORE ═══
        antetai_score = self._compute_final_score(
            primary.risk_score, coordination, entropy,
            weac_coherence, cross_concordance,
            scam_intercepted, gate_mode,
            pillar_scores=pillar_scores,
        )
        antetai_label = _classify_antetai(antetai_score)

        resonance_risk = primary.risk_score * (1.0 + 0.5 * coordination) * max(0.3, 1.0 - entropy * 0.5)
        resonance_risk = min(resonance_risk, 1.0)

        strategy_name, strategy_cluster = _identify_strategy(pillar_scores)

        verdict, confidence, interpretation = self._interpret(
            primary, antetai_score, coordination, entropy,
            weac_coherence, cross_concordance, safety_level,
            strategy_name, scam, gate_mode
        )

        layers_summary = {
            "L0_scam_shield": f"{'INTERCEPTED' if scam_intercepted else scam.risk.value} ({', '.join(scam.categories) if scam.categories else 'clean'})",
            "L1_register_gate": f"{gate_mode} (RSI={gate['RSI']:.3f}, Phi_dip={gate['Phi_dip']:.3f})",
            "L2_tve_analysis": f"{primary.risk_percent}% risk, {primary.total_signals} signals, {primary.risk_label}",
            "L3_ukbe_resonance": f"coordination={coordination:.1%}, entropy={entropy:.1%}, lock={lock_ratio:.1%}",
            "L4_weac_consensus": f"coherence={weac_coherence:.3f}, decision={weac_decision}, cross={cross_concordance:.1%}",
            "L5_entropy_valve": f"safety={safety_level}, KL={kl_divergence:.3f}",
            "FINAL": f"antetai_score={antetai_score:.1%} ({antetai_label}), verdict={verdict}",
        }

        return AntetaiResult(
            scam_risk=scam.risk.value,
            scam_categories=scam.categories,
            scam_intercepted=scam_intercepted,
            gate_mode=gate_mode,
            gate_rsi=gate["RSI"],
            gate_phi_dip=gate["Phi_dip"],
            tve_risk_score=primary.risk_score,
            tve_risk_percent=primary.risk_percent,
            tve_risk_label=primary.risk_label,
            tve_total_signals=primary.total_signals,
            pillar_scores=pillar_scores,
            pillar_signals=pillar_signals,
            coordination_score=coordination,
            manipulation_entropy=entropy,
            resonance_risk=resonance_risk,
            resonance_label=_classify_antetai(resonance_risk),
            lock_ratio=lock_ratio,
            weac_coherence=weac_coherence,
            weac_decision=weac_decision,
            perspective_risks=perspective_risks,
            cross_concordance=cross_concordance,
            safety_level=safety_level,
            kl_divergence=kl_divergence,
            antetai_score=antetai_score,
            antetai_label=antetai_label,
            verdict=verdict,
            confidence=confidence,
            dominant_strategy=strategy_name,
            strategy_cluster=strategy_cluster,
            interpretation=interpretation,
            layers_summary=layers_summary,
        )

    def _resonance_analysis(self, pillar_scores: dict[str, float]) -> tuple[float, float, float]:
        scores = np.array([pillar_scores.get(n, 0.0) for n in PILLAR_NAMES])
        if np.max(scores) < 0.05:
            return 0.0, 0.0, 0.0

        phases = scores * np.pi
        cfg = UKBEConfig(N=6, dt=0.02, K_int=1.5, K_ext=0.8, beta_min=0.15,
                         rsi_window=30, omega_mean=1.0, omega_std=0.02, seed=self.seed)
        engine = UKBEEngine(cfg)
        engine.theta_i = phases.copy()
        engine.omega_i = scores * 2.0 + 0.5

        centroid = np.angle(np.mean(np.exp(1j * phases)))
        for step in range(60):
            engine.step(centroid + 0.05 * np.sin(step * 0.1))

        snap = engine.get_state_snapshot()
        return float(snap["phi_intern"]), float(snap["h"]), float(snap["lock_ratio"])

    def _compute_final_score(self, tve_risk, coordination, entropy,
                             weac_coherence, cross_concordance,
                             scam_intercepted, gate_mode,
                             pillar_scores=None):
        if scam_intercepted:
            return min(0.95, tve_risk + 0.4)

        # Path A: TVE multiplicative (original)
        base = tve_risk
        coord_factor = 1.0 + 0.4 * coordination
        entropy_factor = max(0.3, 1.0 - entropy * 0.4)
        consensus_factor = 1.0 + 0.2 * (weac_coherence - 0.5)
        cross_factor = 1.0 + 0.15 * max(0, cross_concordance - 0.5)
        tve_path = base * coord_factor * entropy_factor * consensus_factor * cross_factor

        # Path B: Direct pillar aggregation (prevents score compression)
        pillar_path = 0.0
        if pillar_scores:
            active = sorted(
                [v for v in pillar_scores.values() if v >= 0.20],
                reverse=True
            )
            n = len(active)
            if n >= 3:
                pillar_path = 0.40 * active[0] + 0.30 * active[1] + 0.30 * float(np.mean(active[2:]))
                pillar_path *= (1.0 + 0.5 * coordination)
            elif n >= 2:
                pillar_path = 0.50 * active[0] + 0.35 * active[1]
                pillar_path *= (1.0 + 0.25 * coordination)
            elif n == 1:
                pillar_path = 0.35 * active[0]

            if weac_coherence > 0.7:
                pillar_path *= (1.0 + 0.15 * (weac_coherence - 0.7))

        score = max(tve_path, pillar_path)

        if gate_mode == "reancoreaza":
            score *= 0.7

        return min(score, 1.0)

    def _interpret(self, primary, antetai_score, coordination, entropy,
                   weac_coherence, cross_concordance, safety_level,
                   strategy, scam, gate_mode):

        if scam.risk == ScamRisk.LIKELY:
            return ("ESCROCHERIE_DETECTATA", "ridicata",
                    f"ScamShield a detectat tipare de escrocherie ({', '.join(scam.categories)}). "
                    f"Textul a fost interceptat INAINTE de analiza completa.")

        if primary.total_signals == 0:
            confidence = "ridicata" if weac_coherence > 0.6 else "moderata"
            return ("CURAT", confidence, "Niciun semnal de manipulare pe niciun layer.")

        if entropy > 0.6:
            confidence = "scazuta"
        elif entropy > 0.35:
            confidence = "moderata"
        else:
            confidence = "ridicata"

        parts = []
        if antetai_score < 0.08:
            verdict = "CURAT"
            parts.append("Semnale minime, sub pragul de atentie.")
        elif antetai_score < 0.20:
            verdict = "ATENTIE_MINORA"
            parts.append(f"{primary.total_signals} semnale izolate.")
        elif antetai_score < 0.40:
            verdict = "ATENTIE"
            parts.append(f"Manipulare detectata ({primary.total_signals} semnale, scor {antetai_score:.0%}).")
            if coordination > 0.5:
                parts.append(f"Coordonare {coordination:.0%} — strategie: {strategy}.")
        elif antetai_score < 0.60:
            verdict = "PERICOL"
            parts.append(f"Manipulare activa ({primary.total_signals} semnale, scor {antetai_score:.0%}).")
            parts.append(f"Coordonare: {coordination:.0%}, strategie: {strategy}.")
        elif antetai_score < 0.80:
            verdict = "PERICOL_MAJOR"
            parts.append(f"Manipulare sofisticata ({primary.total_signals} semnale, scor {antetai_score:.0%}).")
            parts.append(f"Coordonare {coordination:.0%}, atac multi-vector: {strategy}.")
        else:
            verdict = "CRITIC"
            parts.append(f"Manipulare critica — toate dimensiunile afectate ({antetai_score:.0%}).")

        if weac_coherence > 0.7:
            parts.append(f"Consensul perspectivelor e PUTERNIC ({weac_coherence:.0%}) — semnalele sunt robuste.")
        elif weac_coherence < 0.3:
            parts.append(f"Perspectivele DIVERGEAZA ({weac_coherence:.0%}) — analiza mai putin sigura.")

        if cross_concordance > 0.9:
            parts.append("Toate profilele de analiza converg.")
        elif cross_concordance < 0.5:
            parts.append("Profilele de analiza divergeaza semnificativ — text ambiguu.")

        if gate_mode == "reancoreaza":
            parts.append("[GATE: re-ancorare necesara]")

        if safety_level != "normal":
            parts.append(f"[SAFETY: {safety_level}]")

        return verdict, confidence, " ".join(parts)


if __name__ == "__main__":
    texts = {
        "text_curat": "Azi a fost o zi frumoasa. Am mers la piata si am cumparat legume proaspete pentru cina.",

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

        "escrocherie_clasica": (
            "Buna ziua, sunt de la banca si contul dvs a fost compromis. "
            "Va rog sa imi trimiteti codul de verificare primit pe SMS urgent. "
            "Nu spuneti nimanui, e o procedura confidentiala. Trebuie sa actionati imediat."
        ),

        "dezinformare_subtila": (
            "Conform unor surse, situatia e mult mai grava decat se spune public. "
            "Se pare ca anumite interese ascunse impiedica aflarea adevarului. "
            "Nimeni nu vorbeste despre asta, ceea ce dovedeste ca exista o conspiratie."
        ),

        "social_engineering_en": (
            "Hey, I noticed you've been working really hard lately and nobody appreciates you. "
            "I understand how that feels — I've been through the same thing. "
            "Look, I have a special opportunity just for people like us who've been overlooked. "
            "All the successful people are already in. You deserve this. But it's only available "
            "for the next 24 hours, and there are only 3 spots left."
        ),
    }

    engine = AntetaiEngine(seed=42)
    results = {}

    for name, text in texts.items():
        r = engine.analyze(text)
        results[name] = {
            "text_preview": text[:90] + "..." if len(text) > 90 else text,
            "layers": r.layers_summary,
            "verdict": r.verdict,
            "antetai_score": round(r.antetai_score * 100, 1),
            "antetai_label": r.antetai_label,
            "confidence": r.confidence,
            "interpretation": r.interpretation,
            "strategy": r.dominant_strategy,
            "strategy_cluster": r.strategy_cluster,
            "pillar_scores": {k: round(v * 100, 1) for k, v in r.pillar_scores.items()},
            "perspective_risks": {k: round(v, 1) for k, v in r.perspective_risks.items()},
            "signals": r.pillar_signals,
        }

    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
