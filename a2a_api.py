# -*- coding: utf-8 -*-
"""
A2A Oversight API — FastAPI wrapper for the Agent-to-Agent Oversight Engine
==========================================================================

Endpoints:
  POST /agents/register          — register an agent identity
  GET  /agents/{agent_id}        — agent trust report
  GET  /agents                   — list all agents

  POST /evaluate                 — evaluate a single A2A message
  POST /evaluate/batch           — evaluate multiple messages

  GET  /health                   — system health (Kuramoto coherence, trust stats)
  GET  /audit                    — audit log (paginated)

  POST /antetai/stamp            — standalone antetai text analysis (no A2A)

LICENTA: AGPL-3.0 / commercial dual license.
"""
from __future__ import annotations

import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from a2a_oversight import (
    A2AOversightEngine, A2AMessage, AgentIdentity,
    EscalationSignal, CoalitionSignal,
)
from antetai_engine import AntetaiEngine

# ═══════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════

class AgentRegisterRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=128)
    agent_type: str = Field("llm", pattern=r"^(llm|tool|service|human_proxy|orchestrator)$")
    declared_purpose: str = ""
    authority_scope: list[str] = Field(default_factory=list)
    initial_trust: float = Field(0.7, ge=0.0, le=1.0)

class AgentResponse(BaseModel):
    agent_id: str
    type: str
    purpose: str
    trust_score: float
    trust_phase: float
    messages_sent: int
    flags: int
    blocks: int
    recent_decisions: list[str]
    avg_antetai_score: float

class MessageRequest(BaseModel):
    sender_id: str = Field(..., min_length=1, max_length=128)
    receiver_id: str = Field(..., min_length=1, max_length=128)
    content: str = Field(..., min_length=1, max_length=50000)
    message_type: str = Field("request", pattern=r"^(request|response|negotiation|notification|command)$")
    declared_action: str = ""

class BatchMessageRequest(BaseModel):
    messages: list[MessageRequest] = Field(..., min_length=1, max_length=100)

class EscalationOut(BaseModel):
    pattern: str
    severity: float
    turn_span: int
    evidence: list[str]

class CoalitionOut(BaseModel):
    agents: list[str]
    coordination_score: float
    shared_strategy: str
    target_id: str

class PillarScores(BaseModel):
    semantic: float = 0.0
    intent: float = 0.0
    emotional: float = 0.0
    disinfo: float = 0.0
    logic: float = 0.0
    context: float = 0.0

class AntetaiLayerOut(BaseModel):
    scam_risk: str
    scam_categories: list[str]
    scam_intercepted: bool
    gate_mode: str
    tve_risk_score: float
    tve_total_signals: int
    pillar_scores: PillarScores
    coordination_score: float
    manipulation_entropy: float
    weac_coherence: float
    safety_level: str
    kl_divergence: float
    antetai_score: float
    antetai_label: str
    verdict: str
    confidence: str
    dominant_strategy: str

class VerdictResponse(BaseModel):
    decision: str
    confidence: str
    reasoning: str
    sender_trust_before: float
    sender_trust_after: float
    trust_delta: float
    authority_check: str
    escalation_signals: list[EscalationOut]
    coalition_signals: list[CoalitionOut]
    antetai: AntetaiLayerOut
    processing_ms: float
    layer_trace: dict[str, str]

class BatchVerdictResponse(BaseModel):
    results: list[VerdictResponse]
    total_ms: float

class HealthResponse(BaseModel):
    status: str
    agents: int
    system_coherence: float
    avg_trust: float
    min_trust: float
    flagged_agents: int
    blocked_agents: int
    total_messages: int
    total_flags: int
    engine_version: str

class StampRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)

class StampResponse(BaseModel):
    antetai_score: float
    antetai_label: str
    verdict: str
    confidence: str
    scam_risk: str
    scam_categories: list[str]
    scam_intercepted: bool
    tve_total_signals: int
    pillar_scores: PillarScores
    coordination_score: float
    weac_coherence: float
    dominant_strategy: str
    safety_level: str

class AuditEntry(BaseModel):
    ts: float
    sender: str
    receiver: str
    action: str
    decision: str
    antetai_score: float
    trust_delta: float
    escalations: int
    coalitions: int

class AuditResponse(BaseModel):
    total: int
    offset: int
    limit: int
    entries: list[AuditEntry]


# ═══════════════════════════════════════════════════════════════════
# APP SETUP
# ═══════════════════════════════════════════════════════════════════

engine = A2AOversightEngine(seed=42)

app = FastAPI(
    title="AntetAI A2A Oversight API",
    description="Agent-to-Agent oversight — antetai as arbiter between autonomous AI agents",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _antetai_to_out(r) -> AntetaiLayerOut:
    return AntetaiLayerOut(
        scam_risk=r.scam_risk,
        scam_categories=r.scam_categories,
        scam_intercepted=r.scam_intercepted,
        gate_mode=r.gate_mode,
        tve_risk_score=round(r.tve_risk_score, 4),
        tve_total_signals=r.tve_total_signals,
        pillar_scores=PillarScores(**{k: round(v, 4) for k, v in r.pillar_scores.items()}),
        coordination_score=round(r.coordination_score, 4),
        manipulation_entropy=round(r.manipulation_entropy, 4),
        weac_coherence=round(r.weac_coherence, 4),
        safety_level=r.safety_level,
        kl_divergence=round(r.kl_divergence, 4),
        antetai_score=round(r.antetai_score, 4),
        antetai_label=r.antetai_label,
        verdict=r.verdict,
        confidence=r.confidence,
        dominant_strategy=r.dominant_strategy,
    )

def _verdict_to_response(v) -> VerdictResponse:
    return VerdictResponse(
        decision=v.decision,
        confidence=v.confidence,
        reasoning=v.reasoning,
        sender_trust_before=round(v.sender_trust_before, 4),
        sender_trust_after=round(v.sender_trust_after, 4),
        trust_delta=round(v.trust_delta, 4),
        authority_check=v.authority_check,
        escalation_signals=[
            EscalationOut(
                pattern=e.pattern,
                severity=round(e.severity, 4),
                turn_span=e.turn_span,
                evidence=e.evidence,
            ) for e in v.escalation_signals
        ],
        coalition_signals=[
            CoalitionOut(
                agents=c.agents,
                coordination_score=round(c.coordination_score, 4),
                shared_strategy=c.shared_strategy,
                target_id=c.target_id,
            ) for c in v.coalition_signals
        ],
        antetai=_antetai_to_out(v.antetai_result),
        processing_ms=round(v.processing_ms, 2),
        layer_trace=v.layer_trace,
    )


# ═══════════════════════════════════════════════════════════════════
# AGENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.post("/agents/register", response_model=AgentResponse, tags=["agents"])
async def register_agent(req: AgentRegisterRequest):
    if req.agent_id in engine.registry:
        raise HTTPException(409, f"Agent '{req.agent_id}' already registered")
    engine.register_agent(
        agent_id=req.agent_id,
        agent_type=req.agent_type,
        purpose=req.declared_purpose,
        scope=req.authority_scope,
        initial_trust=req.initial_trust,
    )
    return engine.get_agent_report(req.agent_id)

@app.get("/agents/{agent_id}", response_model=AgentResponse, tags=["agents"])
async def get_agent(agent_id: str):
    report = engine.get_agent_report(agent_id)
    if "error" in report:
        raise HTTPException(404, report["error"])
    return report

@app.get("/agents", tags=["agents"])
async def list_agents():
    return {
        "agents": [
            {
                "agent_id": a.agent_id,
                "type": a.agent_type,
                "trust_score": round(a.trust_score, 4),
                "messages": a.message_count,
                "flags": a.flag_count,
                "blocks": a.block_count,
            }
            for a in engine.registry.values()
        ],
        "total": len(engine.registry),
    }


# ═══════════════════════════════════════════════════════════════════
# EVALUATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.post("/evaluate", response_model=VerdictResponse, tags=["oversight"])
async def evaluate_message(req: MessageRequest):
    msg = A2AMessage(
        sender_id=req.sender_id,
        receiver_id=req.receiver_id,
        content=req.content,
        timestamp=time.time(),
        message_type=req.message_type,
        declared_action=req.declared_action,
    )
    verdict = engine.evaluate(msg)
    return _verdict_to_response(verdict)

@app.post("/evaluate/batch", response_model=BatchVerdictResponse, tags=["oversight"])
async def evaluate_batch(req: BatchMessageRequest):
    t0 = time.perf_counter()
    results = []
    for m in req.messages:
        msg = A2AMessage(
            sender_id=m.sender_id,
            receiver_id=m.receiver_id,
            content=m.content,
            timestamp=time.time(),
            message_type=m.message_type,
            declared_action=m.declared_action,
        )
        verdict = engine.evaluate(msg)
        results.append(_verdict_to_response(verdict))
    total_ms = (time.perf_counter() - t0) * 1000
    return BatchVerdictResponse(results=results, total_ms=round(total_ms, 2))


# ═══════════════════════════════════════════════════════════════════
# STANDALONE ANTETAI STAMP
# ═══════════════════════════════════════════════════════════════════

@app.post("/antetai/stamp", response_model=StampResponse, tags=["antetai"])
async def stamp_text(req: StampRequest):
    r = engine.antetai.analyze(req.text)
    return StampResponse(
        antetai_score=round(r.antetai_score, 4),
        antetai_label=r.antetai_label,
        verdict=r.verdict,
        confidence=r.confidence,
        scam_risk=r.scam_risk,
        scam_categories=r.scam_categories,
        scam_intercepted=r.scam_intercepted,
        tve_total_signals=r.tve_total_signals,
        pillar_scores=PillarScores(**{k: round(v, 4) for k, v in r.pillar_scores.items()}),
        coordination_score=round(r.coordination_score, 4),
        weac_coherence=round(r.weac_coherence, 4),
        dominant_strategy=r.dominant_strategy,
        safety_level=r.safety_level,
    )


# ═══════════════════════════════════════════════════════════════════
# SYSTEM ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def system_health():
    h = engine.system_health()
    return HealthResponse(
        status=h.get("status", "no_agents"),
        agents=h.get("agents", 0),
        system_coherence=h.get("system_coherence", 0.0),
        avg_trust=h.get("avg_trust", 0.0),
        min_trust=h.get("min_trust", 0.0),
        flagged_agents=h.get("flagged_agents", 0),
        blocked_agents=h.get("blocked_agents", 0),
        total_messages=h.get("total_messages", 0),
        total_flags=h.get("total_flags", 0),
        engine_version=A2AOversightEngine.VERSION,
    )

@app.get("/audit", response_model=AuditResponse, tags=["system"])
async def audit_log(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    agent_id: Optional[str] = Query(None),
    decision: Optional[str] = Query(None, pattern=r"^(PASS|FLAG|BLOCK|ESCALATE_TO_HUMAN)$"),
):
    entries = engine.audit_log
    if agent_id:
        entries = [e for e in entries if e["sender"] == agent_id]
    if decision:
        entries = [e for e in entries if e["decision"] == decision]
    total = len(entries)
    page = entries[offset:offset + limit]
    return AuditResponse(total=total, offset=offset, limit=limit, entries=page)


# ═══════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("a2a_api:app", host="0.0.0.0", port=8100, reload=True)
