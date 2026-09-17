"""Map operator-console state and analysis JSON into a graph payload."""

from __future__ import annotations

from typing import Any

PROCESSING_STAGES: tuple[str, ...] = (
    "INGEST",
    "CONTEXT",
    "RETRIEVAL",
    "RELATIONSHIPS",
    "EVIDENCE",
    "REASONING",
    "VALIDATION",
    "ASSESSMENT",
)


def _short(text: str, limit: int = 72) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _incident_from_issue(issue: dict[str, Any] | None) -> dict[str, Any]:
    issue = issue or {}
    return {
        "id": f"incident-{issue.get('number', 'unknown')}",
        "kind": "incident",
        "label": _short(str(issue.get("title") or "Selected incident"), 42),
        "sublabel": f"#{issue.get('number', '—')} · {issue.get('state') or 'open'}",
        "weight": 1.0,
        "detail": {
            "title": "INCIDENT",
            "fields": [
                {"label": "Issue", "value": str(issue.get("number", "—"))},
                {"label": "Title", "value": str(issue.get("title") or "")},
                {"label": "State", "value": str(issue.get("state") or "")},
                {
                    "label": "Labels",
                    "value": " · ".join(issue.get("labels") or []) or "None",
                },
            ],
        },
    }


def analysis_payload(issue: dict[str, Any] | None) -> dict[str, Any]:
    """Abstract processing graph used while the analyze request is in flight."""
    incident = _incident_from_issue(issue)
    nodes: list[dict[str, Any]] = [incident]
    edges: list[dict[str, Any]] = []
    previous = incident["id"]
    for index, stage in enumerate(PROCESSING_STAGES):
        node_id = f"process-{stage.lower()}"
        nodes.append(
            {
                "id": node_id,
                "kind": "process",
                "label": stage,
                "sublabel": "signal",
                "weight": 0.72 if index >= 4 else 0.58,
                "detail": {
                    "title": stage,
                    "fields": [
                        {
                            "label": "Role",
                            "value": "Conceptual analysis activity — not a completed backend stage",
                        }
                    ],
                },
            }
        )
        edges.append({"source": previous, "target": node_id, "kind": "flow"})
        previous = node_id
        for satellite in range(2):
            sat_id = f"signal-{stage.lower()}-{satellite}"
            nodes.append(
                {
                    "id": sat_id,
                    "kind": "signal",
                    "label": "",
                    "sublabel": "",
                    "weight": 0.22,
                    "detail": {
                        "title": "SIGNAL",
                        "fields": [
                            {
                                "label": "Status",
                                "value": "Unlabeled processing signal",
                            }
                        ],
                    },
                }
            )
            edges.append({"source": node_id, "target": sat_id, "kind": "orbit"})
    return {
        "mode": "analysis",
        "incident": incident,
        "nodes": nodes,
        "edges": edges,
        "meta": {"risk": None, "confidence": None, "human_approval_required": False},
    }


def idle_payload(issue: dict[str, Any] | None) -> dict[str, Any]:
    incident = _incident_from_issue(issue)
    nodes = [incident]
    edges: list[dict[str, Any]] = []
    for index in range(6):
        node_id = f"idle-{index}"
        nodes.append(
            {
                "id": node_id,
                "kind": "signal",
                "label": "",
                "weight": 0.2,
                "detail": {"title": "SIGNAL", "fields": []},
            }
        )
        edges.append({"source": incident["id"], "target": node_id, "kind": "idle"})
    return {
        "mode": "idle",
        "incident": incident,
        "nodes": nodes,
        "edges": edges,
        "meta": {"risk": None, "confidence": None, "human_approval_required": False},
    }


def _evidence_nodes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for index, item in enumerate(items, 1):
        chunk_id = str(item.get("chunk_id") or f"evidence-{index}")
        relevance = item.get("relevance")
        nodes.append(
            {
                "id": f"evidence:{chunk_id}",
                "kind": "evidence",
                "label": _short(
                    str(item.get("header_breadcrumb") or f"Evidence {index:02d}"),
                    36,
                ),
                "sublabel": f"EVIDENCE //{index:02d}",
                "weight": 0.64,
                "relevance": relevance,
                "detail": {
                    "title": f"EVIDENCE //{index:02d}",
                    "fields": [
                        {"label": "Source", "value": str(item.get("runbook_id") or "unknown")},
                        {"label": "Relevance", "value": str(relevance or "unspecified")},
                        {
                            "label": "Summary",
                            "value": str(item.get("header_breadcrumb") or ""),
                        },
                        {
                            "label": "Supporting context",
                            "value": str(item.get("excerpt") or ""),
                        },
                    ],
                },
            }
        )
    return nodes


def _hypothesis_nodes(result: dict[str, Any]) -> list[dict[str, Any]]:
    hypotheses = result.get("ai_hypotheses") or {}
    ranked: list[tuple[int, dict[str, Any], str]] = []
    diagnosis = hypotheses.get("likely_diagnosis")
    if isinstance(diagnosis, dict) and diagnosis:
        ranked.append((1, diagnosis, "hypothesis-root"))
    for offset, factor in enumerate(hypotheses.get("possible_contributing_factors") or [], 2):
        if isinstance(factor, dict):
            ranked.append((offset, factor, f"hypothesis-{offset}"))
    nodes: list[dict[str, Any]] = []
    for rank, claim, node_id in ranked:
        confidence = claim.get("confidence")
        weight = 0.9 if rank == 1 else 0.55
        if isinstance(confidence, (int, float)):
            weight = max(0.35, min(1.0, float(confidence) * (1.05 if rank == 1 else 0.85)))
        nodes.append(
            {
                "id": node_id,
                "kind": "hypothesis",
                "label": _short(str(claim.get("statement") or f"Hypothesis {rank:02d}"), 40),
                "sublabel": f"HYPOTHESIS //{rank:02d}",
                "weight": weight,
                "rank": rank,
                "confidence": confidence,
                "claim_id": claim.get("claim_id"),
                "evidence_ids": [
                    str(item.get("chunk_id"))
                    for item in claim.get("evidence") or []
                    if item.get("chunk_id")
                ],
                "detail": {
                    "title": f"HYPOTHESIS //{rank:02d}",
                    "fields": [
                        {"label": "Rank", "value": str(rank)},
                        {
                            "label": "Confidence",
                            "value": (
                                f"{float(confidence):.0%}"
                                if isinstance(confidence, (int, float))
                                else "Not provided"
                            ),
                        },
                        {"label": "Reasoning", "value": str(claim.get("statement") or "")},
                        {
                            "label": "Verification",
                            "value": (
                                "Grounding verified"
                                if claim.get("verified")
                                else "Needs review"
                            ),
                        },
                        {
                            "label": "Supporting evidence",
                            "value": str(len(claim.get("evidence") or [])),
                        },
                    ],
                },
            }
        )
    return nodes


def _action_nodes(
    actions: list[dict[str, Any]], *, kind: str, prefix: str
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for index, action in enumerate(actions, 1):
        action_id = str(action.get("action_id") or f"{prefix}-{index}")
        risk = action.get("risk_level")
        nodes.append(
            {
                "id": f"{prefix}:{action_id}",
                "kind": kind,
                "label": _short(str(action.get("description") or f"Step {index:02d}"), 40),
                "sublabel": f"{prefix.upper()} //{index:02d}",
                "weight": 0.5,
                "step": index,
                "risk": risk,
                "approval": kind == "approval",
                "detail": {
                    "title": (
                        "HUMAN APPROVAL REQUIRED"
                        if kind == "approval"
                        else f"INVESTIGATION //{index:02d}"
                    ),
                    "fields": [
                        {"label": "Description", "value": str(action.get("description") or "")},
                        {"label": "Risk", "value": str(risk or "unspecified")},
                        {"label": "Command", "value": str(action.get("command") or "None")},
                        {
                            "label": "Expected impact",
                            "value": str(action.get("expected_impact") or ""),
                        },
                        {
                            "label": "Duration (s)",
                            "value": str(action.get("estimated_duration_seconds") or ""),
                        },
                    ],
                },
            }
        )
    return nodes


def _highest_risk(actions: list[dict[str, Any]]) -> str | None:
    order = {"low": 1, "medium": 2, "high": 3, "dangerous": 4, "critical": 5}
    best: str | None = None
    best_rank = 0
    for action in actions:
        level = str(action.get("risk_level") or "").lower()
        rank = order.get(level, 0)
        if rank > best_rank:
            best = level
            best_rank = rank
    return best


def assessment_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Interactive knowledge graph built only from returned analysis fields."""
    source = result.get("source_issue") or {}
    incident = _incident_from_issue(source)
    evidence = _evidence_nodes(list(result.get("retrieved_evidence") or []))
    hypotheses = _hypothesis_nodes(result)
    investigation = _action_nodes(
        list(result.get("recommended_investigation") or []),
        kind="investigation",
        prefix="invest",
    )
    remediation = list(result.get("proposed_remediation") or [])
    approval_required = bool(result.get("human_approval_required"))
    approvals = (
        _action_nodes(remediation, kind="approval", prefix="approve")
        if approval_required
        else _action_nodes(remediation, kind="remediation", prefix="remediate")
    )

    confidence = result.get("confidence")
    facts = result.get("facts_extracted") or {}
    risk_value = facts.get("severity") or _highest_risk(remediation)

    extra: list[dict[str, Any]] = []
    if confidence is not None:
        extra.append(
            {
                "id": "confidence",
                "kind": "confidence",
                "label": (
                    f"{float(confidence):.0%}"
                    if isinstance(confidence, (int, float))
                    else str(confidence)
                ),
                "sublabel": "CONFIDENCE",
                "weight": float(confidence) if isinstance(confidence, (int, float)) else 0.5,
                "confidence": confidence,
                "detail": {
                    "title": "CONFIDENCE",
                    "fields": [
                        {
                            "label": "Overall",
                            "value": (
                                f"{float(confidence):.0%}"
                                if isinstance(confidence, (int, float))
                                else str(confidence)
                            ),
                        }
                    ],
                },
            }
        )
    if risk_value:
        extra.append(
            {
                "id": "risk",
                "kind": "risk",
                "label": str(risk_value).upper(),
                "sublabel": "CURRENT RISK",
                "weight": 0.7,
                "risk": str(risk_value),
                "detail": {
                    "title": "CURRENT RISK",
                    "fields": [
                        {"label": "Risk level", "value": str(risk_value)},
                        {
                            "label": "Component",
                            "value": str(facts.get("component") or ""),
                        },
                        {
                            "label": "Environment",
                            "value": str(facts.get("environment") or ""),
                        },
                    ],
                },
            }
        )
    extra.append(
        {
            "id": "assessment-core",
            "kind": "assessment",
            "label": "ASSESSMENT",
            "sublabel": "SYNTHESIS",
            "weight": 0.8,
            "detail": {
                "title": "ASSESSMENT",
                "fields": [
                    {
                        "label": "Incident",
                        "value": str(result.get("incident_id") or ""),
                    },
                    {
                        "label": "Diagnosis",
                        "value": str(
                            ((result.get("ai_hypotheses") or {}).get("likely_diagnosis") or {}).get(
                                "statement"
                            )
                            or ""
                        ),
                    },
                ],
            },
        }
    )

    nodes = [incident, *evidence, *hypotheses, *investigation, *approvals, *extra]
    edges: list[dict[str, Any]] = []
    evidence_ids = {node["id"] for node in evidence}

    for item in evidence:
        edges.append({"source": incident["id"], "target": item["id"], "kind": "evidence"})
    for hypo in hypotheses:
        edges.append({"source": incident["id"], "target": hypo["id"], "kind": "hypothesis"})
        edges.append({"source": hypo["id"], "target": "assessment-core", "kind": "synthesis"})
        for chunk_id in hypo.get("evidence_ids") or []:
            evidence_id = f"evidence:{chunk_id}"
            if evidence_id in evidence_ids:
                edges.append(
                    {"source": evidence_id, "target": hypo["id"], "kind": "supports"}
                )
    previous_invest: str | None = None
    for step in investigation:
        source = hypotheses[0]["id"] if hypotheses else incident["id"]
        if previous_invest is None:
            edges.append({"source": source, "target": step["id"], "kind": "investigate"})
        else:
            edges.append(
                {"source": previous_invest, "target": step["id"], "kind": "sequence"}
            )
        previous_invest = step["id"]
    for action in approvals:
        origin = "assessment-core"
        edges.append({"source": origin, "target": action["id"], "kind": "approval"})
    if extra:
        for node in extra:
            if node["id"] != "assessment-core":
                edges.append(
                    {"source": incident["id"], "target": node["id"], "kind": node["kind"]}
                )
        edges.append(
            {"source": incident["id"], "target": "assessment-core", "kind": "assessment"}
        )
    if risk_value and any(node["id"] == "risk" for node in extra) and previous_invest:
        edges.append({"source": "risk", "target": previous_invest, "kind": "risk-path"})

    return {
        "mode": "assessment",
        "incident": incident,
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "risk": risk_value,
            "confidence": confidence,
            "human_approval_required": approval_required,
        },
    }
