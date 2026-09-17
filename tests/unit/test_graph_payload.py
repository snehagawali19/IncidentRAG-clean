"""Graph payload mapping must use only returned analysis fields."""

from __future__ import annotations

from incidentrag.approval.ui.graph_payload import (
    analysis_payload,
    assessment_payload,
    idle_payload,
)


def test_analysis_payload_has_no_fabricated_evidence_labels() -> None:
    payload = analysis_payload({"number": 12, "title": "Sync timeout", "state": "open"})
    assert payload["mode"] == "analysis"
    assert payload["incident"]["kind"] == "incident"
    assert "12" in payload["incident"]["sublabel"]
    kinds = {node["kind"] for node in payload["nodes"]}
    assert kinds <= {"incident", "process", "signal"}
    assert all(not node.get("label") for node in payload["nodes"] if node["kind"] == "signal")
    assert payload["meta"]["confidence"] is None


def test_assessment_payload_binds_real_evidence_and_hypotheses() -> None:
    result = {
        "incident_id": "inc-1",
        "source_issue": {"number": 9, "title": "Crash", "state": "open"},
        "confidence": 0.81,
        "human_approval_required": True,
        "facts_extracted": {"severity": "high", "component": "api"},
        "retrieved_evidence": [
            {
                "chunk_id": "c1",
                "runbook_id": "rb-1",
                "header_breadcrumb": "Timeout handling",
                "excerpt": "Increase wait",
                "relevance": "direct",
            }
        ],
        "ai_hypotheses": {
            "likely_diagnosis": {
                "statement": "Queue saturation",
                "confidence": 0.81,
                "verified": True,
                "evidence": [{"chunk_id": "c1"}],
            },
            "possible_contributing_factors": [
                {
                    "statement": "Retry storm",
                    "confidence": 0.4,
                    "evidence": [{"chunk_id": "c1"}],
                }
            ],
        },
        "recommended_investigation": [
            {"action_id": "a1", "description": "Inspect queue depth", "risk_level": "low"}
        ],
        "proposed_remediation": [
            {"action_id": "r1", "description": "Scale workers", "risk_level": "high"}
        ],
    }
    payload = assessment_payload(result)
    kinds = {node["kind"] for node in payload["nodes"]}
    assert "evidence" in kinds
    assert "hypothesis" in kinds
    assert "investigation" in kinds
    assert "approval" in kinds
    assert "risk" in kinds
    assert payload["meta"]["human_approval_required"] is True
    evidence_ids = {node["id"] for node in payload["nodes"] if node["kind"] == "evidence"}
    support = [
        edge for edge in payload["edges"] if edge["kind"] == "supports"
    ]
    assert support
    assert all(edge["source"] in evidence_ids for edge in support)
    hypo = next(node for node in payload["nodes"] if node.get("rank") == 1)
    assert hypo["confidence"] == 0.81
    approval = next(node for node in payload["nodes"] if node["kind"] == "approval")
    assert "HUMAN APPROVAL REQUIRED" in approval["detail"]["title"]
    assert all(node["kind"] not in {"process", "signal"} for node in payload["nodes"])


def test_idle_payload_anchors_selected_issue() -> None:
    payload = idle_payload({"number": 3, "title": "TLS handshake"})
    assert payload["mode"] == "idle"
    assert payload["nodes"][0]["kind"] == "incident"
    assert "TLS" in payload["nodes"][0]["label"]
