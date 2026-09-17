"""Plain-language briefing extracted from returned analysis JSON."""

from __future__ import annotations

from incidentrag.approval.ui.briefing import build_briefing, render_briefing_html


def test_briefing_uses_diagnosis_and_evidence_from_response() -> None:
    brief = build_briefing(
        {
            "confidence": 0.75,
            "human_approval_required": True,
            "source_issue": {"number": 42, "title": "TLS cannot be disabled"},
            "facts_extracted": {"severity": "sev2", "component": "repo-server"},
            "ai_hypotheses": {
                "likely_diagnosis": {
                    "statement": "TLS is required by the current config",
                    "confidence": 0.75,
                    "verified": True,
                },
                "possible_contributing_factors": [{"statement": "Config mismatch"}],
            },
            "retrieved_evidence": [
                {
                    "header_breadcrumb": "TLS settings",
                    "excerpt": "Disable only when insecure is explicit",
                    "runbook_id": "tls.md",
                    "relevance": "direct",
                }
            ],
            "recommended_investigation": [{"description": "Inspect TLS flags"}],
            "proposed_remediation": [
                {
                    "description": "Change server config",
                    "risk_level": "high",
                    "command": "echo review",
                }
            ],
            "additional_information_required": ["Exact helm values"],
        }
    )
    assert brief["diagnosis"] == "TLS is required by the current config"
    assert brief["evidence"][0]["excerpt"].startswith("Disable only")
    assert brief["investigation"] == ["Inspect TLS flags"]
    assert brief["approval_required"] is True
    html = render_briefing_html(
        {
            "source_issue": {"number": 1, "title": "x"},
            "ai_hypotheses": {"likely_diagnosis": {"statement": "Cause A"}},
            "retrieved_evidence": [],
            "recommended_investigation": [],
        },
        sanitize=lambda value: str(value),
    )
    assert "What is going on" in html
    assert "Cause A" in html
    assert "No runbook evidence was returned" in html
