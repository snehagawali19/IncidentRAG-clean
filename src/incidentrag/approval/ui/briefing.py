"""Turn a returned assessment into a plain-language operator briefing."""

from __future__ import annotations

from html import escape
from typing import Any


def _text(value: object) -> str:
    return " ".join(str(value or "").split())


def _pct(value: object) -> str | None:
    if isinstance(value, (int, float)):
        return f"{float(value):.0%}"
    return None


def build_briefing(result: dict[str, Any]) -> dict[str, Any]:
    """Extract only fields present on the analysis response."""
    source = result.get("source_issue") or {}
    hypotheses = result.get("ai_hypotheses") or {}
    diagnosis = hypotheses.get("likely_diagnosis") or {}
    factors = [
        item
        for item in (hypotheses.get("possible_contributing_factors") or [])
        if isinstance(item, dict) and _text(item.get("statement"))
    ]
    evidence = [
        item
        for item in (result.get("retrieved_evidence") or [])
        if isinstance(item, dict)
    ]
    investigation = [
        item
        for item in (result.get("recommended_investigation") or [])
        if isinstance(item, dict) and _text(item.get("description"))
    ]
    remediation = [
        item
        for item in (result.get("proposed_remediation") or [])
        if isinstance(item, dict) and _text(item.get("description"))
    ]
    missing = [str(item) for item in (result.get("additional_information_required") or []) if str(item).strip()]
    return {
        "issue_number": source.get("number"),
        "issue_title": _text(source.get("title")),
        "diagnosis": _text(diagnosis.get("statement")),
        "diagnosis_confidence": diagnosis.get("confidence"),
        "verified": bool(diagnosis.get("verified")),
        "overall_confidence": result.get("confidence"),
        "severity": _text((result.get("facts_extracted") or {}).get("severity")),
        "component": _text((result.get("facts_extracted") or {}).get("component")),
        "factors": [_text(item.get("statement")) for item in factors],
        "evidence": [
            {
                "title": _text(item.get("header_breadcrumb")) or "Runbook evidence",
                "excerpt": _text(item.get("excerpt")),
                "source": _text(item.get("runbook_id")),
                "relevance": _text(item.get("relevance")),
            }
            for item in evidence
        ],
        "investigation": [_text(item.get("description")) for item in investigation],
        "remediation": [
            {
                "description": _text(item.get("description")),
                "risk": _text(item.get("risk_level")),
                "command": _text(item.get("command")),
            }
            for item in remediation
        ],
        "approval_required": bool(result.get("human_approval_required")),
        "missing": missing,
    }


def _field(label: str, value: str) -> str:
    if not value:
        return ""
    return (
        f"<div class='brief-kicker'>{escape(label)}</div>"
        f"<div class='brief-copy'>{escape(value)}</div>"
    )


def render_briefing_html(result: dict[str, Any], *, sanitize: Any) -> str:
    brief = build_briefing(result)
    title = sanitize(brief["issue_title"])
    diagnosis = sanitize(brief["diagnosis"]) or "No diagnosis statement was returned."
    number = brief["issue_number"] or "—"
    confidence = _pct(brief["overall_confidence"]) or _pct(brief["diagnosis_confidence"]) or "not provided"
    verified = "Grounded in retrieved evidence" if brief["verified"] else "Needs human review"
    chips = []
    if brief["component"]:
        chips.append(sanitize(brief["component"]))
    if brief["severity"]:
        chips.append(sanitize(str(brief["severity"]).upper()))
    chips.append(f"Confidence {confidence}")
    chip_html = "".join(f"<span class='brief-chip'>{escape(str(chip))}</span>" for chip in chips)

    evidence_html = ""
    for index, item in enumerate(brief["evidence"], 1):
        meta = " · ".join(
            part
            for part in (
                sanitize(item["source"]),
                sanitize(item["relevance"]).title() if item["relevance"] else "",
            )
            if part
        )
        evidence_html += (
            "<article class='brief-card'>"
            f"<div class='brief-kicker'>Evidence {index:02d}</div>"
            f"<strong>{escape(sanitize(item['title']))}</strong>"
            f"<p>{escape(sanitize(item['excerpt']) or 'No excerpt was returned.')}</p>"
            f"<span>{escape(meta)}</span>"
            "</article>"
        )
    if not evidence_html:
        evidence_html = "<p class='brief-empty'>No runbook evidence was returned.</p>"

    factors_html = "".join(
        f"<li>{escape(sanitize(item))}</li>" for item in brief["factors"]
    )
    factors_block = (
        f"<div class='brief-block'><div class='brief-kicker'>Also consider</div><ul>{factors_html}</ul></div>"
        if factors_html
        else ""
    )

    steps_html = "".join(
        f"<li><b>{index:02d}</b><span>{escape(sanitize(step))}</span></li>"
        for index, step in enumerate(brief["investigation"], 1)
    )
    steps_block = (
        f"<ol class='brief-steps'>{steps_html}</ol>"
        if steps_html
        else "<p class='brief-empty'>No investigation steps were returned.</p>"
    )

    approval_html = ""
    if brief["approval_required"]:
        approval_html = (
            "<div class='brief-alert'>Human approval is required before any proposed change. "
            "Nothing shown here has been executed.</div>"
        )
    remediate_html = ""
    for item in brief["remediation"]:
        risk = sanitize(item["risk"]).upper() if item["risk"] else ""
        command = sanitize(item["command"])
        remediate_html += (
            "<article class='brief-card'>"
            f"<div class='brief-kicker'>Proposed change{' · ' + escape(risk) if risk else ''}</div>"
            f"<p>{escape(sanitize(item['description']))}</p>"
            + (f"<code>{escape(command)}</code>" if command else "")
            + "</article>"
        )

    missing_html = ""
    if brief["missing"]:
        items = "".join(f"<li>{escape(sanitize(item))}</li>" for item in brief["missing"])
        missing_html = (
            "<div class='brief-block'><div class='brief-kicker'>Would improve confidence</div>"
            f"<ul>{items}</ul></div>"
        )

    return f"""
<section class="briefing">
  <div class="brief-kicker">What is going on</div>
  <h2>Issue #{escape(str(number))}: {escape(title) if title else "Assessment"}</h2>
  <div class="brief-chips">{chip_html}</div>
  {_field("Likely cause", diagnosis)}
  <p class="brief-note">{escape(verified)}.</p>
  {factors_block}
  <div class="brief-kicker">Why we think that</div>
  <div class="brief-grid">{evidence_html}</div>
  <div class="brief-kicker">What to check next</div>
  {steps_block}
  {approval_html}
  {remediate_html}
  {missing_html}
</section>
"""
