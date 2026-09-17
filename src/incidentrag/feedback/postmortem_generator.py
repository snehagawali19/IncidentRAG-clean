"""Generate a reviewable Markdown postmortem from a resolved assessment."""

from __future__ import annotations

from incidentrag.core.models import IncidentAssessment, RawAlert


class PostmortemGenerator:
    def generate(self, assessment: IncidentAssessment, alert: RawAlert) -> str:
        actions = "\n".join(
            f"- {action.description} (`{action.risk_level.value}` risk)"
            for action in assessment.proposed_actions
        ) or "- No remediation action recorded."
        evidence = "\n".join(
            f"- `{item.chunk_id}` — {item.header_breadcrumb}"
            for item in assessment.root_cause.evidence
        )
        return (
            f"# Incident {assessment.incident_id}\n\n"
            f"## Summary\n\n{alert.alert_text}\n\n"
            f"## Root cause\n\n{assessment.root_cause.statement}\n\n"
            f"Confidence: {assessment.overall_confidence:.2f}\n\n"
            f"## Evidence\n\n{evidence}\n\n"
            f"## Remediation\n\n{actions}\n\n"
            "## Follow-up\n\n"
            "- Review this AI-assisted draft with the incident owner.\n"
        )
