"""
Phase 9 — Approval Gate.

Routes RemediationActions based on risk_level:
  LOW       → auto_execute if settings.auto_execute_low_risk, else pending
  MEDIUM    → pending (requires approval)
  HIGH      → pending (requires senior approval)
  DANGEROUS → escalated (blocked from execution)

Uses a RiskClassifier that OVERRIDES the LLM's risk_level if the command
matches known dangerous patterns.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from incidentrag.core.models import IncidentAssessment, RemediationAction, RiskLevel
from incidentrag.core.settings import get_settings

logger = logging.getLogger(__name__)


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_EXECUTED = "auto_executed"
    ESCALATED = "escalated"


class ApprovalRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    incident_id: str
    action: RemediationAction
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver: str | None = None
    notes: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None


class RiskClassifier:
    """Overrides LLM-supplied risk_level for commands matching dangerous patterns."""

    _DANGEROUS_PATTERNS = [
        re.compile(r"kubectl\s+delete\s+all\b", re.I),
        re.compile(r"kubectl\s+drain\b", re.I),
        re.compile(r"aws\s+rds\s+delete-db-instance", re.I),
        re.compile(r"aws\s+s3\s+rm.*--recursive", re.I),
        re.compile(r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b", re.I),
        re.compile(r"\bTRUNCATE\s+TABLE\b", re.I),
        re.compile(r"\brm\s+-[a-z]*r[a-z]*f\b", re.I),
        re.compile(r"\brm\s+-[a-z]*f[a-z]*r\b", re.I),
    ]

    def classify(self, action: RemediationAction) -> RiskLevel:
        if action.command:
            for pattern in self._DANGEROUS_PATTERNS:
                if pattern.search(action.command):
                    if action.risk_level != RiskLevel.DANGEROUS:
                        logger.warning(
                            "Upgrading action risk %s → dangerous: %s",
                            action.risk_level.value, action.command[:80],
                        )
                    return RiskLevel.DANGEROUS
        return action.risk_level


class ApprovalGate:
    def __init__(self) -> None:
        self._classifier = RiskClassifier()
        self._requests: dict[str, ApprovalRequest] = {}
        self._auto_execute_low = get_settings().auto_execute_low_risk

    def process(self, assessment: IncidentAssessment) -> list[ApprovalRequest]:
        requests: list[ApprovalRequest] = []
        for action in assessment.proposed_actions:
            action.risk_level = self._classifier.classify(action)

            if action.risk_level == RiskLevel.DANGEROUS:
                status = ApprovalStatus.ESCALATED
            elif action.risk_level == RiskLevel.LOW and self._auto_execute_low:
                status = ApprovalStatus.AUTO_EXECUTED
            else:
                status = ApprovalStatus.PENDING

            req = ApprovalRequest(
                incident_id=assessment.incident_id,
                action=action,
                status=status,
            )
            self._requests[req.request_id] = req
            requests.append(req)
        return requests

    def approve(self, request_id: str, approver: str, notes: str = "") -> ApprovalRequest:
        req = self._requests[request_id]
        if req.status == ApprovalStatus.ESCALATED:
            raise ValueError(f"Request {request_id} is ESCALATED and cannot be approved here")
        req.status = ApprovalStatus.APPROVED
        req.approver = approver
        req.notes = notes
        req.resolved_at = datetime.now(timezone.utc)
        return req

    def reject(self, request_id: str, approver: str, notes: str = "") -> ApprovalRequest:
        req = self._requests[request_id]
        req.status = ApprovalStatus.REJECTED
        req.approver = approver
        req.notes = notes
        req.resolved_at = datetime.now(timezone.utc)
        return req

    def get(self, request_id: str) -> ApprovalRequest | None:
        return self._requests.get(request_id)

    def list_pending(self) -> list[ApprovalRequest]:
        return [r for r in self._requests.values() if r.status == ApprovalStatus.PENDING]
