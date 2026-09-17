"""Tests for Phase 9 — Approval Gate."""
from __future__ import annotations

import pytest

from incidentrag.approval.gate import ApprovalGate, ApprovalStatus, RiskClassifier
from incidentrag.core.models import RiskLevel
from tests.unit.conftest import make_assessment, make_query, make_remediation_action


class TestRiskClassifier:
    def test_upgrades_kubectl_delete_all(self):
        clf = RiskClassifier()
        action = make_remediation_action(
            command="kubectl delete all -n payments",
            risk_level=RiskLevel.MEDIUM,
        )
        assert clf.classify(action) == RiskLevel.DANGEROUS

    def test_upgrades_kubectl_drain(self):
        clf = RiskClassifier()
        action = make_remediation_action(
            command="kubectl drain node-1 --force",
            risk_level=RiskLevel.HIGH,
        )
        assert clf.classify(action) == RiskLevel.DANGEROUS

    def test_upgrades_rm_rf(self):
        clf = RiskClassifier()
        action = make_remediation_action(
            command="rm -rf /var/log/",
            risk_level=RiskLevel.LOW,
        )
        assert clf.classify(action) == RiskLevel.DANGEROUS

    def test_upgrades_s3_recursive_delete(self):
        clf = RiskClassifier()
        action = make_remediation_action(
            command="aws s3 rm s3://bucket/data/ --recursive",
            risk_level=RiskLevel.MEDIUM,
        )
        assert clf.classify(action) == RiskLevel.DANGEROUS

    def test_upgrades_drop_table(self):
        clf = RiskClassifier()
        action = make_remediation_action(
            command="DROP TABLE users",
            action_type="shell",
            risk_level=RiskLevel.LOW,
        )
        assert clf.classify(action) == RiskLevel.DANGEROUS

    def test_safe_command_not_upgraded(self):
        clf = RiskClassifier()
        action = make_remediation_action(
            command="kubectl logs payment-xyz -n payments --tail=100",
            risk_level=RiskLevel.LOW,
        )
        assert clf.classify(action) == RiskLevel.LOW

    def test_no_command_not_upgraded(self):
        clf = RiskClassifier()
        action = make_remediation_action(
            command=None,
            action_type="manual",
            risk_level=RiskLevel.MEDIUM,
        )
        assert clf.classify(action) == RiskLevel.MEDIUM


class TestApprovalGate:
    def test_medium_becomes_pending(self, sample_query):
        gate = ApprovalGate()
        action = make_remediation_action(risk_level=RiskLevel.MEDIUM)
        assessment = make_assessment(sample_query, actions=[action])

        requests = gate.process(assessment)
        assert len(requests) == 1
        assert requests[0].status == ApprovalStatus.PENDING

    def test_dangerous_becomes_escalated(self, sample_query):
        gate = ApprovalGate()
        action = make_remediation_action(
            command="kubectl delete all -n payments",
            risk_level=RiskLevel.HIGH,  # will be upgraded to DANGEROUS by classifier
        )
        assessment = make_assessment(sample_query, actions=[action])

        requests = gate.process(assessment)
        assert requests[0].status == ApprovalStatus.ESCALATED

    def test_approve_flow(self, sample_query):
        gate = ApprovalGate()
        action = make_remediation_action(risk_level=RiskLevel.MEDIUM)
        assessment = make_assessment(sample_query, actions=[action])

        requests = gate.process(assessment)
        approved = gate.approve(requests[0].request_id, approver="alice", notes="safe restart")

        assert approved.status == ApprovalStatus.APPROVED
        assert approved.approver == "alice"
        assert approved.resolved_at is not None

    def test_reject_flow(self, sample_query):
        gate = ApprovalGate()
        action = make_remediation_action(risk_level=RiskLevel.HIGH)
        assessment = make_assessment(sample_query, actions=[action])

        requests = gate.process(assessment)
        rejected = gate.reject(requests[0].request_id, approver="bob", notes="too risky")

        assert rejected.status == ApprovalStatus.REJECTED
        assert rejected.approver == "bob"

    def test_cannot_approve_escalated(self, sample_query):
        gate = ApprovalGate()
        action = make_remediation_action(
            command="kubectl delete all -n payments",
            risk_level=RiskLevel.HIGH,
        )
        assessment = make_assessment(sample_query, actions=[action])

        requests = gate.process(assessment)
        escalated = [r for r in requests if r.status == ApprovalStatus.ESCALATED]
        assert escalated

        with pytest.raises(ValueError, match="ESCALATED"):
            gate.approve(escalated[0].request_id, approver="alice")

    def test_list_pending(self, sample_query):
        gate = ApprovalGate()
        actions = [
            make_remediation_action(risk_level=RiskLevel.MEDIUM, description="a1"),
            make_remediation_action(risk_level=RiskLevel.HIGH, description="a2"),
            make_remediation_action(
                command="kubectl delete all -n prod",
                risk_level=RiskLevel.HIGH,
                description="a3-dangerous",
            ),
        ]
        assessment = make_assessment(sample_query, actions=actions)
        gate.process(assessment)

        pending = gate.list_pending()
        # 2 pending (a1 medium, a2 high) — a3 was escalated
        assert len(pending) == 2

    def test_get_returns_none_for_unknown(self):
        gate = ApprovalGate()
        assert gate.get("nonexistent") is None
