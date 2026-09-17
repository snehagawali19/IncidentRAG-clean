"""Tests for Layer 9 — Sandboxed Executor + SigmaHQ Rules."""
from __future__ import annotations

import pytest

from incidentrag.execution.sandbox import SandboxExecutor, scan_sigma


class TestSigmaRules:
    def test_kubectl_delete_all_is_critical(self):
        violations = scan_sigma("kubectl delete all -n payments")
        assert any("delete_all_resources" in v for v in violations)

    def test_kubectl_drain_is_critical(self):
        violations = scan_sigma("kubectl drain node-1 --force")
        assert any("drain_node" in v for v in violations)

    def test_rm_rf_is_critical(self):
        violations = scan_sigma("rm -rf /var/log/app/*")
        assert any("rm_rf" in v for v in violations)

    def test_s3_recursive_delete_is_critical(self):
        violations = scan_sigma("aws s3 rm s3://bucket/ --recursive")
        assert any("recursive_s3_delete" in v for v in violations)

    def test_rds_delete_is_critical(self):
        violations = scan_sigma("aws rds delete-db-instance --db-instance-identifier prod-db")
        assert any("rds_instance_delete" in v for v in violations)

    def test_force_delete_pod_is_high(self):
        violations = scan_sigma("kubectl delete pod xyz --force --grace-period=0")
        assert any("force_delete_pod" in v for v in violations)

    def test_safe_get_returns_no_critical(self):
        """kubectl get should not trigger critical rules."""
        violations = scan_sigma("kubectl get pods -n payments")
        assert not any("critical" in v for v in violations)

    def test_kubectl_logs_no_critical(self):
        violations = scan_sigma("kubectl logs payment-xyz -n payments --tail=100")
        assert not any("critical" in v for v in violations)

    def test_kubectl_describe_no_critical(self):
        violations = scan_sigma("kubectl describe pod payment-xyz -n payments")
        assert not any("critical" in v for v in violations)

    def test_sudo_su_is_high(self):
        violations = scan_sigma("sudo su - root")
        assert any("privilege_escalation" in v for v in violations)


class TestSandboxExecutor:
    @pytest.mark.asyncio
    async def test_dry_run_does_not_execute(self):
        result = await SandboxExecutor(dry_run=True).execute(
            action_id="a1", command="kubectl get pods -n payments", approved_status="approved"
        )
        assert result.success is True
        assert result.dry_run is True
        assert "[DRY RUN]" in result.stdout

    @pytest.mark.asyncio
    async def test_blocks_unapproved(self):
        result = await SandboxExecutor(dry_run=True).execute(
            action_id="a1", command="kubectl get pods", approved_status="pending"
        )
        assert result.success is False
        assert "Blocked" in result.stderr

    @pytest.mark.asyncio
    async def test_blocks_escalated(self):
        result = await SandboxExecutor(dry_run=True).execute(
            action_id="a1", command="kubectl delete all", approved_status="escalated"
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_blocks_critical_sigma_violation(self):
        result = await SandboxExecutor(dry_run=False).execute(
            action_id="a1",
            command="kubectl delete all -n production",
            approved_status="approved",
        )
        assert result.success is False
        assert "SigmaHQ" in result.stderr
        assert len(result.sigma_violations) > 0

    @pytest.mark.asyncio
    async def test_auto_executed_dry_runs(self):
        result = await SandboxExecutor(dry_run=True).execute(
            action_id="a1", command="kubectl get pods -n payments", approved_status="auto_executed"
        )
        assert result.success is True
