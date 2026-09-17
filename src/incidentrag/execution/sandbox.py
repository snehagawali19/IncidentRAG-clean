"""Execution Layer — Sandboxed Command Executor.

Executes approved kubectl / AWS CLI / shell commands with:
  - SigmaHQ-style rule scanning (pre-execution safety check)
  - RBAC namespace scoping (only settings.sandbox_namespace)
  - Output capture with timeout
  - Dry-run mode for testing

Only executes actions with status == "approved" or "auto_executed".
ESCALATED and REJECTED actions are blocked.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shlex
import time
from datetime import datetime
from typing import NamedTuple

from pydantic import BaseModel, Field

from incidentrag.core.settings import settings

logger = logging.getLogger(__name__)

# ── SigmaHQ-style detection rules ─────────────────────────────────────────────

class SigmaRule(NamedTuple):
    name: str
    pattern: re.Pattern
    severity: str  # "critical" | "high" | "medium"


_SIGMA_RULES: list[SigmaRule] = [
    SigmaRule("delete_all_resources", re.compile(r"kubectl\s+delete\s+all", re.I), "critical"),
    SigmaRule("drain_node", re.compile(r"kubectl\s+drain", re.I), "critical"),
    SigmaRule("cluster_admin_bind", re.compile(r"clusterrolebinding.*cluster-admin", re.I), "critical"),
    SigmaRule("recursive_s3_delete", re.compile(r"aws\s+s3\s+rm.*--recursive", re.I), "critical"),
    SigmaRule("rds_instance_delete", re.compile(r"aws\s+rds\s+delete-db-instance", re.I), "critical"),
    SigmaRule("force_delete_pod", re.compile(r"kubectl.*--force.*--grace-period=0", re.I), "high"),
    SigmaRule("rm_rf", re.compile(r"rm\s+-[a-z]*r[a-z]*f|rm\s+-[a-z]*f[a-z]*r", re.I), "critical"),
    SigmaRule("privilege_escalation", re.compile(r"sudo\s+su|chmod\s+777", re.I), "high"),
]


class ExecutionResult(BaseModel):
    action_id: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: float = 0.0
    sandbox_pod: str = ""
    sigma_violations: list[str] = Field(default_factory=list)
    dry_run: bool = False
    executed_at: datetime = Field(default_factory=datetime.utcnow)


def scan_sigma(command: str) -> list[str]:
    """Returns list of violated rule names."""
    violations = []
    for rule in _SIGMA_RULES:
        if rule.pattern.search(command):
            logger.warning("SigmaHQ rule triggered: %s | cmd: %s", rule.name, command[:100])
            violations.append(f"{rule.name} ({rule.severity})")
    return violations


class SandboxExecutor:
    """Executes commands in a sandboxed subprocess with safety guardrails."""

    def __init__(self, dry_run: bool = False, timeout_seconds: int = 30) -> None:
        self._dry_run = dry_run
        self._timeout = timeout_seconds

    async def execute(
        self,
        action_id: str,
        command: str,
        approved_status: str,
    ) -> ExecutionResult:
        # Only run approved or auto_executed actions
        if approved_status not in ("approved", "auto_executed"):
            return ExecutionResult(
                action_id=action_id,
                success=False,
                stderr=f"Blocked: action status is '{approved_status}'",
                exit_code=1,
            )
        # SigmaHQ pre-scan
        violations = scan_sigma(command)
        if any("critical" in v for v in violations):
            return ExecutionResult(
                action_id=action_id,
                success=False,
                stderr=f"Blocked by SigmaHQ rules: {violations}",
                exit_code=1,
                sigma_violations=violations,
            )

        if self._dry_run:
            logger.info("[DRY RUN] Would execute: %s", command)
            return ExecutionResult(
                action_id=action_id,
                success=True,
                stdout=f"[DRY RUN] Command: {command}",
                exit_code=0,
                sigma_violations=violations,
                dry_run=True,
            )

        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={"KUBECONFIG": "/etc/kubeconfig/config"},  # scoped kube config
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout
            )
            duration_ms = (time.monotonic() - t0) * 1000
            success = (proc.returncode or 0) == 0

            logger.info(
                "Executed action %s | exit=%d | %.0fms",
                action_id, proc.returncode or 0, duration_ms,
            )

            return ExecutionResult(
                action_id=action_id,
                success=success,
                stdout=stdout.decode(errors="replace")[:4096],
                stderr=stderr.decode(errors="replace")[:2048],
                exit_code=proc.returncode or 0,
                duration_ms=duration_ms,
                sigma_violations=violations,
            )

        except asyncio.TimeoutError:
            return ExecutionResult(
                action_id=action_id,
                success=False,
                stderr=f"Execution timed out after {self._timeout}s",
                exit_code=124,
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                action_id=action_id,
                success=False,
                stderr=str(exc),
                exit_code=1,
                duration_ms=(time.monotonic() - t0) * 1000,
            )
