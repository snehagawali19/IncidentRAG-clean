"""Restricted Kubernetes command executor."""

from __future__ import annotations

import shlex

from incidentrag.execution.sandbox import ExecutionResult, SandboxExecutor


class KubectlExecutor:
    """Run only explicitly approved kubectl commands through the sandbox."""

    def __init__(self, sandbox: SandboxExecutor | None = None) -> None:
        self._sandbox = sandbox or SandboxExecutor(dry_run=True)

    async def execute(
        self,
        action_id: str,
        command: str,
        approved_status: str,
    ) -> ExecutionResult:
        try:
            executable = shlex.split(command, posix=False)[0].lower()
        except (ValueError, IndexError):
            executable = ""
        if executable not in {"kubectl", "kubectl.exe"}:
            return ExecutionResult(
                action_id=action_id,
                success=False,
                stderr="Blocked: KubectlExecutor accepts kubectl commands only",
                exit_code=1,
            )
        return await self._sandbox.execute(action_id, command, approved_status)
