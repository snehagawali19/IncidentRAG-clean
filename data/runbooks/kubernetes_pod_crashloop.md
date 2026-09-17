---
runbook_id: rb-003
title: Kubernetes Pod CrashLoopBackOff
service: kubernetes-platform
environment: [prod, staging, dev]
severity_applicable: [sev1, sev2, sev3]
author: platform-reliability-team
created_at: 2024-02-10T00:00:00Z
last_updated_at: 2026-09-01T00:00:00Z
version: 4
tags: [kubernetes, crashloop, pod, container, restart, k8s]
related_runbook_ids: [rb-001, rb-002]
source_uri: file://data/runbooks/kubernetes_pod_crashloop.md
---

# Kubernetes Pod CrashLoopBackOff

This runbook covers the diagnosis and remediation of pods stuck in
`CrashLoopBackOff` state — one of the most common Kubernetes incidents.

A pod enters `CrashLoopBackOff` when its container exits with a non-zero
exit code and Kubernetes restarts it with exponential back-off (10 s, 20 s,
40 s … up to 5 min between restarts).

## Symptoms

| Alert | Description | Severity |
|---|---|---|
| `KubePodCrashLooping` | Pod restart count > 5 in 1 hour | SEV2 |
| `KubePodNotReady` | Pod not ready for > 15 minutes | SEV3 |
| `KubeDeploymentReplicasMismatch` | Available replicas < desired | SEV2 |

⚠️ If the crashing pod serves production traffic and the deployment has no
other healthy replicas, treat this as SEV1 and page the on-call immediately.

## Common Root Causes

| Root Cause | Key Indicators |
|---|---|
| Application startup failure | Exit code 1, config/DB errors in logs |
| Missing environment variable | `panic: required env var not set` |
| OOMKilled | Exit code 137, `OOMKilled` reason |
| Liveness probe failure | `Liveness probe failed` event, no crash log |
| Image pull error | `ImagePullBackOff` instead of `CrashLoopBackOff` |
| Config/Secret not mounted | `no such file or directory`, exit code 1 |

## Quick Triage

### 1. Identify the Failing Pod

```bash
# Find all CrashLoopBackOff pods across all namespaces
kubectl get pods --all-namespaces \
  | grep -E "CrashLoop|Error|OOMKilled"

# Detailed status for a specific deployment
kubectl get pods -n <NAMESPACE> -l app=<APP_NAME> -o wide
```

### 2. Describe the Pod for Events

```bash
kubectl describe pod -n <NAMESPACE> <POD_NAME>
```

Look for:
- **Last State** — exit code and reason (`OOMKilled`, `Error`, `Completed`)
- **Events** section — `Liveness probe failed`, `Back-off restarting failed container`
- **Conditions** — which readiness/liveness checks are failing

### 3. Fetch the Crash Logs

```bash
# Logs from the CURRENT (crashing) container
kubectl logs -n <NAMESPACE> <POD_NAME> --previous

# Tail logs with timestamps
kubectl logs -n <NAMESPACE> <POD_NAME> --previous --timestamps=true | tail -100
```

> **Note:** If the pod restarts before you can read logs, use `kubectl logs
> --previous` to fetch the terminated container's log buffer.

## Remediation by Root Cause

### Root Cause 1 — Application Startup Failure

**Indicators:** Exit code 1, database connection refused, config parse errors.

**Remediation Steps**

1. Check if dependent services are up:
   ```bash
   # Verify database connectivity from within the cluster
   kubectl run -it --rm debug --image=postgres:15-alpine \
     -n <NAMESPACE> --restart=Never -- \
     psql postgresql://<DB_HOST>:5432/<DB_NAME> -c "SELECT 1;"
   ```

2. Verify all required ConfigMaps and Secrets are present:
   ```bash
   kubectl get configmap,secret -n <NAMESPACE> | grep <APP_NAME>
   kubectl describe configmap <APP_CONFIG_NAME> -n <NAMESPACE>
   ```

3. Check for recent changes in the Deployment spec:
   ```bash
   kubectl rollout history deployment/<DEPLOYMENT_NAME> -n <NAMESPACE>
   kubectl rollout history deployment/<DEPLOYMENT_NAME> -n <NAMESPACE> \
     --revision=<PREV_REVISION>
   ```

4. If a recent deployment introduced the failure, roll back:
   ```bash
   kubectl rollout undo deployment/<DEPLOYMENT_NAME> -n <NAMESPACE>
   kubectl rollout status deployment/<DEPLOYMENT_NAME> -n <NAMESPACE> --timeout=120s
   ```

### Root Cause 2 — OOMKilled

**Indicators:** Exit code 137, `OOMKilled` in `kubectl describe`.

1. Confirm the kill reason:
   ```bash
   kubectl describe pod -n <NAMESPACE> <POD_NAME> \
   | grep -A3 "Last State"
   ```

2. Check current memory request/limit:
   ```bash
   kubectl get pod -n <NAMESPACE> <POD_NAME> \
     -o jsonpath='{.spec.containers[0].resources}' | jq .
   ```

3. Temporarily increase the memory limit to restore service:
   ```bash
   kubectl set resources deployment/<DEPLOYMENT_NAME> -n <NAMESPACE> \
     --limits=memory=<NEW_LIMIT>Gi --requests=memory=<REQUEST>Gi
   ```

4. Monitor pod memory after the restart:
   ```bash
   watch -n 10 kubectl top pod -n <NAMESPACE> -l app=<APP_NAME>
   ```

5. Follow the memory-specific runbook (rb-001) for deeper root cause analysis.

### Root Cause 3 — Liveness Probe Failure

**Indicators:** `Liveness probe failed` in events, container is running but
probe endpoint returns non-200 or times out.

1. Verify the probe path is reachable from inside the pod:
   ```bash
   kubectl exec -n <NAMESPACE> <POD_NAME> -- \
     curl -sv http://localhost:<PORT><PROBE_PATH>
   ```

2. Increase the probe failure threshold temporarily to stop the restart loop:
   ```bash
   kubectl patch deployment <DEPLOYMENT_NAME> -n <NAMESPACE> \
     --type='json' \
     -p='[{"op":"replace","path":"/spec/template/spec/containers/0/livenessProbe/failureThreshold","value":10}]'
   ```

3. Investigate why the health endpoint is failing — check application logs
   and metrics for the probe endpoint's dependencies.

4. Revert the failure threshold once the underlying issue is fixed:
   ```bash
   kubectl patch deployment <DEPLOYMENT_NAME> -n <NAMESPACE> \
     --type='json' \
     -p='[{"op":"replace","path":"/spec/template/spec/containers/0/livenessProbe/failureThreshold","value":3}]'
   ```

### Root Cause 4 — Missing ConfigMap or Secret

**Indicators:** `no such file or directory`, `key not found`, exit code 1 at startup.

1. List all volumes and their mount status:
   ```bash
   kubectl describe pod -n <NAMESPACE> <POD_NAME> | grep -A20 "Volumes:"
   ```

2. Check that the referenced ConfigMap/Secret exists:
   ```bash
   kubectl get configmap <CONFIG_NAME> -n <NAMESPACE>
   kubectl get secret <SECRET_NAME> -n <NAMESPACE>
   ```

3. If the resource is missing, create it from the correct source:
   ```bash
   # ConfigMap from a file
   kubectl create configmap <CONFIG_NAME> -n <NAMESPACE> \
     --from-file=config.yaml=./configs/production.yaml

   # Secret from literal values
   kubectl create secret generic <SECRET_NAME> -n <NAMESPACE> \
     --from-literal=DATABASE_URL="<VALUE>"
   ```

4. Force a pod restart to pick up the new resource:
   ```bash
   kubectl rollout restart deployment/<DEPLOYMENT_NAME> -n <NAMESPACE>
   ```

## Preventing Future CrashLoopBackOff

1. Set appropriate `resources.requests` and `resources.limits` for all containers.
2. Configure `startupProbe` for slow-starting applications so liveness probes
   do not kill pods before they finish initialising.
3. Use `preStop` lifecycle hooks and `terminationGracePeriodSeconds` to allow
   graceful shutdown.
4. Pin image tags — never use `latest` in production.

## Escalation Path

| Condition | Escalate to |
|---|---|
| All replicas crash-looping | platform-reliability on-call immediately |
| OOMKilled with no code fix | Memory runbook + backend-systems team |
| Cluster-wide pod failures | SRE incident commander |
| Data loss risk during crash | Engineering leadership |

## Post-Incident Actions

1. Add the pod name, exit code, and root cause to the post-mortem template.
2. If OOMKilled: permanently adjust memory limits in the Helm chart values.
3. If liveness probe: add a startup probe or adjust thresholds in values files.
4. Update synthetic canary tests to catch the failure mode in staging before
   it reaches production.
