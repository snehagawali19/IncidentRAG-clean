---
runbook_id: rb-001
title: Payment Service Memory Issues
service: payment-service
environment: [prod, staging]
severity_applicable: [sev1, sev2]
author: payments-platform-team
created_at: 2024-01-15T00:00:00Z
last_updated_at: 2026-08-20T00:00:00Z
version: 3
tags: [memory, oom, heap, payment, jvm]
related_runbook_ids: [rb-002, rb-003]
source_uri: file://data/runbooks/payment_service_memory.md
---

# Payment Service Memory Issues

This runbook covers diagnosis and remediation of memory-related incidents in
the **payment-service**, including OOMKilled pod restarts, heap exhaustion,
and connection-pool leaks.

## Symptoms

You are likely reading this runbook because one of the following alerts fired:

| Alert name | Threshold | Severity |
|---|---|---|
| `PaymentServiceHighMemory` | RSS > 2 GB for 5 min | SEV2 |
| `PaymentServiceOOMKilled` | `OOMKilled` reason in pod events | SEV1 |
| `PaymentServiceHeapExhaustion` | JVM heap > 90 % for 10 min | SEV2 |

⚠️ If you see `OOMKilled` in production, treat this as SEV1 and page the
payments on-call immediately before attempting any manual remediation.

## Quick Triage

Run these diagnostic commands within the first 5 minutes to establish a
baseline.

### Confirm the Pod Restart

```bash
# Check pod restarts and OOMKilled events
kubectl get pods -n payments -l app=payment-service \
  --sort-by='.status.containerStatuses[0].restartCount'

kubectl describe pod -n payments -l app=payment-service \
  | grep -A5 "Last State:"
```

### Capture Current Memory Usage

```bash
# Per-container memory for all payment-service pods
kubectl top pods -n payments -l app=payment-service --containers

# Node-level memory pressure
kubectl describe node $(kubectl get pod -n payments -l app=payment-service \
  -o jsonpath='{.items[0].spec.nodeName}') | grep -A5 "Allocated resources"
```

### Pull JVM Heap Metrics

```bash
# JVM heap used/committed (via kubectl exec into a running pod)
POD=$(kubectl get pod -n payments -l app=payment-service \
  -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n payments "$POD" -- \
  curl -s http://localhost:8080/actuator/metrics/jvm.memory.used \
  | jq '.measurements[0].value / 1073741824 | . * 100 | round / 100'
```

## Root Cause Categories

### Category 1 — Connection Pool Leak

The most common root cause. Occurs when payment transactions fail
mid-flight and the JDBC connection is not returned to the pool.

**Diagnostic indicator:**

```bash
kubectl exec -n payments "$POD" -- \
  curl -s http://localhost:8080/actuator/metrics/hikaricp.connections.active \
  | jq '.measurements[0].value'
```

If active connections are at or near `maximumPoolSize` (default: 20) and
queries are queuing, you have a leak.

**Remediation Steps**

1. Identify the leaking transaction using slow-query logs:
   ```bash
   kubectl logs -n payments "$POD" --since=30m | grep "Connection is not available"
   ```

2. Force-recycle the connection pool without restarting the pod:
   ```bash
   kubectl exec -n payments "$POD" -- \
     curl -s -X POST http://localhost:8080/actuator/hikari/pool/reset
   ```

3. If the above does not clear the leak within 2 minutes, perform a
   rolling restart:
   ```bash
   kubectl rollout restart deployment/payment-service -n payments
   kubectl rollout status deployment/payment-service -n payments --timeout=120s
   ```

4. Monitor pool health for 10 minutes post-restart:
   ```bash
   watch -n 10 kubectl exec -n payments "$POD" -- \
     curl -s http://localhost:8080/actuator/metrics/hikaricp.connections.active
   ```

5. If the leak reappears, escalate to the payments engineering team and
   attach the thread dump:
   ```bash
   kubectl exec -n payments "$POD" -- jstack 1 > /tmp/payment_thread_dump.txt
   ```

### Category 2 — Heap Leak (Long-Running Objects)

Manifests as slowly rising old-gen heap usage with full GC pauses every
few hours. Usually caused by unbounded caches or listener accumulation.

**Remediation Steps**

1. Capture a heap dump for offline analysis:
   ```bash
   kubectl exec -n payments "$POD" -- \
     jcmd 1 GC.heap_dump /tmp/payment_heap.hprof

   kubectl cp payments/"$POD":/tmp/payment_heap.hprof ./payment_heap.hprof
   ```

2. Trigger a full GC to confirm whether memory is reclaimable:
   ```bash
   kubectl exec -n payments "$POD" -- jcmd 1 GC.run
   ```

3. If heap usage drops by > 30 % after GC, this is a live-data issue.
   Increase the pod memory limit temporarily while the root cause is fixed:
   ```bash
   kubectl set resources deployment/payment-service -n payments \
     --limits=memory=4Gi --requests=memory=3Gi
   ```

4. Open an engineering ticket with the heap dump attached and assign to
   the payments-platform team. Include the pod name, deployment version,
   and the time the alert fired.

## Rollback Procedure

If remediation steps make the situation worse, roll back to the previous
deployment revision:

```bash
kubectl rollout undo deployment/payment-service -n payments
kubectl rollout status deployment/payment-service -n payments --timeout=120s
```

## Escalation Path

| Condition | Escalate to |
|---|---|
| Pod OOMKilled 3+ times in 30 min | payments-platform Slack + PagerDuty |
| Heap dump analysis needed | backend-systems on-call |
| DB connection pool at 100 % | database-reliability team |
| No improvement after rolling restart | incident commander |

## Post-Incident Actions

After the incident is resolved:

1. File a post-mortem within 48 hours using the template in Confluence.
2. Update `maximumPoolSize` in `configs/production.yaml` if the root cause
   was pool exhaustion at normal load.
3. Add a Prometheus alert for `hikaricp.connections.active > 18` if one
   does not already exist.
4. Review whether a memory limit increase should be permanent or whether
   the code fix is sufficient.
