---
runbook_id: rb-002
title: Database Connection Pool Exhaustion
service: database-proxy
environment: [prod, staging]
severity_applicable: [sev1, sev2]
author: database-reliability-team
created_at: 2024-03-01T00:00:00Z
last_updated_at: 2026-07-15T00:00:00Z
version: 2
tags: [database, postgres, pgbouncer, connection-pool, latency]
related_runbook_ids: [rb-001, rb-003]
source_uri: file://data/runbooks/database_connection_pool.md
---

# Database Connection Pool Exhaustion

This runbook covers diagnosing and remediating connection pool exhaustion in
the PostgreSQL stack, including PgBouncer saturation, Postgres `max_connections`
ceiling hits, and idle-in-transaction lock build-up.

## Symptoms

The following alerts indicate a connection pool problem:

| Alert | Threshold | Severity |
|---|---|---|
| `PgBouncerPoolSaturated` | pool_size ≥ max_client_conn for 3 min | SEV2 |
| `PostgresMaxConnectionsNear` | connections > 90 % max_connections | SEV1 |
| `DatabaseQueryLatencyP99High` | p99 > 2 s for 5 min | SEV2 |
| `IdleInTransactionSessions` | count > 5 for 2 min | SEV2 |

⚠️ Connection pool exhaustion causes cascading failures. If p99 latency is
rising across **multiple** services simultaneously, check this runbook first
before investigating individual services.

## Architecture Overview

```
Application pods  →  PgBouncer (port 5432)  →  PostgreSQL primary (port 5433)
                         ↓
                   PgBouncer (replica, port 5434)  →  PostgreSQL replica
```

PgBouncer operates in **transaction pooling** mode. Each connection is
returned to the pool after each transaction commit, allowing many application
connections to share a smaller Postgres connection set.

## Quick Triage

### Check PgBouncer Pool Status

```bash
# SSH into the pgbouncer host or exec into the pgbouncer pod
PGBOUNCER_POD=$(kubectl get pod -n database \
  -l app=pgbouncer -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n database "$PGBOUNCER_POD" -- \
  psql -p 6432 -U pgbouncer pgbouncer -c "SHOW POOLS;" 2>/dev/null
```

Columns to inspect: `cl_active`, `cl_waiting`, `sv_active`, `sv_idle`.
`cl_waiting > 0` means clients are queuing — you have pool exhaustion.

### Check PostgreSQL Active Connections

```bash
PGPOD=$(kubectl get pod -n database \
  -l app=postgres,role=primary -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n database "$PGPOD" -- psql -U postgres -c \
  "SELECT count(*), state, wait_event_type, wait_event
   FROM pg_stat_activity
   GROUP BY state, wait_event_type, wait_event
   ORDER BY count DESC;"
```

### Identify Long-Running Queries

```bash
kubectl exec -n database "$PGPOD" -- psql -U postgres -c \
  "SELECT pid, now() - pg_stat_activity.query_start AS duration, query, state
   FROM pg_stat_activity
   WHERE (now() - pg_stat_activity.query_start) > interval '30 seconds'
   ORDER BY duration DESC
   LIMIT 10;"
```

### Check for Idle-in-Transaction Sessions

```bash
kubectl exec -n database "$PGPOD" -- psql -U postgres -c \
  "SELECT pid, usename, application_name, state, query_start,
          now() - query_start AS idle_duration
   FROM pg_stat_activity
   WHERE state = 'idle in transaction'
   ORDER BY idle_duration DESC;"
```

## Remediation Steps

### Step 1 — Kill Idle-in-Transaction Sessions (Safe)

1. Identify PIDs of sessions idle-in-transaction for > 5 minutes:
   ```bash
   kubectl exec -n database "$PGPOD" -- psql -U postgres -c \
     "SELECT pid FROM pg_stat_activity
      WHERE state = 'idle in transaction'
        AND now() - query_start > interval '5 minutes';"
   ```

2. Terminate those sessions (safe — rolled back automatically):
   ```bash
   kubectl exec -n database "$PGPOD" -- psql -U postgres -c \
     "SELECT pg_terminate_backend(pid)
      FROM pg_stat_activity
      WHERE state = 'idle in transaction'
        AND now() - query_start > interval '5 minutes';"
   ```

3. Verify the pool starts draining within 60 seconds.

### Step 2 — Reload PgBouncer Configuration (No Downtime)

If pool parameters need adjustment, reload config without restarting:

1. Edit `/etc/pgbouncer/pgbouncer.ini` inside the pod (or update the
   ConfigMap and trigger a reload):
   ```bash
   kubectl exec -n database "$PGBOUNCER_POD" -- \
     psql -p 6432 -U pgbouncer pgbouncer -c "RELOAD;"
   ```

2. Confirm new pool_size is in effect:
   ```bash
   kubectl exec -n database "$PGBOUNCER_POD" -- \
     psql -p 6432 -U pgbouncer pgbouncer -c "SHOW CONFIG;" \
   | grep pool_size
   ```

### Step 3 — Emergency Connection Limit Increase

> **Use only if Steps 1–2 do not resolve within 10 minutes.**

Temporarily raise `max_connections` in PostgreSQL (requires restart):

1. Update the Postgres ConfigMap:
   ```bash
   kubectl edit configmap postgres-config -n database
   # Set max_connections = 300 (from default 200)
   ```

2. Restart PostgreSQL with a rolling strategy:
   ```bash
   kubectl rollout restart statefulset/postgres -n database
   kubectl rollout status statefulset/postgres -n database --timeout=300s
   ```

3. After the restart, update PgBouncer `server_pool_size` to match.

### Step 4 — Scale Application Connection Pools Down (Last Resort)

If the database cannot handle more connections, reduce client-side pool sizes:

1. Identify the highest-connection applications:
   ```bash
   kubectl exec -n database "$PGPOD" -- psql -U postgres -c \
     "SELECT application_name, count(*)
      FROM pg_stat_activity GROUP BY application_name ORDER BY count DESC;"
   ```

2. For the top offenders, patch the Deployment to reduce `HIKARI_MAXIMUM_POOL_SIZE`:
   ```bash
   kubectl set env deployment/payment-service -n payments \
     HIKARI_MAXIMUM_POOL_SIZE=10
   kubectl rollout status deployment/payment-service -n payments
   ```

## Monitoring Recovery

After applying remediation, watch these metrics for 15 minutes:

```bash
# PgBouncer client wait queue (should return to 0)
watch -n 5 'kubectl exec -n database "$PGBOUNCER_POD" -- \
  psql -p 6432 -U pgbouncer pgbouncer -c "SHOW POOLS;" 2>/dev/null | grep main'

# Postgres connection count
watch -n 5 'kubectl exec -n database "$PGPOD" -- psql -U postgres -c \
  "SELECT count(*) FROM pg_stat_activity;"'
```

## Escalation Matrix

| Condition | Action |
|---|---|
| `cl_waiting` > 50 for > 5 min | Page database-reliability on-call |
| Postgres restart required in business hours | Coordinate with incident commander |
| Data loss suspected | Immediately escalate to VP Engineering |
| PgBouncer crash loop | Check PgBouncer logs and escalate to infra team |

## Post-Incident Actions

1. Capture the final state of `pg_stat_activity` for the post-mortem.
2. Review `idle_in_transaction_session_timeout` Postgres setting — if it is
   not set, add it to prevent future accumulation.
3. Check whether application deployments in the 30 minutes before the
   incident introduced new query patterns.
4. Update connection pool sizing in `configs/production.yaml` if the root
   cause was misconfigured pool sizes.
