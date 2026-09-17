---
runbook_id: rb-argocd-repo-server
title: Argo CD Repository Server Failures
service: argocd-repo-server
environment: [production, staging]
severity_applicable: [sev1, sev2, sev3]
author: platform-team
created_at: 2026-01-01T00:00:00Z
last_updated_at: 2026-09-01T00:00:00Z
version: 1
tags: [argocd, repo-server, git, tls, manifest-generation]
related_runbook_ids: [rb-kubernetes-pod-crashloop]
source_uri: file://data/runbooks/argocd_repo_server.md
---

# Argo CD Repository Server Failures

## Symptoms

Applications remain OutOfSync or report manifest generation errors. The
`argocd-repo-server` logs may show Git process crashes, repository initialization
failures, TLS verification errors, authentication failures, or request timeouts.

## Investigation steps

1. Check the repository server pods and recent restarts.
2. Read the previous container logs when a pod has restarted.
3. Confirm repository connectivity and inspect the application conditions.
4. Compare the repository credentials and TLS settings with the affected repository.

```bash
kubectl -n argocd get pods -l app.kubernetes.io/name=argocd-repo-server
kubectl -n argocd logs deploy/argocd-repo-server --since=30m
kubectl -n argocd get application APP_NAME -o yaml
argocd repo get REPOSITORY_URL
```

## Git process crashes

A Git crash can leave repository initialization incomplete and prevent manifest
generation. Inspect pod memory limits, node pressure, repository size, and concurrent
manifest requests. Preserve the crash logs before restarting a pod.

| Signal | Interpretation | Next check |
| --- | --- | --- |
| Exit code 137 | Memory pressure or OOM kill | Pod limits and node memory |
| Exit code 128 | Git or repository failure | Git stderr and credentials |
| Repeated restarts | Persistent process failure | Previous container logs |

## TLS failures

Validate the repository certificate chain from the same network as the repository server.
Prefer installing the correct CA certificate. Disabling TLS verification should be a
temporary, explicitly approved diagnostic step because it weakens repository authenticity.

## Safe remediation

1. Correct invalid repository credentials or install the required CA certificate.
2. Increase repository server memory only when logs or metrics show memory pressure.
3. Restart one unhealthy repository server pod after collecting logs.
4. Re-run application refresh and confirm that manifest generation succeeds.

```bash
kubectl -n argocd rollout restart deployment/argocd-repo-server
kubectl -n argocd rollout status deployment/argocd-repo-server
argocd app get APP_NAME --refresh
```

Do not rotate credentials, disable TLS verification, or restart all replicas without human
approval and a rollback plan.
