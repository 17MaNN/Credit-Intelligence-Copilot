# Credit Intelligence Copilot

Agentic RAG system for lending/collections ops. Public datasets only — no real financial data.

## Structure
```
services/   one folder per microservice (risk-model, doc-cv, nlp-classifier, rag, agent)
lib/        shared code: schemas, auth, logging — imported by every service, never duplicated
infra/      Dockerfiles + k8s manifests
eval/       golden-set evaluation harness, run in CI
```

## Principles
- Every service is single-responsibility and stateless
- Shared contracts live in `lib/`, not copy-pasted
- Secrets via env vars only, injected by K8s Secrets in deployment
- Base Docker image in `infra/base.Dockerfile`; each service adds only its own layer

## Phases
See project plan — built incrementally, one service per phase.
