+++
title = 'A Pragmatic Checklist for RAG Platforms'
date = 2024-01-28T08:15:00-08:00
description = 'Shipping retrieval-augmented generation at scale demands boring fundamentals. This is the checklist I run through with every team.'
tags = ['rag', 'platforms', 'evaluation']
categories = ['Applications']
excerpt = 'Before wiring together yet another embedding service, make sure the fundamentals—datasets, routing, evaluation, and observability—are in place.'
+++

Every company building copilots or knowledge assistants eventually converges on the same RAG platform requirements. I keep this checklist handy whenever we plan the next quarter of work; it helps anchor ambitious ideas to operational reality.

## 1. Version everything

- Snapshot documents, chunks, and embeddings together so rollbacks are trivial.
- Tag retrieval pipelines with semantic versions and store the metadata in the index.
- Promote new releases the same way you ship code: via tracked environments and change logs.

## 2. Plan for smart retrieval, not just nearest neighbors

Experiments consistently show that routing requests to specialized retrievers beats a single monolithic index. Invest in:

- Metadata-rich chunking (field boosts, doc importance, filters).
- Lightweight heuristic routing before LLM-powered routing—cheaper and shockingly effective.
- Hybrid search (sparse + dense) for product surfaces where recall matters more than latency.

## 3. Keep evaluations close to real users

- Build goldens directly from support tickets, CRM notes, or actual prompts.
- Evaluate both recall and answer quality—LLMs must cite documents that were actually retrieved.
- Wire evaluations into deployment gates. If accuracy regresses, the release blocks itself.

## 4. Instrumentation is non-negotiable

Operate the RAG platform like a multi-tenant service:

- Emit per-tenant latency, cost, and relevance metrics.
- Add request tracing from the entry point through retrieval, re-ranking, and generation.
- Store sampled request/response payloads with redaction so product teams can debug without tailing logs.

## 5. Build guardrails for drift

- Alert when document mix or query vocabulary drifts from what the index expects.
- Run automated `diff` jobs comparing the current ranked list with previous versions.
- Schedule load tests whenever corpus size or traffic doubles—retrievers often degrade non-linearly.

---

Fancy features like agentic orchestration or on-the-fly knowledge graphs are exciting, but they only compound complexity. Nail this checklist first and you'll have a reliable, diagnosable platform that can actually sustain those future experiments.*** End Patch
