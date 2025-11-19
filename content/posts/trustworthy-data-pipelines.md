+++
title = 'Designing Trustworthy Data Pipelines for AI Infra'
date = 2024-02-11T09:30:00-08:00
description = 'How I set up opinionated data layers, observability, and ownership so model teams can trust every batch.'
tags = ['data', 'observability', 'best-practices']
categories = ['Pipelines']
excerpt = 'Data issues are silent regressions for AI systems. Here is the rubric I use to keep pipelines honest without slowing teams down.'
+++

The most painful outages I've handled during the past year were silent regressions introduced in data pipelines. Models happily retrained on corrupted data, dashboards stayed green, and we only noticed something was wrong after customer tickets piled up. Since then, I've adopted a three-layer strategy that keeps pipelines honest without requiring a new platform team.

## 1. Declare the contract up front

Every dataset that feeds a model gets a machine-readable schema, ownership tag, and data-sensitivity label stored alongside the code that produces it. The contract lives in Git, can be imported into tests, and powers automated alerts whenever a breaking change shows up in CI.

```yaml
contract:
  owner: applied-ml
  dataset: rewards_v2
  schema:
    - name: prompt
      type: string
    - name: completion
      type: string
      max_length: 4096
    - name: reward
      type: float
      min: -1
      max: 1
```

By treating contracts as normal code, we get peer review, version history, and reproducibility for free.

## 2. Run cheap validation everywhere

Validation rulesets run in three places:

1. **CI** — Unit tests assert that transformations comply with the contract before merging to main.
2. **Batch jobs** — Lightweight profilers execute for each output shard and push metrics (row counts, uniques, distribution shifts) to the observability stack.
3. **Streaming checks** — Canary consumers read the next batch before promotion and diff it against the last good version using the same rules.

Most regressions are caught inside CI, the rest are blocked by the canary.

## 3. Instrument pipelines like user-facing services

Pipelines deserve the same SLO rigor we apply to production services. For each critical dataset we define:

- **Freshness SLO** – e.g., *90% of batches arrive within 15 minutes of schedule.*
- **Correctness SLO** – *99% of runs emit <0.5% invalid rows.*
- **Coverage SLO** – *Model-serving logs must observe ≥98% of new product flows within 24 hours.*

We emit metrics per SLO, feed them into alerting, and keep status dashboards next to feature dashboards. When a pipeline breaks an SLO, the owning team writes a mini postmortem just like any other incident.

---

None of these ideas are novel, but the discipline to apply them consistently is still rare. When founders ask how to de-risk their ML roadmap, this is the first playbook I hand over. Get the data foundation right and everything upstream—training, evaluation, deployment—becomes dramatically less chaotic.*** End Patch
