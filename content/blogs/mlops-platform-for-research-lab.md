---
title: "MLOps Platform for Research Lab"
date: 2025-11-19
tags: ["mlops","platform"]
author: "Ryan H."
description: "This blog post covers the MLOps platform for research lab."
summary: "This blog post covers the MLOps platform for research lab."
cover:
    image: "mlops-platform-for-research-lab.png"
    alt: "MLOps Platform for Research Lab"
    relative: true
---

## Introduction

What's special about MLOps? Traditional software system discourage flexibility of random changes, it try to add guadrails everywehre, like unit tests, CI tests, integration tests, etc. It focus on repeated running the same code with failures (with different inputs, under various traffic, scale to many machines). on the contrary, ML is highly experimental, it should be take little efforts to make changes (model architecture, hyper parameters, etc). On experiment maybe only run once and discarded. MLOps platform need to provide an environment that embrace change as first-citizen, create no friction to new experiments, in the mean while, make the system robust and scalable, guardrail against failures. Strike the right balance between flexibility and robustness. depends on the system, the balance can be different, we need to make good trade-offs.

## Rule #1 - Flexibility is the King
if no critical reason, always bias toward providing flexibility to ML team. as ML research velocity is a key differetiator for the company.


## Rule #2 - One code base, run everywhere!
Run everywhere, experiment quickly on local code, scale to large datasets and models, and deploy to production, without the same code.
forster quick experimentation. 
biggest bottleneck if thoese environments are different.

requries:
* same runtime dependencies
* access to the same data on blob storage
* same interface to access data
* same access to GPU hardware


## Rule #3 - Tracking with configuration-driven workload
Configuration driven makes code reproducible and trackable.
Implement tracking system, like MLFLow to track everything

## Rule #3 - Rock-solid "infrastructure".
ML experiments is very fragile, it is easy to break, and requires many hand tuned hyper parameters. But there is certain invariant in the system that is always true, like executing a job, run jobs in parallel, allocating resource. Identify all these common "infrastracture" piece and treat them same as traditional software system, harden them to guard against failures.

## Rule #4 - Gradually raise up the abstraction
Build key abstractions, not ad-hoc scripts
Key abstractions:
Data, training, inference, evaluation, model, job (low-level)

Provides CLI or UI interface 

## Rule #5 - GPU efficiency
GPU is expensive, make it efficient has a huge impact on the bottom line. 



