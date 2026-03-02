---
title: "AI Compute - Scaling Up"
date: 2026-02-27
tags: ["ai", "compute"]
author: "Ryan H."
description: "This blog post covers the AI compute scaling up."
summary: "This blog post covers the AI compute scaling up."
cover:
    image: "ai-compute-scaling-up.png"
    alt: "AI Compute Scaling Up"
    relative: true
---

# Introduction
Scaling up: add more CPU, GPU, memory on one server.

GPU architecuture

Multiple GPUs per server.


# Scaling Up with NVLink
GB300 example:

![GB300 NVL72 Architecture](/static/gb300-nvl72.png)

Each DGX GB300 rack is built with 18 compute trays and 9 NVLink switch trays.
Each NVLink switch tray is equipped with 2 NVLink switch chips and are
responsible for the full-mesh connectivity between all 72 GPUs within the same
DGX GB300 rack. Each B300 GPU features 18 NVL5 links and has one dedicated
NVL5 link connectivity to each one of the 18 switch chips, delivering a total
bandwidth of 1.8 TB/s low latency bandwidth.

