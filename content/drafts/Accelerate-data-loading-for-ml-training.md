---
title: "Accelerate Data Loading for ML Training"
date: 2026-02-09
tags: ["machine learning","data loading"]
author: "Ryan H."
description: "This blog post covers the accelerate data loading for machine learning training."
summary: "This blog post covers the accelerate data loading for machine learning training."
cover:
    image: "accelerate-data-loading-for-ml-training.png"
    alt: "Accelerate Data Loading for ML Training"
    relative: true
---

## Introduction

TODO: how does pytroch dataloader work?
- where does it fall short? (duplicate memory, python overhead, sequential batching)


TODO: Optimization, Rust implementation, avoid python overhead

TODO: Performance comparison with pytorch dataloader

TODO: preprocessing data as much as possible
- TODO: link to distributed cache with RDMA

TODO: Other optimizations, e.g. mmap data, async loading to GPU
