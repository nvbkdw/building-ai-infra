---
title: "LLM Inference Tensor Parallel"
date: 2026-01-19
tags: ["llm","inference","tensor parallel"]
author: "Ryan H."
description: "This blog post covers the LLM inference tensor parallel."
summary: "This blog post covers the LLM inference tensor parallel."
cover:
    image: "llm-inference-tensor-parallel.png"
    alt: "LLM Inference Tensor Parallel"
    relative: true
---

## Introduction

The Limits of Tensor ParallelismTensor Parallelism (TP), pioneered by Megatron-LM, partitions the computation along the hidden dimension $h$. Specifically, the weight matrices for the Query ($W_Q$), Key ($W_K$), and Value ($W_V$) projections are split column-wise across $N$ devices.
* Device $i$ computes a slice of the query, key, and value heads.
* Crucially, in standard TP inference, each device holds the entire sequence length $S$ for its subset of heads.
* While this reduces the size of the weight matrices per GPU, the memory required for the KV cache on each GPU is proportional to $S \times (h/N)$.
* Because the subsequent linear projection ($W_O$) requires an All-Reduce operation to sum the results from all heads, the communication pattern is efficient for weights but does not solve the redundancy of sequence storage if the KV cache is not partitioned across the sequence dimension.

