---
title: "Data Jet"
description: "A high-performance dataloader implemented in Rust with a PyTorch-compatible interface. Drop-in replacement for PyTorch's DataLoader with faster data loading throughput."
summary: "High-performance Rust dataloader compatible with PyTorch"
tags: ["rust", "pytorch", "data-loading", "machine-learning"]
externalUrl: "https://github.com/nvbkdw/data-jet"
weight: 4
hidemeta: true
---

A high-performance dataloader implemented in Rust that serves as a drop-in replacement for PyTorch's DataLoader. Leverages Rust's concurrency and memory safety to deliver faster data loading throughput for ML training pipelines.

## Key Features

- **PyTorch-compatible API** — swap out `torch.utils.data.DataLoader` with minimal code changes
- **Rust performance** — multithreaded data loading and prefetching with zero-copy where possible
- **Seamless Python integration** — exposes a Python interface while keeping the hot path in Rust

[View on GitHub](https://github.com/nvbkdw/data-jet)
