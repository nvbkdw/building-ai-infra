---
title: "KV-RDMA"
description: "A high-performance distributed key-value cache built in Rust that uses RDMA for zero-copy data transfers between client and server. Features a gRPC control plane, RDMA data plane with push model, threshold-based routing, and built-in benchmarking."
summary: "Distributed key-value cache with RDMA zero-copy data transfer"
tags: ["rust", "rdma", "distributed-systems"]
externalUrl: "https://github.com/nvbkdw/kv-rdma-poc"
weight: 1
hidemeta: true
---

A high-performance distributed key-value cache built in Rust that uses RDMA (Remote Direct Memory Access) for zero-copy data transfers between client and server.

## Architecture

The system separates the control plane (gRPC) from the data plane (RDMA). The control plane coordinates GET, PUT, and DELETE requests via gRPC and Protocol Buffers. The data plane uses RDMA Write operations to transfer values directly into pre-allocated client memory buffers, bypassing the CPU entirely.

## Key Design Decisions

- **Push model** — the server writes data directly to registered client buffers, eliminating CPU overhead during transfers
- **Threshold-based routing** — values under 64KB are transferred via gRPC; larger values use RDMA
- **Page-aligned buffers** — 4KB-aligned memory regions for optimal RDMA performance
- **Transport abstraction** — pluggable mock RDMA transport for development without specialized hardware, and libfabric for production RDMA NICs

## Features

TTL-based expiration, memory pool management for RDMA-registered buffers, concurrent client support, multi-NIC scalability, and built-in benchmarking with latency percentiles (p50, p95, p99).

**Tech stack**: Rust, gRPC, Protocol Buffers, libfabric

[View on GitHub](https://github.com/nvbkdw/kv-rdma-poc)
