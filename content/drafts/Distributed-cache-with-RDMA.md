---
title: "RDMA Dev Notes"
date: 2026-01-03
tags: ["RDMA", "dev"]
author: "Ryan H."
description: "This blog post covers the RDMA dev notes."
summary: "This blog post covers the RDMA dev notes."
cover:
    image: "RDMA-dev-notes.png"
    alt: "RDMA Dev Notes"
    relative: true
---


## Introduction
Motivation: fast distributed cache to accelerate data loading during model training.

Why data loading is slow?

remote storage -> <network> -> local memory -> GPU memory
- network is the bottleneck

cache data on local NVMe disk to avoid network latency.
NVMe/SSD -> local memory -> GPU memory
- However, local memory has limited capacity.
- SSD bandwidth is still limited.

Build distributed in-memory cache connected with networks.
- Memory pool is shared, and scale to multiple nodes.
- However, TCP network bandwidth between nodes is still a bottleneck.

Solution: use RDMA to transfer data between nodes.
All nodes in the cluster connected with high bandwidth RDMA network, all nodes share the same memory pool.

TODO: example architecture diagram of H100





## Architecture of distributed cache
TODO: architecture diagram of distributed cache, i.e. curvine


## What is RDMA?

TODO: Why RDMA is faster than TCP?

TODO: RDMA network hardware

TODO: RDMA network programming model, i.e. pplx-garden TransferEngine architecture


## How to integrate RDMA with distributed cache
TODO: client to server data flow

### Server side implementation
- page cache table
- threading?

### Client side implementation
- RDMA buffer registration
- completion thread





Get basic inoformation about EFA NIC:

In RDMA, unlike traditional sockets, the receiver must have a RECV operation posted before the sender sends data. If no RECV is waiting when data arrives, the data is lost. This is critical for bidirectional communication.

```
Client                              Server
  |                                   |
  |  ----[CONNECT + address]---->     |  (RECV 1 waiting)
  |                                   |  (RECV 2 waiting)
  |  (RECV waiting for response)      |
  |  ----[DATA: "Hello"]-------->     |  (reverses, sends back)
  |                                   |
  |  <---[DATA: "olleH"]---------     |
  |                                   |
```




Check NIC link speed

``` bash
ethtool eth0 | grep Speed

```







## References
Harnessing 3200 Gbps Network, Lequn Chen, 2024: https://le.qun.ch/en/blog/2024/12/25/libfabric-efa-0-intro/


## Future Work
- QoS to avoid congestion with collective communication.
- Direct RDMA to GPU data transfer