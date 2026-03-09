--- 
title: "Distributed Inference System"
date: 2026-02-18
tags: ["distributed inference","system"]
author: "Ryan H."
description: "This blog post covers the distributed inference system."
summary: "This blog post covers the distributed inference system."
cover:
    image: "distributed-inference-system.png"
    alt: "Distributed Inference System"
    relative: true
---

# Introduction

At a high level, the inference system is composed of four components:

- **Request Router:** Routes incoming requests to the appropriate LLM worker nodes, balancing cache affinity and load.
- **Inference Worker:** Runs the LLM inference engine, handling both prefill and decode stages.
- **KV Cache Manager:** Manages KV cache storage and transfer across a hierarchical storage system (HBM, DRAM, NVMe, remote storage, etc.).
- **Control Plane:** Maintains cluster state and coordinates lifecycle events across all components.

![LLM Distributed Inference System](/static/llm-inference-architecture.png)

### Data Flow

1. The **Request Router** receives a request from the API server and tokenizes the text input into token IDs.

2. The router assigns the request to a **prefill worker** based on cache affinity and current load.

3. **Prefill stage:**
    - (a) The prefill worker computes the KV cache for the input tokens.
    - (b) The resulting KV cache is persisted to the **KV Cache Manager**, which stores it across the hierarchical storage system (HBM → DRAM → NVMe → remote storage).

4. Once prefill completes, the router forwards the request to a **decode worker**.

5. **Decode stage:**
    - (a) The decode worker generates output tokens auto-regressively and streams them back to the client.
    - (b) Newly produced KV cache entries are incrementally added to the **KV Cache Manager** for future reuse.

# Request Router:

The router is responsible for assigning each incoming request to the best worker node. It jointly optimizes two competing objectives:

- **KV cache affinity**: Route requests to the worker that already caches the most relevant token prefixes, minimizing redundant prefill computation and lowering time to first token.
- **Load balancing**: Spread requests across workers to prevent hotspots, keeping inter-token latency even and maximizing overall cluster throughput.

For high availability, the router runs as multiple replicas behind a load balancer. Each replica independently maintains its own view of the cluster state, so the failure of one replica does not affect the others.

![Routing layer](/static/llm-router.png)

### Routing as a Multi-Objective Optimization Problem

At its core, request routing is a **multi-objective optimization problem**. The router must jointly minimize two competing costs for every request:

1. **Prefill cost** — how many tokens must be computed from scratch (i.e., not covered by existing KV cache on the target worker).
2. **Queuing cost** — how loaded the target worker already is (active sequences, memory pressure).

These two objectives are inherently in tension. Routing to the worker with the best cache overlap minimizes prefill but may overload a single worker. Routing to the least-loaded worker balances throughput but throws away cache locality. A well-designed router exposes this as a tunable weight so operators can slide between latency-optimized and throughput-optimized regimes depending on the workload.

### Cache-Aware Routing: Knowing What Each Worker Has

To make cache-aware decisions, the router needs a global view of which token prefixes are cached on which workers. The standard approach is to maintain a **prefix index** (e.g., a radix tree) at the router that maps token-block sequences to worker identifiers. Workers publish cache events (block insertions and evictions) to the router, which keeps its index up to date.

When a request arrives, the router queries this index to compute the **overlap** between the request's input tokens and each worker's cached blocks. The worker with the longest prefix match can skip the most prefill computation.

**Key trade-off: accuracy vs. coordination overhead.** Maintaining a globally consistent prefix index requires continuous event streaming from every worker. As cluster size grows, this coordination traffic can itself become a bottleneck. One mitigation is to make cache tracking optional — falling back to load-only routing when the coordination cost exceeds the benefit of cache-aware placement.

### Load Estimation: Push vs. Pull

The router also needs to know how busy each worker is. There are two broad approaches:

- **Pull-based (metrics polling):** The router periodically scrapes load metrics from workers. Simple to implement, but the metrics are inherently stale — by the time the router reads them, the worker's state may have changed.
- **Push-based (dispatch tracking):** Each router replica tracks the requests *it* has dispatched and knows about load changes the instant they happen. To achieve a cluster-wide view when multiple router replicas exist, replicas periodically broadcast their dispatch counts to peers.

Push-based tracking is far more responsive, which matters when routing decisions are latency-sensitive. The trade-off is **approximate consistency**: each replica's view lags slightly behind reality, and brief double-counting can occur during broadcast propagation. In practice, this small inaccuracy is vastly preferable to the staleness of periodic polling.

### Deterministic vs. Stochastic Selection

Once the router scores every worker, it must choose one. The simplest policy is **greedy**: always pick the lowest-cost worker. This maximizes cache reuse but can create feedback loops where a popular worker attracts more requests, caches more data, and attracts even more requests — eventually becoming a hotspot.

Introducing **stochastic selection** (e.g., softmax sampling over worker scores with a temperature parameter) breaks this feedback loop. Higher temperature spreads requests more evenly at the expense of cache locality. This gives operators a second tuning knob alongside the cache-vs-load weight.

### Trade-off Summary

| Dimension | Favoring Cache Affinity | Favoring Load Balance |
| --- | --- | --- |
| **Time to first token (TTFT)** | Lower — reuses prefill computation | Higher — may recompute cached tokens |
| **Inter-token latency (ITL)** | Risk of hotspots on popular workers | More even across workers |
| **GPU utilization** | Uneven — some workers idle | More balanced across the cluster |
| **Best suited for** | Latency-sensitive, multi-turn conversations | Throughput-oriented batch workloads |

Together, the **cache-vs-load weight** and **selection temperature** form a two-dimensional tuning surface: the weight controls *what* the router optimizes for, and the temperature controls *how deterministically* it pursues that optimum.


# KV Cache Management:
Reusing historical KV caches has been proven to be critical for high-performance LLM serving systems. In long multi-turn conversations, and Agentic workflows, context often stretched past hundreds of K tokens around multiple turns per session. Without full KV cache retention, nearly every request required costly re-computation. 

The KV cache can be reused for multiple turns, reducing the number of full prefill computation and improving the performance.

i.g. SGLang introduced RadixAttention achieved state-of-the-art performance by reusing KV caches stored in GPU memory. However, the caching benefit is inevitably limited by a capacity bottleneck: as contexts grow longer and more clients engage in more rounds of conversations, the cache hit rate declines due to limited capacity.

Caching common techniques used in every corner of computer systems. The key idea is to keep a hot working set of data "close" and available for fast access.
Multiple levels of caching to achieve different trade-offs between latency, throughput, and capacity. LLM KV cache is a typical example of this.
Need a system to manage KV cache storage and transfer across layers of storage hierarchy.

- HBM: fast access, low capacity.
- DRAM: fast access, medium capacity.
- NVMe: slow access, high capacity.
- Remote storage: slow access, high capacity.


### How does KV cache stored (on local machine and across different storage layers)?

#### On GPU memory
on GPU memory, KV cache is stored as layer-first tensor, [2,layers, pages, heads, dim].
where:
- 2: Q and K
- layers: number of layers
- pages: number of pages
- heads: number of heads
- dim: dimension of the KV vector

This tensor layout on GPU is for compatibility with computation kernels
TODO: explain the computation kernels (only attention kernel requires KV cache loop up, in MLP and other layers, token does computation individually)

KV cache lookup and management can be stored in Radix Tree structure.

Radix Tree:
![Radix Tree](/static/radix_kv.jpg)


TODO: draw a diagram of Radix Tree structure and how it manages GPU KV page table.

#### On Host CPU memory
The Radix Tree: Extended for Three Tiers

Each `TreeNode` in the base `RadixCache` was originally:

```python
node.key    = RadixKey(token_ids)  # edge label
node.value  = torch.Tensor         # GPU KV cache indices
```

`HiRadixCache` extends this to:

```python
node.value            = torch.Tensor | None   # GPU indices (None if evicted from GPU)
node.host_value       = torch.Tensor | None   # CPU indices (None if not backed up)
node.hash_value       = List[str]             # SHA256 per page (storage addressing key)
node.host_ref_counter = int                   # protection from host eviction during async I/O
node.lock_ref         = int                   # protection from GPU eviction
node.hit_count        = int                   # access frequency, triggers write-through
```

##### Node Lifecycle

```
Created → value=GPU_indices, host_value=None
  ↓
  ↓ (write-through triggered by hit_count >= threshold)
  ↓
Backed up → value=GPU_indices, host_value=CPU_indices
  ↓
  ↓ (GPU eviction under memory pressure)
  ↓
GPU-evicted → value=None, host_value=CPU_indices  [NODE STAYS IN TREE]
  ↓
  ↓ (host eviction under memory pressure, but data persisted to storage)
  ↓
Fully evicted → node deleted from tree  [DATA LIVES ONLY IN STORAGE]
```

Key properties on `TreeNode`:
- `node.evicted` → `self.value is None` (GPU data gone)
- `node.backuped` → `self.host_value is not None` (CPU data present)

#### On external storage.
Uses a “page-first” layout for other layers to prioritize IO efficiency. This enables larger transfer sizes per transaction, and when combined with a zero-copy mechanism, achieves up to 2× higher throughput in typical deployments.

TBD: How to store radix tree on distributed layer of storage?

Maintaining Radix tree metadata in control plane, tree nodes points to the KV vectors on different storage layers.

Cons: 
- the control plane node are single point of failure, also QPS bottleneck
- tree can be big, too large to fit in memory

Can shard tree metadata across multiple partitions,

reference: https://lmsys.org/blog/2025-09-10-sglang-hicache/


### How does KV cache data is transferred across different storage layers?

When a cache miss happens on the GPU but hits the CPU memory, since the bandwidth between the two layers is typically high, we apply a layer-wise overlapping mechanism to load the data. This enables concurrent KV cache loading for layer N+ while layer N is executing, effectively hiding data transfer latency behind computation. 

When external storage is involved, significantly higher and less predictable latency is expected. To mitigate this, the cache controller opportunistically prefetches data from storage into host memory once a cache hit is detected at the storage tier. The prefetch strategy is configurable: it can operate in best-effort mode, terminate in-flight prefetching if a request becomes due for scheduling to minimize TTFT, or stage requests (stalled requests to wait for KV cache to be loaded) more aggressively to improve cache reuse and potentially raise overall throughput.

TODO: how to use GPU Direct Storage to fetch KV cache?


#### Cache Write policy:
The excerpt you selected from the [SGLang HiCache blog](https://lmsys.org/blog/2025-09-10-sglang-hicache/) describes how a system manages data movement between fast "tiers" (like GPU memory) and slower "tiers" (like CPU memory or Disk).

Think of these policies as different strategies for a librarian deciding which books to keep in the "Quick Reference" section versus the "Deep Storage" basement.

---

##### 1. Write-Through: The "Safety First" Approach

**"Provides the strongest caching benefits if bandwidth permits"**

In a **Write-Through** policy, every time new data (KV cache) is created in the fast GPU memory, it is **immediately** copied to the slower tiers (CPU or Disk).

* **Why it's the "Strongest":** If the GPU runs out of space and needs to delete something, a perfect copy already exists in the slower tier. There is zero delay in "backing up" because it was done instantly.
* **The Catch:** It requires massive **bandwidth**. If you are generating data faster than your connection to the disk can handle, the system will lag. It’s like trying to BCC your boss on every single Slack message you send—it's safe, but it's a lot of extra "typing."

##### 2. Write-Through-Selective: The "Smart Filter"

**"Leverages hit-count tracking to back up only hot spots"**

Instead of backing up everything, the system watches which data is being reused frequently (the "hot spots").

* **How it works:** It keeps a "hit count." If a specific piece of conversation or code is requested multiple times, the system marks it as "hot" and backs it up to the slower tier.
* **Why use it:** It drastically **reduces I/O load**. You aren't wasting bandwidth saving "one-off" data that will never be asked for again. This is the "Best of both worlds" approach for busy systems.

##### 3. Write-Back: The "Procrastinator’s Relief"

**"Can effectively mitigate the pressure... when tiers become capacity-constrained"**

In a **Write-Back** policy, the system only writes data to the slower tier when it **absolutely has to**—usually right before that data is about to be deleted from the fast tier to make room for something else.

* **Why it mitigates pressure:** It avoids constant background data movement. It only uses the "I/O pipes" when necessary.
* **The Risk:** If the system crashes before the "write-back" happens, that data is lost forever because it was never backed up. However, when your storage is nearly full or your bandwidth is choked, this "lazy" approach keeps the system running smoothly by prioritizing current tasks over background backups.

---

##### Summary Table

| Policy | Priority | Best Used When... |
| --- | --- | --- |
| **Write-Through** | Data Reliability | You have high bandwidth and want instant cache hits. |
| **Selective** | Efficiency | You want to save the most important data without clogging the system. |
| **Write-Back** | System Speed | Bandwidth is low or the system is under heavy load. |



### [Implementation details] What is the API of a KV cache manager? interface between KVCache and Inference Engine?

SGLang HiCache is how simple it is to plug in a new storage backend. Thanks to our clean, generic interfaces, integration requires implementing only three functionalities in your backend: get(key), exist(key), set(key, value). Everything else, including heavy-lifting tasks such as scheduling and synchronization coordination, is handled by the central cache controller.

Interface:
```python
class KVCacheManager:
    def get(self, key: str) -> bytes:
        pass
    def exist(self, key: str) -> bool:
        pass
    def set(self, key: str, value: bytes):
        pass    
```
TODO: how does this integrate with LLM scheduler? how does prefetch work?

KV cache prefetched when scheduler adds new request to queue, not at scheduling time. The goal is to start slow storage I/O as early as possible so data is ready by the time the request is actually scheduled.

walk through radix tree to find cached location for each token.
```
fill_ids:  [tok0, tok1, tok2, ..., tok99, tok100, ..., tok199]
           |---- GPU hit ----|
                             |-- CPU hit ----|
                                             |-- uncached --|
```


using the matched prefix chain, we can calculate the hash key to fetch KV cache from storage.


TBD: how does the implementation like? 
Integrate backends—Mooncake, 3FS, and NIXL, etc.
https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage

TBD: Input text or tokens? (before or after tokenizer?)
Tokenizer can run anywhere, so it is not part of the KV cache manager.
Assume request has already been tokenized before it is sent to the KV cache manager.

TODO: under the hood of Mooncake.

TODO: Special Cache optimization in LLMCache (Compressing, blending, etc.)

TODO: what's special about NIXL? How does GPU-direct storage work?

- https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/storage/nixl
- https://github.com/ai-dynamo/dynamo/blob/main/docs/pages/components/kvbm/README.md




# Inference Worker:
TBD: inference engine design and implementation.

TBD: prefill and decode stages.