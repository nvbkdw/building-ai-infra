---
title: "LLM Inference Context Parallel"
date: 2026-01-19
tags: ["llm","inference","context parallel"]
author: "Ryan H."
description: "This blog post covers the LLM inference context parallel."
summary: "This blog post covers the LLM inference context parallel."
cover:
    image: "llm-inference-context-parallel.png"
    alt: "LLM Inference Context Parallel"
    relative: true
---

## Introduction

With tensor parallelism, the maximum sequence length is still bounded by the memory of a single device. Context Parallelism is defined by its ability to partition the input tensors along the sequence dimension, allowing the effective memory budget to scale linearly with the number of CP shards.

Traditional parallelism strategies, specifically Tensor Parallelism (TP) and Pipeline Parallelism (PP), were architected to address the storage of model *weights*, not the transient *activation states* of long sequences. Tensor Parallelism, the workhorse of distributed training, partitions the model's weight matrices across devices but typically replicates the KV cache on every worker to facilitate local attention computation. This redundancy becomes fatal in long-context regimes; utilizing eight GPUs for TP results in eight copies of a massive KV cache, effectively wasting seven-eighths of the cluster's aggregate memory on redundant data. Pipeline Parallelism avoids this duplication but introduces significant latency "bubbles" and is ill-suited for the low-latency requirements of interactive decoding.

**Context Parallelism (CP)** has emerged as the definitive architectural solution to this bottleneck. By partitioning the input tensors and the attention operation itself along the sequence dimension, CP enables the distribution of the KV cache without duplication. This paradigm shift allows inference engines to scale context lengths linearly with the number of computing devices, limited only by the aggregate memory of the cluster rather than the capacity of a single chip. Unlike Sequence Parallelism (SP)—which partitions only non-attention layers like LayerNorm and Dropout—Context Parallelism partitions the activations of all layers, including the critical attention mechanism.

This post provides an exhaustive technical analysis of Context Parallelism. We differentiate the distinct requirements of the **Prefill** (computation-bound) and **Decode** (memory-bound) phases, analyze the "Helix" architectural paradigm which decouples attention mechanisms, and perform a forensic examination of how leading inference engines—**vLLM** and **SGLang**—have implemented these concepts.

## **2\. Theoretical Framework of Distributed Attention**

To comprehend the mechanics of Context Parallelism, it is necessary to deconstruct the mathematical and operational foundations of the attention mechanism in a distributed setting. The standard self-attention operation for a sequence of length $S$, hidden dimension $h$, and head dimension $d$ is defined as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$
Where $Q, K, V \in \mathbb{R}^{S \times h}$. In a single-device setting, all matrices reside in local memory. The challenge arises when $S$ becomes large enough that the matrices—specifically $K$ and $V$, which must be cached—exceed single-device capacity.

### **2.1 The Limits of Tensor Parallelism**

Tensor Parallelism (TP), pioneered by Megatron-LM, partitions the computation along the hidden dimension $h$. Specifically, the weight matrices for the Query ($W\_Q$), Key ($W\_K$), and Value ($W\_V$) projections are split column-wise across $N$ devices.

* Device $i$ computes a slice of the query, key, and value heads.  
* Crucially, in standard TP inference, each device holds the *entire* sequence length $S$ for its subset of heads.  
* While this reduces the size of the weight matrices per GPU, the memory required for the KV cache on each GPU is proportional to $S \\times (h/N)$.  
* Because the subsequent linear projection ($W\_O$) requires an All-Reduce operation to sum the results from all heads, the communication pattern is efficient for weights but does not solve the redundancy of sequence storage if the KV cache is not partitioned across the sequence dimension.

With TP2, each rank keeps a copy of the sequence's KV cache (for its heads), meaning the maximum sequence length is still bounded by the memory of a single device. Context Parallelism is defined by its ability to partition the input tensors along the *sequence* dimension, allowing the effective memory budget to scale linearly with the number of CP shards.

### **2.2 Context Parallelism: The Sequence Partition**

In Context Parallelism, the global sequence $S$ is divided into $N\_{CP}$ chunks. If the global sequence is 100,000 tokens and $N\_{CP} = 4$, each GPU manages 25,000 tokens.

* GPU 0: Tokens $0$ to $24,999$  
* GPU 1: Tokens $25,000$ to $49,999$  
* ... and so on.

The immediate mathematical challenge is that the attention operation is non-local. To compute the output for a query token at position $40,000$ (residing on GPU 1), the model must attend to keys and values at positions $0-24,999$ (residing on GPU 0). This necessitates communication between devices during the attention calculation itself. This fundamental requirement creates two distinct classes of problems corresponding to the two phases of LLM inference: **Prefill** and **Decode**.

* **For Prefill:** The challenge is computing the massive $Q \\times K^T$ matrix where both $Q$ and $K$ are distributed. The primary change is that each card stores only a portion of the final KV cache.  
* **For Decode:** The challenge is that we have a single new query token, but the history it needs to attend to is scattered across all cards. A ring communication or broadcast mechanism is needed to obtain the final result.

## **3\. Prefill Context Parallelism: Algorithms and Architectures**

The Prefill phase (also known as the "Prompt Processing" phase) is computation-bound. The engine receives a long prompt (e.g., 100k tokens) and must compute the initial KV cache and the first token output. Because all tokens in the prompt are available simultaneously, this phase offers significant opportunities for parallelization, but also imposes massive communication bandwidth requirements.

There are two primary algorithmic approaches to implementing Prefill Context Parallelism: **Ring Attention** and **DeepSpeed Ulysses** (All-to-All).

### **3.1 Ring Attention**

Ring Attention is a memory-efficient approach that avoids gathering the full sequence on any single GPU. It is conceptually similar to the Ring All-Reduce algorithm but applied to the attention computation.

#### **3.1.1 The Algorithm**

1. **Initialization:** The input sequence is partitioned across $P$ GPUs. Each GPU $i$ holds a query block $Q\_i$, a key block $K\_i$, and a value block $V\_i$.  
2. **Local Computation:** In the first step, each GPU computes attention using its local $Q\_i$ and local $K\_i, V\_i$.$$O\_i \leftarrow \text{Attention}(Q\_i, K\_i, V\_i)$$  
3. **The Ring Pass:**  
   * Each GPU sends its $K\_i, V\_i$ block to its neighbor (Rank $i+1$) and receives a block from its predecessor (Rank $i-1$).  
   * Let $K\_{recv}, V\_{recv}$ be the blocks received.  
4. **Accumulation:**  
   * The GPU computes attention between its static local $Q\_i$ and the traveling $K\_{recv}, V\_{recv}$.  
   * $$O\_i \leftarrow \text{Update}(O\_i, \text{Attention}(Q\_i, K\_{recv}, V\_{recv}))$$  
   * The "Update" step involves carefully merging softmax statistics (Log-Sum-Exp) to ensure numerical stability.  
5. **Iteration:** This process repeats $P-1$ times until every KV block has visited every GPU.

#### **3.1.2 Advantages and Constraints**

The primary advantage of Ring Attention is that it maintains a constant memory footprint. At no point does a GPU need to hold more than its local shard plus one incoming buffer block. This allows for processing infinitely long sequences, limited only by the aggregate memory of the cluster.

However, Ring Attention introduces a strict serialization dependency. The computation at step $t$ cannot proceed until the data from step $t-1$ has arrived (although computation and communication can be overlapped). It works best when the chunk size is large enough that the computation time of the attention block ($O(Chunk^2)$) hides the latency of the data transfer. vLLM's implementation specifically keeps the sequence dimension sharded throughout the computation and circulates Key/Value blocks through a ring topology.

### **3.2 DeepSpeed Ulysses (All-to-All)**

DeepSpeed Ulysses takes a different approach that leverages the high bisection bandwidth of modern GPU clusters (e.g., NVLink) to parallelize the attention heads rather than the sequence loop.

#### **3.2.1 The Algorithm**

Ulysses relies on the mathematical property that attention heads are independent.

1. **Initial State:** Input is partitioned by Sequence. GPU $i$ holds all heads for a subset of tokens.  
2. **Transpose 1 (All-to-All):** The system performs a distributed transpose operation. It redistributes the data such that the partitioning switches from **Sequence** to **Heads**.  
   * After this step, GPU $i$ holds the *entire* sequence for a subset of heads.  
3. **Local Attention:** Because the GPU now has the full sequence for its assigned heads, it can run a standard, highly optimized attention kernel (like FlashAttention-2 or cuDNN) without any modification.  
4. **Transpose 2 (All-to-All):** The output is transposed back. The partitioning switches from **Heads** back to **Sequence**.

#### **3.2.2 Advantages and Constraints**

Ulysses is typically faster than Ring Attention on clusters with high-bandwidth interconnects (like NVSwitch) because the All-to-All operation can utilize the full injection bandwidth of the network. Furthermore, it does not require writing custom "ring-aware" attention kernels; it can leverage standard, vendor-optimized kernels.

However, Ulysses has a hard constraint: the number of attention heads must be divisible by the degree of parallelism ($N\_{CP}$). If a model has 32 heads and one wishes to use 64 GPUs for context parallelism, Ulysses is not directly applicable without further splitting heads, whereas Ring Attention has no such constraint.

## **4\. Decode Context Parallelism: Overcoming the Latency Barrier**

While Prefill CP focuses on throughput (processing massive prompt chunks), Decode CP focuses on latency. In the autoregressive decoding phase, the model generates one token at a time. The system is memory-bound, as it must load the massive KV cache from HBM to compute attention for a single query token.

### **4.1 The Memory-Bound Challenge**

In a naive implementation without CP, the KV cache for the entire context must reside on the GPU generating the token. For a 1M token context, this is impossible. With CP, the KV cache is sharded.

* GPU 0 holds KV for tokens 0-250k.  
* GPU 1 holds KV for tokens 250k-500k.  
* ...  
* The **Query Token** (the most recently generated token) exists on one GPU (or is replicated).

To generate the next token, the Query must attend to *all* 1M tokens. We cannot move the 1M tokens to the Query GPU (that would be terabytes of transfer per second). Instead, we must move the Query to the data.

### **4.2 The Pass-Q Strategy (Broadcast Query)**

The industry standard for Decode CP is the "Pass-Q" or "Broadcast Query" strategy.

1. **Broadcast Q:** The single Query token vector $Q$ is broadcast to all CP ranks.  
2. **Local Attention:** Each rank computes the attention scores between the global $Q$ and its local shard of $K$ and $V$.  
   * Rank 0 computes attention scores for tokens 0-250k.  
   * Rank 1 computes attention scores for tokens 250k-500k.  
3. **Reduction:** The system performs a reduction operation to combine the partial outputs.  
   * This is not a simple sum. It requires combining the Log-Sum-Exp (LSE) values from the softmax operation on each rank to correctly normalize the attention weights globally.

### **4.3 Helix Parallelism: The Decoupling Paradigm**

The most advanced realization of Decode CP is formally described in the **Helix Parallelism** framework (arXiv:2507.07120). Helix identifies a critical inefficiency: Tensor Parallelism (TP) is good for compute-heavy layers (Feed-Forward Networks \- FFNs), while Context Parallelism (CP) is necessary for memory-heavy layers (Attention).

Helix proposes a hybrid execution schedule:

1. **Attention Phase (CP Mode):** The GPUs operate in Context Parallel mode. The KV cache is sharded by sequence. The Query is broadcast, attention is computed locally, and results are reduced.  
2. **FFN Phase (TP Mode):** The GPUs switch roles. The same GPUs now operate in Tensor Parallel mode to compute the MLP layers.  
   * This requires that the weights of the FFN are sharded (TP) while the KV cache is sharded (CP).

#### **4.3.1 The HOP-B Algorithm**

The primary bottleneck in Decode CP is the latency of the communication (the All-Reduce of attention outputs). Even though the data is small, the "ping" time across the network adds up over thousands of tokens, increasing **Token-To-Token Latency (TTL)**.

Helix introduces the **HOP-B (Helix Overlap Pipeline \- Batch-wise)** algorithm to solve this.

* **Concept:** It hides the communication latency of the Attention phase behind the computation of the FFN phase.  
* **Mechanism:** In a continuous batching system serving multiple requests, Helix overlaps the communication of Request A's attention reduction with the computation of Request B's FFN layers.  
* Compared to conventional parallelism, Helix reduces TTL by up to 1.5x.

This algorithm is critical for making CP viable for interactive applications, as it decouples the network latency from the critical path of generation.

## **5\. Architectural Case Study: vLLM**

vLLM (Virtual Large Language Model) is a high-throughput inference engine distinguished by its **PagedAttention** memory management system, which treats KV cache memory like virtual memory pages in an operating system. Its implementation of Context Parallelism has evolved from experimental support to a robust feature set.

### **5.1 Architecture and Configuration**

vLLM implements a centralized **Scheduler** and distributed **Workers**. The integration of CP required significant refactoring of the BlockManager, which tracks which GPU holds which page of the KV cache. In a CP setup, the BlockManager must be aware that a single logical sequence is physically fragmented across multiple workers.

Configuration of CP in vLLM (v0.10.2+) is achieved via command-line arguments:

* \--tensor-parallel-size (TP): Defines the number of GPUs splitting the weights.  
* \--decode-context-parallel-size (DCP): Defines the number of CP shards.  
* Note: vLLM typically nests these groups. If you have 8 GPUs and set TP=4 and DCP=2, the system effectively uses 8 GPUs. However, behaviors imply a nuanced layout where the resource groups are tightly coupled (tp\_size needs to be divisible by dcp\_size).

### **5.2 Implementation of Decode CP (DCP)**

vLLM's implementation of DCP (PR \#23734) utilizes the **Pass-Q** strategy.

* **KV Layout:** The KV cache is partitioned along the sequence dimension using an **Interleave Strategy**. Instead of simple contiguous blocks (First half on GPU 0, Second half on GPU 1), interleaving (Block 0 on GPU 0, Block 1 on GPU 1, Block 2 on GPU 0...) helps balance the load, especially when using PagedAttention where blocks are allocated dynamically.  
* **Execution Flow:**  
  1. **Broadcast:** The query token is broadcast to all DCP ranks.  
  2. **Local Look-up:** Each rank consults its local Page Table to find the physical blocks corresponding to the sequence.  
  3. **Compute:** The FLASH\_ATTN\_MLA backend or standard attention kernels compute the local attention scores.  
  4. **Reduction:** A custom reduction kernel merges the partial results, essentially performing an All-Reduce function.

### **5.3 Code-Level Analysis: ring\_attention.py**

vLLM maintains ring\_attention.py which likely implements the Prefill CP logic. In vLLM, the preference for prefill is often **Ulysses** (if head counts allow) due to its speed on NVLink-connected H100s. However, Ring Attention is maintained as a fallback for scenarios where the number of heads is not divisible by the CP size or for specific hardware topologies where ring bandwidth is more consistent than all-to-all performance.

The integration with **Ray** is also pivotal. worker\_use\_ray \= True is standard for these distributed setups. Ray actors manage the individual GPU processes, and the AsyncLLMEngine serves as the orchestrator, dispatching the scheduling metadata to the Ray workers which then execute the distributed kernels.

## **6\. Architectural Case Study: SGLang**

SGLang (Structured Generation Language) is an inference engine that prioritizes high performance and complex control flows (e.g., agentic loops). It is often faster than vLLM in specific benchmarks due to its aggressive optimization of the **RadixAttention** mechanism (automatic prefix caching) and the use of **FlashInfer** kernels.

### **6.1 RadixAttention in a Sharded World**

SGLang's defining feature is the Radix Tree, a data structure that manages the KV cache as a tree of tokens to enable instant reuse of shared prefixes (e.g., system prompts). Implementing CP in SGLang presented a unique challenge: How to shard a Radix Tree?

The approach involves partitioning the physical storage while maintaining a logical central scheduler:

* The **logical** Radix Tree remains in the central scheduler. It tracks which tokens exist in the cache.  
* The **physical** storage is partitioned. When a request is scheduled, the scheduler maps the logical nodes of the tree to physical GPU ranks.  
* Crucially, "kv cache for one context sequence is split to dcp ranks."

### **6.2 The "All-Gather Q" vs "All-Gather KV" Divergence**

SGLang's implementation details highlight a distinct implementation choice compared to vLLM's flexible Ring/Ulysses approach.

* **Prefill:** SGLang tends to use an **All-Gather KV** approach for prefill. "In our implementation, an all\_gather of kv cache is done for prefill stage." This means for the prefill chunk, SGLang gathers the relevant KV data to the local GPU to utilize its highly optimized FlashInfer kernels, then discards the remote data, keeping only the local shard in HBM. This favors kernel speed over communication bandwidth efficiency for prefill.  
* **Decode:** SGLang utilizes the **All-Gather Q** (Pass-Q) strategy. "An all\_gather of q is done for decode stage to compute attn\_output."

### **6.3 Performance and Benchmarks**

Benchmark data comparing CP implementations shows significant advantages for CP-enabled systems (like "SLOPE" and SGLang's CP mode) over pure Tensor Parallelism for long contexts.

* **TTFT (Time To First Token):** At 100k context length, SGLang with TP=8 (without CP) struggles with TTFTs approaching 9 seconds due to the overhead of processing the massive prompt with replicated KV.  
* **Scalability:** CP implementations maintain TTFT under 2 seconds even at 100k+ lengths.  
* "SLOPE provides a practical solution for long-context LLM inference," outperforming baseline SGLang TP=8 in throughput at the same TTFT constraints.

## **7\. Implementation Guide: Engineering Context Parallelism**

For engineers looking to implement Context Parallelism in a custom inference engine, here are the actionable steps distilled from the architectural patterns observed in vLLM and SGLang.

### **Step 1: Topology and Process Groups**

The first step is defining the distributed environment. You typically need orthogonal process groups for Tensor Parallelism (TP) and Context Parallelism (CP).

* **Concept:** If you have 8 GPUs ($Rank\_{0}$ to $Rank\_{7}$) and want $TP=2, CP=4$:  
  * **TP Groups:** $[0,1], [2,3], [4,5], [6,7]$. (GPUs within these groups split the weights).  
  * **CP Groups:** $[0,2,4,6]$ and $[1,3,5,7]$. (GPUs within these groups share the same model weights but split the sequence).  
* **Implementation:** Use torch.distributed.new\_group to create these communicators. You must ensure that every tensor operation uses the correct group handle.

### **Step 2: The KV Partitioning Strategy**

You must modify your memory allocator.

* **Naive Allocation:** Allocate pages linearly.  
* **CP Allocation:** Use a **Round-Robin** or **Interleaved** strategy.  
  * Logical Block 0 $\rightarrow$ CP Rank 0  
  * Logical Block 1 $\rightarrow$ CP Rank 1  
  * ...  
  * Logical Block $N$ $\rightarrow$ CP Rank $(N \mod CP\_Size)$  
* **Why?** This ensures that as the sequence grows, the memory pressure is evenly distributed across all CP ranks. If you filled Rank 0 before moving to Rank 1, Rank 0 would OOM while Rank 3 is empty.

### **Step 3: Implementing the Attention Kernel**

You cannot use off-the-shelf flash\_attn\_func for the decode phase because it doesn't handle distributed reductions. You must wrap it.

**Pseudo-Code for Decode Step (Pass-Q):**

```python
def context_parallel_decode(query, kv_cache_manager, cp_group):  
    # 1. Broadcast Query to all CP ranks  
    # query shape: [batch, 1, heads, dim]  
    torch.distributed.broadcast(query, src=root_rank, group=cp_group)

    # 2. Fetch Local KV Blocks  
    # Each rank retrieves only the blocks it owns  
    local_k, local_v = kv_cache_manager.get_local_blocks()

    # 3. Compute Local Attention (Result + LSE)  
    # We need the Log-Sum-Exp (LSE) for correct reduction  
    local_out, local_lse = flash_attn_interface(query, local_k, local_v, return_softmax_lse=True)

    # 4. Global Reduction (The Tricky Part)  
    # You cannot just sum 'local\_out'. You must weight it by LSE.  
    # Global LSE \= LogSumExp(all local LSEs)  
      
    # Gather all LSEs and Outs (or use a custom reduction kernel)  
    all_lses = all_gather(local_lse, group=cp_group)  
    global_lse = torch.logsumexp(all_lses, dim=0)

    # Re-weight local output  
    # Weight = exp(local_lse - global_lse)  
    weighted_out = local_out * torch.exp(local_lse - global_lse)

    # Sum weighted outputs  
    final_out = all_reduce(weighted_out, op=Sum, group=cp_group)

    return final_out
```

*Note:* This logic is simplified. In production, this reduction is often fused into a custom CUDA kernel to avoid the overhead of allocating intermediate tensors for all_lses.

### **Step 4: Metadata Synchronization**

The most common source of bugs in CP implementation is metadata desynchronization. The scheduler (usually on CPU) maintains the "Ground Truth" of the sequence length and block table.

* You must ensure that when the scheduler says "Sequence A has length 1024", *every* CP rank receives this length.  
* However, the *local* kernel on Rank 1 might only see 256 tokens in its local cache. The kernel must be aware that it is computing a "partial" attention for a global sequence of 1024, which affects positional embeddings (RoPE).  
* **Critical Detail:** When applying Rotary Positional Embeddings (RoPE), you must use the **global** position indices, not the local buffer indices. If Rank 1 holds tokens 256-512, it must apply RoPE for positions 256-512, not 0-256.

## **8\. Conclusion**

Context Parallelism has transitioned from a theoretical novelty to an infrastructural imperative. As models like DeepSeek-V3 and Llama 4 push the boundaries of context into the millions of tokens, the "Memory Wall" that once constrained inference has been dismantled by the architectural elegance of sharding the sequence dimension.

Our analysis of vLLM and SGLang reveals a converging ecosystem. Both engines have adopted the **Pass-Q (Broadcast Query)** strategy for decoding, recognizing it as the optimal solution for memory-bound generation. For prefill, a divergence exists between the flexible **Ring/Ulysses** approach of vLLM and the aggressive **FlashInfer-optimized** approach of SGLang.

Looking forward, the integration of **Helix Parallelism** and its **HOP-B** algorithm represents the next frontier. By decoupling the parallel strategies of Attention (CP) and FFN (TP) and overlapping communication with computation, we are approaching a regime where the cost of infinite context is purely economic (hardware cost), no longer bound by the latency or memory limits of the silicon itself.

## References
https://gemini.google.com/app/b31681db58e769db