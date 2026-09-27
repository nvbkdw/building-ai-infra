# FlashAttention kernel engineering: from the equations to LLM serving

Research snapshot: September 27, 2026. Suggested pace: 12 weeks of core work plus a 2-week capstone, at roughly 8–10 focused hours per week. The schedule is a proposed study plan, not a claim that a production FlashAttention implementation can be recreated in that time.

This course is designed for an engineer comfortable with PyTorch, CUDA fundamentals, GEMM, and introductory CuTe DSL. Use DGX Spark for the early experiments, an H100/H200 for Hopper-specific work, and a B200/B300 for datacenter Blackwell work. A100 access is useful for historical comparisons but optional.

The outcome is an educational attention implementation, a reproducible benchmark suite, and a technical report that connects GPU behavior to inference latency. Implement a deliberately small supported subset yourself; use the production libraries to study the remaining optimizations.

## 1. The central questions

By the end, you should be able to answer:

1. How can attention be computed without storing every attention score or probability?
2. Why can two implementations of the same equations have very different GPU performance?
3. Which bottleneck did each FlashAttention generation address?
4. Why does a strong prefill kernel sometimes perform poorly during decode?
5. How do paging, GQA, prefix sharing, CUDA graphs, and batching affect kernel design?
6. Which attention kernel actually executes inside a model-serving framework?
7. When does a faster attention kernel noticeably improve the whole system?

Use this loop throughout: **derive → predict → implement → check correctness → measure → explain the discrepancy**. Before an optimization, write down the resource you expect it to improve. Afterward, retain both successful and unsuccessful results.

## 2. A map of the names

| Name | What it represents | What to study |
|---|---|---|
| Scaled dot-product attention, or SDPA | A mathematical operation; also a PyTorch API | Shapes, masking, scaling, numerical behavior |
| FlashAttention | An IO-aware family of exact dense-attention algorithms and implementations | Tiling, online normalization, recomputation, hardware scheduling |
| `flash-attn`, FA3, and `flash-attn-4` | Software distributions and implementation paths | Installation, dispatch, supported shapes and devices |
| Flash-Decoding | Parallelizing inference attention across KV partitions, followed by a reduction | Occupancy when the query length is small |
| PagedAttention | Attention over KV data stored in separately allocated pages | Cache management, address translation, fragmentation and sharing |
| FlashInfer | An inference kernel library and generator with multiple implementations behind its APIs | Prefill/decode wrappers, planning, paged/ragged data, backend selection |
| FlexAttention | A programmable attention interface/compiler path | Custom score modifications, masks and block sparsity |
| vLLM, SGLang, TensorRT-LLM | Model execution and serving systems | Scheduling, cache ownership, attention metadata and backend integration |

These layers can compose. For example, an inference framework can use FlashInfer, which can select a FlashAttention-derived or TensorRT-LLM-derived kernel. A paged cache can be consumed by a FlashAttention-style kernel. Neither a Python package version nor a backend label uniquely identifies the paper algorithm being executed. [S5, S12, S15, S18–S20]

## 3. What changed across FlashAttention generations?

| Generation | Historical focus | Main idea to learn | Suggested reproduction |
|---|---|---|---|
| FA1, 2022 | GPU global-memory traffic and large intermediates | Stream tiles through on-chip memory; maintain online softmax statistics; recompute probabilities in backward | Derive and implement streaming attention |
| FA2, 2023 | Parallelism and non-matmul overhead, especially on A100 | Parallelize query tiles, improve warp work partitioning, reduce rescaling/normalization work | Implement a query-tile-owned forward kernel and tune its launch geometry |
| FA3, 2024 | Hopper's asynchronous hardware | Overlap TMA transfers, WGMMA matrix operations and softmax; study FP8 numerical techniques | Inspect and modify a Hopper pipeline; perform one controlled ablation |
| FA4, 2026 paper | Asymmetric resource scaling on datacenter Blackwell | Redesign the pipeline around TMEM and asynchronous MMA; relieve exponential and shared-memory bottlenecks | Reproduce a resource model and modify one CuTe DSL optimization |

FA1 removes quadratic *intermediate storage*, while dense attention still performs quadratic work when both sequence dimensions grow. Its backward pass deliberately recomputes information instead of saving the entire probability matrix. [S1]

FA2 reduces non-matmul work and assigns independent query rows to warps, avoiding the shared-memory reduction needed by the older within-block KV partition. Its query-length parallelism also exposes more independent thread blocks. This within-block partitioning issue is different from inference's cross-block split-KV reduction. [S2]

FA3 is the stage where execution timelines become as important as loop structure: the implementation overlaps memory transfer, matrix multiplication and softmax work. FP8 is a separate numerical study, including block quantization and incoherent processing, rather than a prerequisite for learning the BF16 pipeline. [S3, S4]

FA4's paper focuses on B200-class hardware. Its techniques include asynchronous MMA, tensor memory, software-assisted exponential evaluation, conditional rescaling, and a backward design using paired thread blocks. The current CuTe implementation contains paths for several architectures, so “running FA4” does not imply using every Blackwell-specific technique. [S6–S8]

The published speedups belong to specific shapes, precisions and software baselines. In particular, the FA4 author post reports that newer cuDNN versions incorporated related improvements. Include current cuDNN-backed attention in eligible comparisons. [S7]

## 4. Use the right hardware for each question

| Hardware | Compute capability | Best role in this course |
|---|---|---|
| DGX Spark / GB10 | 12.1, `sm_121` | Math, PyTorch references, compatible Triton/CuTe kernels, decode and serving experiments |
| A100 | 8.0, `sm_80` | FA1/FA2 performance archaeology and `mma.sync`/asynchronous-copy experiments |
| L40S | 8.9, `sm_89` | Optional comparison of a different compute-to-bandwidth balance |
| H100/H200 | 9.0, `sm_90` | Hopper WGMMA, TMA and FA3 |
| B200/GB200 | 10.0, `sm_100` | FA4's principal datacenter Blackwell mechanisms |
| B300/GB300 | 10.3, `sm_103` | Newer datacenter Blackwell experiments with a compatible software build |

Compute capabilities above come from NVIDIA's GPU table. [S9]

**Spark is useful, but it does not substitute for B200/B300 pipeline experiments.** The upstream `flash_fwd_sm120.py` path explicitly uses an SM80-style `mma.sync` implementation and adjusts shared-memory constraints. The dispatcher groups 12.x devices into that architecture family. This is distinct from the SM100 TMEM-based implementation. Verify your installed release and requested feature combination rather than assuming either universal support or universal lack of support. [S8]

Spark also shares LPDDR5x memory with its CPU, rather than providing a datacenter GPU's HBM environment. Its bandwidth, CPU activity, power and thermal conditions affect measurements. [S10]

Start each machine's experiment log with GPU name, compute capability, driver, CUDA, Python architecture, PyTorch, Triton, CUTLASS DSL, library version/commit, power configuration, and exact benchmark command. Record the runtime-selected backend. A successful import is insufficient evidence of kernel eligibility.

Rent an H100 only after your baseline and profiler scripts work. Prepare B200/B300 experiments before the rental begins. The purpose of cross-device measurements is to test a hardware hypothesis; do not interpret absolute latency differences as the effect of an algorithm alone.

## 5. The math you should derive first

For one head, let `Q` have shape `[Nq, D]`, `K` shape `[Nk, D]`, and `V` shape `[Nk, Dv]`:

\[
S=\frac{QK^T}{\sqrt D}+M,\qquad P=\operatorname{softmax}(S),\qquad O=PV.
\]

`M` adds zero for allowed positions and negative infinity for disallowed positions. Softmax operates across keys separately for each query. A query compares itself with keys and uses the resulting weights to average the values.

For a row of scores, subtracting any shared offset does not change its softmax. Choosing the maximum prevents exponentials from overflowing:

\[
p_j=\frac{e^{s_j-m}}{\sum_k e^{s_k-m}},\qquad m=\max_j s_j.
\]

For keys processed so far, maintain three quantities:

\[
m=\max_j s_j,\quad \ell=\sum_j e^{s_j-m},\quad
u=\sum_j e^{s_j-m}v_j.
\]

For a new score tile `s` and value tile `Vt`, update:

\[
m'=\max(m,\max(s)),\qquad a=e^{m-m'},\qquad p=e^{s-m'},
\]
\[
\ell'=a\ell+\sum p,\qquad u'=au+pV_t.
\]

After all tiles, `O = u / l`. The factor `a` converts previous sums to the new exponential scale. That invariant explains why earlier score tiles can be discarded. The online-normalizer paper is the short prerequisite reading; FA1 extends this idea to attention and GPU tiling. [S11, S1]

Initialize `m = -inf`, `l = 0`, `u = 0`. Handle an entirely masked tile explicitly: blindly computing `-inf - -inf` produces NaN. In this course, an entirely masked output row returns zero with log-sum-exp `-inf`; check each external API's contract.

For independent key partitions A and B, their states can be combined:

\[
m=\max(m_A,m_B),\quad a=e^{m_A-m},\quad b=e^{m_B-m},
\]
\[
\ell=a\ell_A+b\ell_B,\qquad u=au_A+bu_B.
\]

This merge is associative in exact arithmetic. Floating-point order can change rounding. It connects streaming attention to parallel decode and shared-prefix calculations; FlashInfer's recursive-attention tutorial develops this connection. [S13]

“Exact attention” means computing the dense-attention operation without introducing sparsity or a low-rank approximation. It does not promise bitwise equality across implementations, and optional low-precision paths introduce additional numerical considerations.

## 6. A small PyTorch reference to start with

This original teaching implementation handles one head, no dropout, and optionally causal attention. It returns FP64 output and natural-log log-sum-exp. The streaming function builds only tile-sized scores and masks, and is intentionally slow. Python loops and separate PyTorch operations do not reproduce a fused GPU kernel's performance.

`q_offset` is the absolute position of the first query when keys begin at position zero. For a suffix query over a full cache, use `q_offset = Nk - Nq`. Using zero instead gives upper-left causal alignment.

```python
import math
import torch


def prepare(q, k, v):
    assert q.ndim == k.ndim == v.ndim == 2
    assert q.shape[1] == k.shape[1]
    assert k.shape[0] == v.shape[0]
    assert q.shape[0] > 0 and k.shape[0] > 0
    assert q.device == k.device == v.device
    return tuple(x.to(torch.float64) for x in (q, k, v))


def finish(m, ell, u):
    denom = torch.where(ell > 0, ell, torch.ones_like(ell))
    out = u / denom[:, None]
    lse = torch.where(ell > 0, m + torch.log(denom),
                      torch.full_like(m, -torch.inf))
    return out, lse


def dense_attention(q, k, v, causal=False, q_offset=0):
    q, k, v = prepare(q, k, v)
    scores = (q @ k.T) / math.sqrt(q.shape[-1])
    if causal:
        qp = torch.arange(q.shape[0], device=q.device) + q_offset
        kp = torch.arange(k.shape[0], device=q.device)
        scores = scores.masked_fill(kp[None, :] > qp[:, None],
                                    -torch.inf)
    m = scores.amax(dim=-1)
    safe_m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    p = torch.exp(scores - safe_m[:, None])
    return finish(m, p.sum(dim=-1), p @ v)


def streaming_attention(q, k, v, block_q=32, block_k=64,
                        causal=False, q_offset=0):
    q, k, v = prepare(q, k, v)
    assert block_q > 0 and block_k > 0
    outs, lses = [], []
    for i in range(0, q.shape[0], block_q):
        qi = q[i:i + block_q]
        nr = qi.shape[0]
        m = torch.full((nr,), -torch.inf, device=q.device,
                       dtype=torch.float64)
        ell = torch.zeros_like(m)
        u = torch.zeros((nr, v.shape[1]), device=q.device,
                        dtype=torch.float64)
        qp = torch.arange(i, i + nr, device=q.device) + q_offset
        for j in range(0, k.shape[0], block_k):
            kt, vt = k[j:j + block_k], v[j:j + block_k]
            scores = (qi @ kt.T) / math.sqrt(q.shape[-1])
            if causal:
                kp = torch.arange(j, j + kt.shape[0], device=q.device)
                scores = scores.masked_fill(kp[None, :] > qp[:, None],
                                            -torch.inf)
            new_m = torch.maximum(m, scores.amax(dim=-1))
            safe_m = torch.where(torch.isfinite(new_m), new_m,
                                 torch.zeros_like(new_m))
            a = torch.exp(m - safe_m)
            p = torch.exp(scores - safe_m[:, None])
            ell = a * ell + p.sum(dim=-1)
            u = a[:, None] * u + p @ vt
            m = new_m
        out, lse = finish(m, ell, u)
        outs.append(out)
        lses.append(lse)
    return torch.cat(outs), torch.cat(lses)


torch.manual_seed(7)
q = torch.randn(5, 16)
k = torch.randn(13, 16)
v = torch.randn(13, 8)
ref, ref_lse = dense_attention(q, k, v, causal=True, q_offset=8)
out, lse = streaming_attention(q, k, v, block_q=3, block_k=4,
                                causal=True, q_offset=8)
torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-10)
torch.testing.assert_close(lse, ref_lse, atol=1e-10, rtol=1e-10)
```

The tight tolerance above is for these FP64 teaching calculations. It is not an appropriate general BF16/FP16 kernel acceptance criterion. This function's query-outer traversal is a useful bridge toward FA2, rather than a literal reproduction of the original FA1 loop schedule.

When comparing with PyTorch SDPA, its documented rectangular `is_causal=True` mask is upper-left aligned. FlashAttention changed its rectangular causal behavior to bottom-right alignment in version 2.1. For append/decode comparisons, construct the intended position-based mask explicitly and avoid relying on matching flag names. [S5, S16]

## 7. The weekly learning path

### Week 1 — Dense attention and a trustworthy baseline

**Question:** What does each tensor mean, and what actually grows quadratically?

Derive attention by hand for three tokens and a two-dimensional head. Explain the dot product, `1/sqrt(D)`, stable softmax and causal mask. Implement the dense reference, then extend it to `[B, H, N, D]`. Add GQA by mapping several query heads to each KV head; use explicit KV repetition only as a correctness reference.

Compare with PyTorch SDPA using `dropout_p=0.0`. Understand that SDPA is a dispatcher: the selected implementation depends on inputs and the installed build. Force the math backend for baseline comparisons, and separately record the automatically selected fast path. [S16]

**Experiment:** Sweep sequence length on small inputs. Calculate the score matrix's storage before allocating it. At `B=1`, `H=32`, `N=8192`, a single two-byte `N x N` matrix per head occupies 4 GiB; storing both scores and probabilities requires more.

**Deliverable:** `01_attention_math.ipynb`, a shape table, and tests for causal/noncausal attention and GQA mapping.

**Ready to move on:** You can distinguish attention's workspace from the persistent KV cache, and can explain a numerical discrepancy caused by a mask mismatch.

### Week 2 — Online softmax and streaming attention

**Question:** How can the output be exact without seeing all scores simultaneously?

Read the online-normalizer paper, then derive the `(m, ell, u)` invariant. Implement the streaming reference above. Instrument it to show how the scale and normalizer change after each tile. Implement a two-state merge and verify that different key partitions produce the same result within floating-point tolerance. [S11, S13]

**Experiment:** Test random lengths, odd dimensions, large score magnitudes, and different tile boundaries. Include a fully masked initial tile and a fully masked output row. Compare forward and reverse KV traversal for noncausal attention. State explicitly that causal masks refer to global token positions, independent of traversal order.

**Deliverable:** `02_online_softmax.ipynb`, with a proof of the recurrence and tests for tiled and partitioned attention.

**Ready to move on:** You can explain the rescaling factor without quoting the paper, and can identify which intermediates a GPU kernel must keep on-chip.

### Week 3 — GPU building blocks and measurement

**Question:** Can you separate memory, reduction and matrix-multiply costs?

Write a fused row-softmax kernel in Triton or CuTe DSL. Then write or reuse a small tiled GEMM whose layout you understand. Since you already have CuTe experience, use Triton as a concise reference and CuTe DSL as the main implementation language if that maintains momentum. NVIDIA's CUTLASS examples and Triton's official tutorials are the reference material. [S14, S24]

Learn coalesced accesses, register ownership, shared-memory layout, warp reductions, tensor-core operand layouts, and synchronization. An SM is a GPU processing unit; a CTA is a thread block; occupancy describes how much work can reside on an SM, rather than directly measuring useful throughput.

**Experiment:** Change one tile dimension at a time. Record latency, register count, shared memory per block, and spills. Benchmark with warmup and CUDA events; separately measure host-to-device launch overhead.

**Deliverable:** A small benchmark harness and one annotated Nsight Compute report.

**Ready to move on:** You can explain why the configuration with the highest occupancy need not be fastest.

### Week 4 — Your first fused FlashAttention-style kernel

**Question:** What changes when the streaming recurrence lives inside one GPU kernel?

Start with one head dimension, FP16 or BF16 inputs, FP32 accumulation, contiguous inputs, no dropout, and noncausal forward only. Assign a CTA to a query tile. Iterate over KV tiles, keeping row statistics and the output accumulator on-chip. Write output and optional log-sum-exp once.

Use the official Triton fused-attention tutorial to connect the math with a readable implementation. It implements an FA2-style algorithm and also contains newer hardware paths; isolate the basic recurrence before studying those extensions. [S14]

**Experiment:** Compare against the dense implementation and SDPA. Show that the GPU kernel avoids full score/probability tensors. Add causal masking and boundary predication, including nonmultiples of the tile sizes. If using base-2 exponentials, convert the score scale and document whether saved LSE is base-2 or natural-log.

**Deliverable:** `attention_fwd`, explicit supported-shape documentation, and output/LSE comparisons.

**Ready to move on:** You can account for every synchronization point and every global-memory write.

### Week 5 — FA2: ownership, parallelism and tuning

**Question:** Why can a memory-efficient kernel still leave the GPU underused?

Read FA2's algorithm and work-partitioning sections. Inspect the upstream CUDA forward kernel. Explain query-tile ownership at the CTA level and query-row ownership at the warp level. Keep these separate from KV splitting across independent CTAs during decode. [S2, S25]

**Experiment:** Sweep batch/head count, query length, `block_q`, `block_k`, warp count and pipeline depth. Try fewer batch-heads with a long query sequence, then many batch-heads with short sequences. Add GQA without physically duplicating KV storage. Measure whether a tile's larger reuse compensates for its additional registers and shared memory.

For causal attention, compare how work is distributed among early and late query tiles. Account for partially full boundary tiles when computing useful FLOPs.

**Deliverable:** A table of winning configurations across a modest shape grid, with profiler evidence explaining at least one reversal.

**Ready to move on:** You can state why one configuration loses on a particular shape, beyond “the benchmark was slower.”

### Week 6 — Backward and numerical behavior

**Question:** Why is recomputation beneficial, and where do errors originate?

For no dropout, derive:

\[
dV=P^T dO,\quad dP=dO V^T,
\]
\[
D_i=\sum_j O_{ij}dO_{ij},\quad dS=P\odot(dP-D[:,None]),
\]
\[
dQ=\frac{dS K}{\sqrt d},\qquad dK=\frac{dS^T Q}{\sqrt d}.
\]

Reconstruct probabilities tile by tile using saved row LSE. Implement a mathematical backward reference and compare with autograd. A fused GPU backward is an optional extension for an inference-focused learner. Study why accumulating `dQ`, `dK` or `dV` across tiles introduces reductions, and why GQA requires contributions from multiple query heads.

**Experiment:** Compare against FP64 or FP32 calculations performed on the *same already-quantized inputs*. Record maximum absolute error, RMS error, and relative error with a floor for near-zero values. Test more than Gaussian inputs. Add extreme score ranges and long sequences.

**Deliverable:** A numerical report that distinguishes input quantization, accumulation order, approximate exponentials and masking errors.

**Ready to move on:** You can explain why an isolated large relative error near zero is insufficient evidence of a broken model.

### Week 7 — Decode, append and split-KV

**Question:** What happens to query-parallel attention when `Nq = 1`?

Implement attention over a contiguous KV cache. Compare full prefill, chunked append, one-token decode, and a short multi-token verification query. Read the Flash-Decoding article and connect its partition reduction to your Week 2 merge. [S17]

**Experiment:** Split the KV sequence into 1, 2, 4, 8 and more partitions. Each produces a partial output plus normalization statistics; a second stage merges them. Sweep context length, batch size and GQA ratio. Find where additional parallelism outweighs extra launches, workspace and reduction overhead.

For one-token attention with ideal KV reuse, equal key/value dimensions, and `s` bytes per KV element:

\[
\text{work}\approx4BH_qN_kD,\quad
\text{KV bytes}\approx2BH_{kv}N_kDs,
\]
\[
\text{arithmetic intensity}\approx\frac{2(H_q/H_{kv})}{s}.
\]

This is a lower-bound traffic model, not a measured bandwidth claim. Cache behavior, repeated loads, quantization scales and metadata change actual traffic. The rest of a transformer can also be dominated by weight reads or GEMMs.

**Deliverable:** A decode phase diagram identifying where split-KV helps.

### Week 8 — Paged KV and FlashInfer

**Question:** How does a mathematical kernel consume real serving-system state?

Implement a tiny page allocator and page table in Python. For page size `P`, logical token `t` maps to `page_table[t // P]` and offset `t % P`. Randomly permute physical pages; the attention output should remain unchanged. Study the original PagedAttention paper for the memory-management motivation. [S22]

Next use FlashInfer's paged prefill and decode wrappers. Read `indptr`, page indices, final-page length, and `NHD` versus `HND` layouts. Treat planning and execution as separate costs. Reuse a plan only while its metadata and layout assumptions remain valid; a sequence growing across a page boundary may require updates. [S12, S23]

**Experiment:** Compare contiguous and paged KV at equal logical workloads. Test several supported page sizes, ragged request lengths, GQA ratios, and cold/warm prefix-cache conditions. Measure graph replay separately from uncaptured execution. Record the actual backend behind `auto`.

**Deliverable:** `08_flashinfer_paged_attention.ipynb`, including metadata construction, correctness checks and planning-versus-execution measurements.

**Ready to move on:** You can distinguish a framework's cache allocator from the attention kernel that reads its pages.

### Week 9 — Hopper and FA3

**Question:** Which independent operations can overlap, and which dependencies prevent overlap?

Use an H100/H200. First learn TMA, a hardware mechanism for tensor transfers, and WGMMA, a warpgroup matrix operation. Read NVIDIA's Hopper tuning guide and the FA3 paper/blog. Inspect the producer/consumer structure in the official Hopper mainloop. [S3, S4, S26, S28]

Draw a timeline of: loading a future KV tile, computing a score tile, performing softmax, and accumulating the weighted values. Annotate barriers with the data whose readiness they protect. Explain why instruction issue and operation completion are different events.

**Experiment:** Change one supported pipeline parameter or disable one overlap mechanism in an isolated kernel fork. Measure a small fixed shape set and inspect the profiler timeline. Study FP8 only after the BF16 path is understood; hold the reference inputs and output-error metric constant.

**Deliverable:** An annotated pipeline diagram and one reproducible ablation.

**Ready to move on:** You can point to the critical dependency rather than merely saying “asynchrony is faster.”

### Weeks 10–11 — Datacenter Blackwell and FA4

**Question:** What becomes expensive after tensor-core matrix multiplication becomes much faster?

Use B200/B300 with a compatible software stack. Learn TMEM, an on-chip tensor-core memory space, and the datacenter Blackwell MMA model. Read the FA4 paper's forward, backward and scheduling sections. Use the upstream CuTe files as a reading target, not as an assignment to rewrite every feature. [S6, S27]

**Week 10 experiment:** Build an estimate for four resources: global-memory traffic, matrix operations, exponential evaluation and shared-memory traffic. Compare it with measurements. Investigate tile size, asynchronous pipeline stages, output correction, or conditional rescaling. Change only one mechanism for a useful ablation.

**Week 11 experiment:** Inspect how intermediate tensors occupy TMEM and how the backward pass can use paired CTAs. Alternatively, take the inference-focused branch and study variable-length scheduling or GQA packing. Compare the supported FA4 path with eligible cuDNN and Triton implementations on the same GPU.

Do not expect a plain PyTorch rewrite to reproduce FA3/FA4's performance. Their contribution includes placement of intermediate state, instruction selection, barriers, and scheduling across hardware resources. A high-level mathematical implementation can validate an invariant but cannot establish that those mechanisms work.

**Deliverable:** A report titled “What is limiting attention on my Blackwell workload?”, supported by an ablation and an error analysis.

### Week 12 — Trace framework integration

**Question:** How do model-level tensors become arguments to the kernel you measured?

Use one model family throughout; Qwen3-8B is a reasonable continuation of your inference work, with Qwen3-32B as an optional scaling case. Read head counts, head dimensions and attention behavior from the actual checkpoint configuration. Begin with synthetic attention tensors before loading the whole model.

Trace: model attention call → Q/K/V layout → cache update → metadata preparation → backend selection → concrete kernel launch → output layout. Identify who owns RoPE, GQA mapping, cache append, masking, scales and scratch buffers. Fused implementations may place these responsibilities differently.

**Experiment:** Compare two supported backends in one framework while holding weights, dtype, cache dtype, model limits, workload and batching controls constant. Then repeat the same experiment in a second framework. Inspect logs or profiler kernel names to verify dispatch.

**Deliverable:** A source-navigation note and one verified backend comparison, including the reasons any configurations were ineligible.

### Weeks 13–14 — Capstone: connect kernel performance to serving

**Question:** Which attention optimization matters for a realistic workload?

Choose three request mixes: long-prompt/short-output, short-prompt/long-output, and repeated-prefix traffic with mixed request lengths. Establish baselines before changing a backend. Use fixed token lengths when comparing compute performance, then evaluate request-level behavior under a controlled arrival process.

Select one improvement supported by earlier evidence: tile tuning, split-KV selection, cache layout, planning reuse, or a prefill/decode backend combination. Integrate it into a minimal decoder or a development backend adapter. A focused integration is enough; avoid turning the project into a new serving framework.

Report kernel latency, attention-layer time, time to first token (TTFT), inter-token latency (ITL), output tokens/second, peak memory, and p50/p95/p99 where the sample size supports percentiles. Separate queueing delay from execution. Report correctness and quality checks as well as speed.

**Final deliverable:** A reproducible repository and a blog-style report containing the hypothesis, setup, controlled comparison, numerical evidence, profiler evidence, and where the optimization loses.

## 8. Framework integration reference

The controls below reflect the documentation reviewed on the research date. Pin a release and check its documentation before running them; options, defaults and feature support evolve.

| Framework/layer | Integration point | Exercise |
|---|---|---|
| PyTorch SDPA | `torch.nn.functional.scaled_dot_product_attention`; `torch.nn.attention.sdpa_kernel` for backend control | Compare math, eligible fused backends and automatic dispatch; inspect profiler output |
| PyTorch FlexAttention | `flex_attention`, compiled through PyTorch; an FA4 backend is available for supported configurations | Implement a sliding-window mask or score modifier; compare supported Triton and Flash paths |
| Hugging Face Transformers | `attn_implementation`, plus `AttentionInterface` and `AttentionMaskInterface` | Wrap/register the educational implementation for a small supported case; validate cache and mask handling |
| vLLM | Attention backend configuration and V1 backend implementations | Compare `FLASH_ATTN` and `FLASHINFER`; inspect the selected FA version and concrete FlashInfer implementation |
| SGLang | `--attention-backend`, `--prefill-attention-backend`, `--decode-attention-backend` | Test a compatible phase-specific combination and trace its metadata construction |
| TensorRT-LLM PyTorch backend | Attention backend and metadata classes; `attn_backend` configuration | Follow plan preparation, cache update and wrapper execution |
| NVIDIA Transformer Engine | Selection among eligible FlashAttention, cuDNN fused attention and unfused attention paths | Enable debug logs and inspect which training attention backend runs |

Sources: SDPA [S16], FlexAttention [S15], Transformers [S21], vLLM [S18], SGLang [S19], TensorRT-LLM [S20], Transformer Engine [S29].

Examples of explicit serving experiments, to run in separate processes on eligible hardware:

```bash
# vLLM: compatible FlashAttention-3 configuration on Hopper
vllm serve Qwen/Qwen3-8B \
  --attention-backend FLASH_ATTN \
  --attention-config.flash_attn_version=3

# vLLM: use the FlashInfer backend family
vllm serve Qwen/Qwen3-8B --attention-backend FLASHINFER

# SGLang: compare compatible Hopper prefill/decode paths
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --prefill-attention-backend fa3 \
  --decode-attention-backend flashinfer
```

These are experiment starting points, not a benchmark configuration: also fix precision, cache dtype, context limits, concurrency and other workload controls. The commands have not been executed as part of this research.

The current vLLM documentation exposes FA versions 2, 3 and 4 under the FlashAttention backend. Its FlashInfer backend can use different underlying implementations depending on architecture and configuration. SGLang also documents page-size and model-feature restrictions, including phase-specific selection. Consequently, compare eligible configurations and retain the actual dispatch in each result. [S18, S19]

For Transformers, registering a custom attention function is only part of the integration. Register the appropriate mask handling as well; its documentation notes that unregistered mask behavior can result in `attention_mask=None`. [S21]

For FlexAttention, the documented Flash backend is selected through `kernel_options={"BACKEND": "FLASH"}` in a compatible compiled setup. Treat score modifiers, block masks, captured tensors, forward/backward support and compilation costs as separate test dimensions. [S15]

## 9. Benchmark and correctness contract

Start small rather than taking the full Cartesian product of every setting:

| Dimension | Initial cases | Later extensions |
|---|---|---|
| Batch size | 1, 4, 16 | 64 or a memory-feasible concurrency sweep |
| Query length | 1, 16, 512, 2048 | Longer prefill and speculative verification shapes |
| KV length | 128, 2048, 8192 | 32768+ when memory and model support permit |
| Head dimensions | 64, 128 | 256, unequal value dimension, backend-specific limits |
| GQA ratio | 1, 4 | 8 and MQA-like cases |
| Precision | FP32/FP64 references; BF16 or FP16 kernel | FP8/FP4 only with separate quality evaluation |
| Sequence structure | Dense, causal, ragged | Sliding window, prefix sharing, supported sparse masks |
| Cache layout | Contiguous | Paged, supported page sizes, permuted physical pages |

For equal key/value head dimension and noncausal attention, count the two matrix products as approximately:

\[
\text{FLOPs}=4BH_qN_qN_kD.
\]

For causal or sparse cases, count valid query-key pairs; distinguish useful work from extra work executed in partial tiles. State whether a reported TFLOP/s figure includes only matrix products. Softmax, masking, layout conversion and cache management consume time even when excluded from that numerator.

For a first lower-bound estimate use `max(compute time, global-memory time)`. For advanced kernels, add exponential throughput, shared-memory traffic and synchronization/dependency constraints. A single global-memory roofline does not explain every attention bottleneck.

**Measure carefully:**

1. Compile and warm up before steady-state timing; report cold-start compilation separately.
2. Use CUDA events or a GPU-aware harness. Report host overhead separately if relevant.
3. Keep input dtype, causal alignment, dropout, precision policy and layouts equivalent.
4. Preallocate buffers for a kernel-only experiment. Also report layout conversion, planning, cache append and allocation costs in an application-level experiment.
5. Report eager and CUDA-graph timings separately. For tiny kernels, host enqueue gaps can still affect an event interval.
6. Use multiple timed runs and report dispersion. Do not use profiled latency as the unprofiled benchmark number.
7. Identify cache conditions. Repeatedly reading the same KV buffer can benchmark a warmer cache than the serving workload.
8. Keep background load and thermal conditions controlled, especially on Spark.
9. Inspect the actual kernel and backend. Avoid attributing another implementation's work to the requested backend name.

**Correctness cases:** non-square causal masks; an entirely masked row; partial tiles; noncontiguous tensors if advertised; GQA head mapping; extreme scores; long contexts; irregular final pages; cache append across a page boundary; different physical page assignments; different split counts; output and LSE conventions; and gradients if backward is supported.

Check FP16/BF16 kernels against a reference computed from the same quantized inputs. Choose tolerances from the operation, dtype and measured error distribution. Supplement elementwise tests with layer-output and teacher-forced model-logit comparisons. Free-running generated text can diverge after tiny early differences and is an inadequate numerical diagnostic by itself.

To connect kernel gains to model gains, use Amdahl's law. If attention takes fraction `f` of baseline time and improves by factor `s`, the ideal total speedup is:

\[
\text{speedup}=\frac{1}{(1-f)+f/s}.
\]

For example, doubling a component responsible for 30% of runtime yields only about 1.18× total speedup before secondary effects. Measure `f` for your workload; it is not a model constant.

## 10. Source-reading order and optional branches

For the shortest practical route, read **S11 → S1 → S2 → S14 → S17 → S12/S13/S23 → S3/S4 → S6/S7 → your chosen framework docs**. Alternate reading with implementation. Read production kernels after you have a small implementation that gives their complexity a purpose.

Useful source landmarks:

| Location | Reading objective |
|---|---|
| Triton `02-fused-softmax` | Reduction, fusion and launch basics |
| Triton `06-fused-attention` | Online recurrence, tile loops, masking and accumulation |
| FlashAttention `csrc/flash_attn/src/flash_fwd_kernel.h` | FA2-style CUDA forward structure |
| FlashAttention `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` | Hopper data movement, MMA and synchronization |
| FlashAttention `flash_attn/cute/interface.py` | Architecture/feature eligibility and dispatch |
| FlashAttention `flash_attn/cute/flash_fwd_sm120.py` | Spark/SM12x-family implementation differences |
| FlashAttention `flash_attn/cute/flash_fwd_sm100.py` and related files | Datacenter Blackwell forward pipeline; locate this in the pinned checkout |
| Framework attention backend and metadata classes | How runtime state becomes kernel inputs |

Pin a commit before annotating source. Upstream paths and feature matrices change; use repository search when a path moves.

After the capstone, choose one branch rather than expanding every topic:

- **Training:** fused backward, dropout RNG replay, deterministic reductions and distributed context parallelism.
- **Inference:** GQA reuse, prefix-aware attention, split-KV heuristics, quantized KV and multi-token verification.
- **Architecture:** compare one kernel on Ampere, Ada, Hopper and the two Blackwell families.
- **Other accelerators:** study ROCm/AMD attention paths and rebuild the resource model using their execution and memory hierarchy. CUDA instruction-level assumptions do not transfer directly.
- **Different attention models:** MLA or sparse/linear attention. First derive their changed tensor algebra; they are separate investigations from standard MHA/GQA.

## 11. Primary references

All links below are papers, project repositories, author explanations, or official framework/hardware documentation. The roadmap and experiments are proposed exercises; published performance results are not measurements performed for this guide.

- **S1 — FlashAttention paper (2022):** [Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135).
- **S2 — FlashAttention-2 paper (2023):** [Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/html/2307.08691v1). Focus on the algorithm and work-partitioning sections.
- **S3 — FlashAttention-3 paper (2024):** [Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608).
- **S4 — FA3 author explanation:** [Tri Dao's FA3 post](https://tridao.me/blog/2024/flash3/).
- **S5 — Official implementation and API behavior:** [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).
- **S6 — FlashAttention-4 paper (2026):** [Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/html/2603.05451v1).
- **S7 — FA4 author explanation:** [Tri Dao's FA4 post](https://tridao.me/blog/2026/flash4/).
- **S8 — Current CuTe implementation paths:** [SM120 forward](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm120.py), [interface/dispatch](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py), and [CuTe README](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/README.md).
- **S9 — GPU architectures:** [NVIDIA CUDA compute-capability table](https://developer.nvidia.com/cuda/gpus).
- **S10 — Spark hardware:** [NVIDIA DGX Spark Porting Guide: System Overview](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html).
- **S11 — Online softmax prerequisite:** [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867).
- **S12 — FlashInfer API:** [Attention kernels](https://docs.flashinfer.ai/api/attention.html).
- **S13 — Attention-state composition:** [FlashInfer recursive-attention tutorial](https://docs.flashinfer.ai/tutorials/recursive_attention.html).
- **S14 — Teaching kernels:** [Triton fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) and [Triton fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).
- **S15 — Programmable FA4 integration:** [FlexAttention + FlashAttention-4](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/).
- **S16 — PyTorch operation contract:** [SDPA documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html). Match the documentation to your installed version.
- **S17 — Decode parallelism:** [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/).
- **S18 — vLLM integration:** [Attention backend feature support and selection](https://docs.vllm.ai/en/latest/design/attention_backends/).
- **S19 — SGLang integration:** [Attention backends and phase-specific selection](https://docs.sglang.io/docs/advanced_features/attention_backend).
- **S20 — TensorRT-LLM integration:** [Attention implementation and metadata lifecycle](https://nvidia.github.io/TensorRT-LLM/features/attention.html).
- **S21 — Transformers integration:** [Attention backends and custom registration](https://huggingface.co/docs/transformers/main/attention_interface).
- **S22 — Paged KV motivation:** [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180).
- **S23 — FlashInfer data layout:** [KV-cache layout tutorial](https://docs.flashinfer.ai/tutorials/kv_layout.html).
- **S24 — CuTe/CUTLASS foundations:** [NVIDIA CUTLASS repository](https://github.com/NVIDIA/cutlass) and [official overview](https://docs.nvidia.com/cutlass/latest/overview.html).
- **S25 — FA2 source:** [CUDA forward kernel](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h).
- **S26 — FA3 source:** [Hopper forward mainloop](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp).
- **S27 — Blackwell resource model:** [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html).
- **S28 — Hopper resource model:** [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).
- **S29 — Training integration:** [Transformer Engine attention backend selection](https://docs.nvidia.com/deeplearning/transformer-engine/examples/attention/attention.html).
- **S30 — FlashInfer design paper:** [Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005).
- **S31 — FlashInfer implementation overview:** [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer).

## 12. Validation status of this guide

The embedded Python example was syntax-checked. The online recurrence and partition merge were checked against a dense NumPy reference on CPU, including rectangular causal alignment, odd tile sizes, large scores and fully masked rows. The PyTorch example itself was not executed because PyTorch was unavailable in the research environment. GPU kernels, serving commands and hardware-specific performance remain course experiments to run on the target machines.
