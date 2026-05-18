# Learning Path: Senior AI Systems Engineer (Anthropic / OpenAI)

## Context

**Goal:** Get hired at Anthropic or OpenAI on a team that owns system + kernel performance of LLM workloads on large GPU clusters (Anthropic Performance Engineering, Inference, Infra Efficiency, AIRE; OpenAI Applied Engineering / Inference / Scaling Systems).

**Current state:**

- Backend / distributed systems engineer with light ML background
- Builds ML infrastructure at Noetik: ~10 nodes / ~80 H100s + L40S, training a <1B-param multimodal model (image encoder + transformer decoder) on medical images and gene sequencing data
- 10–15 hrs/week available over 12–18 months (~600–900 hrs total)
- Local consumer GPU at home; production-grade cluster access at work
- Running technical blog `building-ai-infra` — portfolio surface

**What hiring managers actually screen for (synthesized from the JDs + interview transcripts in `./job-description-and-interviews/`):**

The bar is *production battle scars at scale*, not theoretical knowledge. Specifically:

1. Can reason about why an all-reduce is slow from **NVLink fabric → NCCL → PyTorch distributed** end-to-end
2. Has personally diagnosed degraded IB links, NCCL/tensor-core SM contention, ECC errors, NVLink degradation **under pressure**
3. Can write and integrate **custom GPU kernels** (Triton or CUDA), validate with roofline / Nsight, measure quantitative gains (MFU, latency, throughput)
4. Has built or operated **RDMA-backed** distributed components (KV cache, gradient comms) and understands EFA / InfiniBand / GPUDirect
5. Understands **inference internals**: PagedAttention, continuous batching, prefill/decode disaggregation, prefix caching, parallelism strategies
6. Owns SLOs that include hardware failure modes; has led cross-layer incident response
7. Comfortable in Python + Rust/C++; can debug across Python → C++ → kernel layers

---

## How This Plan Works (Theory → Practice)

This is a two-track curriculum that runs in parallel each phase:

- **Track 1 — Theory foundation:** the canonical reading (papers, books, courses, source code) that gives you the mental models hiring managers test for. Front-loaded heavily in Phase 0–2.
- **Track 2 — Noetik practice:** every phase's hands-on project is grafted onto your actual work. You're not building toy clusters — you're shipping production-relevant improvements to Noetik's training and inference stack. These become your interview stories and your blog post material.

The order matters: **read enough theory in each phase that you know what experiment to run**, then run that experiment on Noetik hardware. Theory without practice produces candidates who can't debug; practice without theory produces candidates who can't generalize. You need both, and your Noetik cluster is a rare advantage most applicants don't have.

---

## The Noetik Workload (your practical anchor)

You're not training a frontier 70B LLM. You're training a **<1B param multimodal model on 80 H100s**. That regime has *different* dominant bottlenecks than what most ML systems papers focus on — and that's actually a more interesting and more current set of problems for an Anthropic/OpenAI portfolio (multimodal inference is everywhere now).

**Architecture:**

- Image encoder (likely ViT-style) on high-resolution medical images
- Transformer decoder on long-context gene sequencing data
- Multimodal fusion between the two

**Characteristic bottlenecks for this regime (you should validate these in Week 1):**

| Bottleneck | Why it matters here | What it isn't |
| --- | --- | --- |
| **Gradient all-reduce overhead** | Small model on many GPUs → comm/compute ratio is high. Small-message NCCL regime. | Not bandwidth-bound large-message all-reduce. |
| **Dataloader I/O** | High-resolution medical images = massive bytes/step. Storage→CPU→GPU pipeline likely the rate limiter. | Not GPU-compute-bound. |
| **Long-context activation memory** | Gene sequences can be very long → activation memory dominates. Sequence parallelism + activation checkpointing matter more than tensor parallelism. | Not parameter-memory-bound (model is small). |
| **Multimodal prefill stalls (inference)** | Image encoder forward = one giant matmul-heavy "prefill". Decoder autoregression waits on it. | Not the classic LLM TTFT problem. |
| **PCIe bandwidth + thermal headroom** | Feeding huge image batches stresses PCIe and cooling more than typical LLM training. | Not pure SM utilization problem. |

**Phase 0 baseline (do this Week 1, before anything else):** Profile one training step of your current Noetik model with PyTorch Profiler + `nsys`. Capture wall-clock time spent in: image encoder forward, decoder forward, optimizer step, all-reduce, dataloader. The largest bucket tells you which phase to lean into hardest. *If your dataloader is 40% of step time, no kernel optimization will save you* — that's the answer the rest of the plan is built to find.

---

## Target Role Profile (the rubric to optimize against)

| Dimension | "Pass the bar" signal |
| --- | --- |
| **GPU kernels** | Written a non-trivial Triton kernel (e.g., fused softmax, long-context attention variant); can profile with Nsight Compute and reason about occupancy, memory bandwidth, tensor-core utilization |
| **Distributed comms** | Understands NCCL ring/tree/double-binary tree algorithms; has tuned NCCL env vars (`NCCL_ALGO`, `NCCL_PROTO`, `NCCL_NSOCKS_PERTHREAD`) on a real cluster; knows why GPUDirect RDMA matters |
| **Distributed training** | Can explain TP/PP/DP/SP/EP/CP/FSDP from first principles; has measured MFU on real hardware; has debugged a hang or NaN at scale |
| **Inference** | Can sketch vLLM scheduler from memory; understands PagedAttention block table; has implemented or tuned continuous batching, prefix cache, chunked prefill |
| **Reliability** | Has built DCGM-based health checks, set up GPU-aware SLOs, written a real post-mortem |
| **Portfolio** | 6–10 deep blog posts; ≥1 merged PR to vLLM / SGLang / NCCL / Megatron / PyTorch / DeepSpeed |

---

## Phase Structure (12–18 months, 10–15 hrs/week)

Each phase: **Theory (reading) → Noetik practice (work-integrated project) → Output → Validation gate.**

---

### Phase 0 — Foundation + Baseline (Months 1–4, ~120 hrs)

**Theory anchor:** Stanford [CS336 Spring 2025 — Language Modeling from Scratch](https://cs336.stanford.edu/spring2025/). Lectures + all 5 assignments (A1 tokenizer/transformer/AdamW, A2 FlashAttn-2 Triton kernel + DDP/FSDP profiling, A3 scaling laws, A4 data, A5 alignment).

**Parallel theory reading (~3 hrs/week):**

- ezyang's [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/) + [PyTorch Dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/) posts
- [GPU MODE lectures](https://github.com/gpu-mode/lectures) 1–5 (profiling, basic CUDA, Triton intro)

**Noetik practice — Week 1 baseline (do this before anything else):**

Profile one training step of the current Noetik multimodal model. Use `torch.profiler` + `nsys`. Produce a step-level breakdown:

- Image encoder forward / backward
- Decoder forward / backward
- Optimizer step
- All-reduce / gradient sync
- Dataloader wait time

This becomes the **diagnostic baseline** for the rest of the plan — every later phase's project will target whichever bucket dominates.

**Noetik practice — Phase 0 project:** Take the Triton FlashAttn-2 kernel you wrote in CS336 A2 and port it to your home RTX 4090, an L40S at Noetik, and an H100 at Noetik. Compare achieved vs. peak FLOPS/bandwidth on each; write up the difference (Ada vs. Ampere/Hopper SM, tensor-core generations, L2/SMEM sizes).

**Outputs:**

- Blog post #1 — *"Where does our 80-H100 multimodal training step actually spend its time?"* (the baseline)
- Blog post #2 — *"Profiling Triton FlashAttention-2 on RTX 4090, L40S, and H100"* (roofline + occupancy)

**Gate to advance:**

- [ ] Completed CS336 A1–A2
- [ ] Can explain (without notes) why FlashAttention is IO-aware
- [ ] Can run `nvprof` / `ncu` / `nsys` and read the output
- [ ] Have a clear, quantitative picture of where Noetik's training step spends its time

---

### Phase 1 — GPU Architecture & Kernel Depth (Months 4–7, ~120 hrs)

**Goal:** Stop being a "PyTorch user" — become someone who reads CUDA C and reasons about SMs.

**Theory reading:**

- [Programming Massively Parallel Processors (PMPP), 5th ed.](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1), Ch. 1–12 (memory hierarchy, warps, reductions, convolutions)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) — full read, learn the metric taxonomy
- GPU MODE lectures 6–20 (CUTLASS, quantization, fused kernels)
- Papers (deep read, take notes):
  - [FlashAttention v1](https://arxiv.org/abs/2205.14135)
  - [FlashAttention v2](https://arxiv.org/abs/2307.08691)
  - [FlashAttention v3](https://arxiv.org/abs/2407.08608) (Hopper-specific: WGMMA, TMA, FP8)

**Noetik practice (ties directly to your baseline):**

1. **Kernel-level profile of image encoder vs. decoder on H100.** Use `ncu` to identify the top-3 expensive kernels in each. Decide which is the most worth optimizing.
2. **Write a custom Triton kernel for your long-context workload.** Options depending on what your baseline showed:
   - A fused attention kernel optimized for the *long, variable-length* gene sequences (most existing FlashAttention assumes uniform context)
   - A fused image-patch embedding + positional encoding kernel for the encoder
   - An FP8 variant of a hot encoder kernel (image encoders tolerate FP8 well)
3. Integrate via `torch.compile` or `load_inline`, measure end-to-end step-time improvement on a real Noetik training run.

**Outputs:**

- Blog post #3 — *"What Nsight Compute told me about our multimodal training: image encoder vs. decoder kernels"*
- Blog post #4 — *"A Triton kernel for long-context gene-sequence attention"* (or whichever kernel you picked) — with `ncu` roofline + measured speedup

**Gate to advance:**

- [ ] Comfortable reading CUTLASS source
- [ ] Can read an `ncu` report and identify the bottleneck (compute / memory / latency / occupancy)
- [ ] Have a working custom kernel integrated into the Noetik training loop with measured speedup

---

### Phase 2 — Collective Communications & RDMA (Months 7–10, ~120 hrs)

**Goal:** Become the person on the team who reasons about NCCL and IB fabrics.

**Theory reading:**

- [Demystifying NCCL (arXiv:2507.04786)](https://arxiv.org/html/2507.04786v1) — the only systematic NCCL internals paper; read 2–3 times
- [NCCL source](https://github.com/NVIDIA/nccl), especially `src/transport/`, `src/collectives/`, and topology detection code
- [NVIDIA GPUDirect RDMA documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [DeepWiki: NCCL InfiniBand and RDMA Transport](https://deepwiki.com/NVIDIA/nccl/5.3-infiniband-and-rdma-transport)
- RDMA fundamentals: Mellanox/NVIDIA RDMA Programming Manual; `ibv_*` Verbs API basics
- Papers: [Ring Attention](https://arxiv.org/abs/2310.01889); [Horovod ring all-reduce](https://arxiv.org/abs/1802.05799)

**Noetik practice (the small-model/many-GPU regime is your edge):**

Most NCCL literature optimizes for the *large-message, bandwidth-bound* regime (huge models). With <1B params on 80 H100s, you're in the **small-message, latency-bound** regime — a much less-documented and more interesting problem.

1. **Baseline `nccl-tests`** (all-reduce, all-gather, reduce-scatter) at varying message sizes across 1, 2, 4, 8 nodes. Compare achieved bandwidth vs. theoretical (NVLink intra-node, IB/EFA inter-node).
2. **Trace gradient all-reduce in your actual training run.** Is it stalling? Where in the step? Use `nsys` with NCCL trace enabled.
3. **Tune NCCL for the small-message regime.** Variables that matter most here: `NCCL_ALGO` (tree often beats ring at small sizes), `NCCL_PROTO` (`LL128` for small messages), `NCCL_IB_HCA`, `NCCL_NSOCKS_PERTHREAD`, gradient bucket size in PyTorch DDP/FSDP. Document what moved the needle and what didn't.
4. **Bonus:** implement a toy ring all-reduce using CUDA + NCCL primitives, benchmark against NCCL's.

**Outputs:**

- Blog post #5 — *"NCCL tuning for the small-model, many-GPU regime: what nobody writes about"*
- Blog post #6 — *"Reading NCCL source: how ring all-reduce really works"*

**Gate to advance:**

- [ ] Can sketch NCCL ring vs. tree vs. double-binary-tree algorithms on a whiteboard
- [ ] Have benchmarked all-reduce at multiple node counts on Noetik and explained the bandwidth curve
- [ ] Understand the GPUDirect RDMA data path: GPU → HCA → wire → HCA → GPU (no CPU bounce)
- [ ] Can answer "why is my all-reduce slow?" by hypothesis (PCIe topology? NUMA? IB link health? wrong `NCCL_ALGO`? small-message latency overhead?)

---

### Phase 3 — Distributed Training Internals (Months 10–13, ~120 hrs)

**Goal:** Debate FSDP vs. ZeRO-3 vs. tensor parallelism vs. Megatron-3D from first principles, with quantitative MFU analysis on *the right kind of workload for each*.

**Theory reading (the full literature — you need this for interviews even if you won't use TP/PP at Noetik):**

- Papers:
  - [Megatron-LM 3D parallelism (arXiv:2104.04473)](https://arxiv.org/abs/2104.04473)
  - [ZeRO (arXiv:1910.02054)](https://arxiv.org/abs/1910.02054)
  - [PyTorch FSDP paper (arXiv:2304.11277)](https://arxiv.org/abs/2304.11277)
  - [Reducing Activation Recomputation in Large Transformers (Korthikanti et al., 2022)](https://arxiv.org/abs/2205.05198) — selective recomputation, sequence parallelism
  - [GPipe](https://arxiv.org/abs/1811.06965) + [PipeDream](https://arxiv.org/abs/1806.03377)
- Source code reading:
  - [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) — `megatron/core/parallel_state.py`, `tensor_parallel`, `pipeline_parallel`
  - PyTorch FSDP2 source in `torch/distributed/_composable/fsdp/`

**Noetik practice (this phase is reshaped from the original plan — your model is too small for TP/PP to make sense):**

Forcing 3D parallelism on a <1B model would be synthetic. Do the *actually-relevant-at-Noetik* projects instead:

1. **Dataloader optimization for high-resolution medical images.** If your Week-1 baseline showed dataloader > 15% of step time, this is your biggest lever. Try:
   - NVIDIA DALI for GPU-resident augmentation
   - A custom Rust loader (great for interview stories)
   - Sharded WebDataset / Mosaic StreamingDataset for S3-backed data
   - Measure: bytes/sec sustained, GPU idle time eliminated
2. **Sequence parallelism for long gene sequences.** Implement SP on top of FSDP — distributes activation memory across the SP group. Compare max sequence length before OOM, before vs. after.
3. **Activation checkpointing tuning.** Selective recomputation (Korthikanti et al.) — recompute only attention, not MLP. Measure memory vs. compute tradeoff.
4. **Communication/compute overlap.** Use `torch.distributed`'s async ops + custom hooks to overlap gradient all-reduce with backward pass. Measure step-time improvement.

**Separate "credentials" project — synthetic 3D parallelism (one weekend, rent A100s on Lambda/RunPod):**

Train a synthetic 7B model with FSDP+TP+PP for 1k steps. Measure MFU at each config. This gives you the interview story without wasting Noetik cluster time on a workload that doesn't fit your real model.

**Outputs:**

- Blog post #7 — *"Killing dataloader bottleneck on a multimodal training run: from 25% idle to 4%"* (or whichever optimization you ran)
- Blog post #8 — *"Sequence parallelism on a <1B model: when activation memory dominates"*
- Blog post #9 — *"3D parallelism on a synthetic 7B: a weekend study"*

**Gate to advance:**

- [ ] Can explain FSDP, ZeRO-3, TP, PP, SP, CP from first principles and when each applies
- [ ] Have measurably improved MFU on Noetik training (target: +20% relative)
- [ ] Have used DCGM to detect a real degraded GPU or thermal throttle (or to validate that none exist)

---

### Phase 4 — LLM Inference Systems (Months 13–15, ~80 hrs)

**Goal:** Be conversant in modern inference architecture; have hands inside vLLM or SGLang; have built something multimodal-inference-specific.

**Theory reading:**

- Papers:
  - [Orca: Continuous Batching (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu)
  - [PagedAttention / vLLM (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
  - [DistServe — prefill/decode disaggregation (arXiv:2401.09670)](https://arxiv.org/abs/2401.09670)
  - [SpecInfer / Speculative Decoding survey](https://arxiv.org/abs/2305.09781)
  - [SARATHI — chunked prefill (arXiv:2308.16369)](https://arxiv.org/abs/2308.16369)
- [vLLM "Anatomy of vLLM" blog (Sept 2025)](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html)
- [Mini-SGLang source + LMSYS post](https://www.lmsys.org/blog/2025-12-17-minisgl/) — read the full ~1k-line minimal impl
- vLLM scheduler source: `vllm/core/scheduler.py`, `vllm/worker/`

**Noetik practice (multimodal inference is your unique angle):**

Multimodal inference has a problem most LLM-serving papers don't address: the image encoder is one *enormous* prefill before any decoder token can be produced. TTFT is dominated by image processing.

1. **Implement chunked prefill for multimodal inference at Noetik.** Stream image-encoder activations into the decoder rather than waiting for full encoder forward. Measure TTFT improvement.
2. **Disaggregated encoder/decoder serving (the DistServe pattern, multimodal flavor).** Run image encoder workers on compute-heavy hardware (H100), decoder workers on memory-bandwidth-heavy hardware (L40S or H100 NVLink-bound). Measure throughput + cost.
3. **Bonus:** prefix-cache the image encoder output for repeated images (common in medical imaging workflows — same scan, different prompts).

**Alternative if (1)/(2) too ambitious:** Add a real feature/fix to vLLM or SGLang and get it merged. The community is actively working on multimodal — there are good entry-point issues.

**Outputs:**

- Blog post #10 — *"Chunked prefill for multimodal inference: TTFT under 100ms on a 1B image-encoder model"*
- Merged OSS PR to vLLM, SGLang, or LMDeploy (target: a multimodal-related contribution)

**Gate to advance:**

- [ ] Can sketch the vLLM step function from memory
- [ ] Understand prefill vs. decode at the kernel level
- [ ] Have one merged upstream PR *or* a working multimodal serving optimization at Noetik with measurable TTFT improvement

---

### Phase 5 — Production Reliability & Observability (Months 15–17, ~80 hrs)

**Goal:** Be the person who designs SLOs that include ECC errors, NVLink degradation, NCCL timeouts, *and* multimodal-specific failures.

**Theory reading:**

- NVIDIA DCGM docs + DCGM Exporter for Prometheus
- Google SRE book chapters on SLI/SLO/error budgets (skim if already known)
- Anthropic / OpenAI / Meta production engineering blog posts on training reliability (e.g., [Meta's Llama-3 16K-GPU training run](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) failure-analysis section)
- eBPF for networking: [Brendan Gregg's eBPF resources](https://www.brendangregg.com/ebpf.html)
- Chaos engineering patterns adapted to GPU clusters (Netflix Chaos Monkey, Gremlin)

**Noetik practice — build the real reliability layer:**

1. **DCGM-based health-check daemon** covering generic LLM cluster failures (GPU temp, ECC errors, NVLink errors, PCIe replay counts, SM clocks) *plus* Noetik-specific signals:
   - PCIe bandwidth saturation from medical-image dataloading (you'll see this stress the bus way more than typical LLM training)
   - Storage→GPU pipeline throughput (latency from object store to GPU memory)
   - Thermal headroom under sustained image I/O load
2. **Automatic quarantine:** degraded GPU → cordoned in Kubernetes
3. **SLO dashboard:** training-job MFU, NCCL timeout rate, inference P99 TTFT (including image-encoder prefill latency), prefix-cache hit rate, data-pipeline throughput
4. **Chaos experiment:** inject a GPU thermal throttle (`nvidia-smi --gom`) or degrade an IB link mid-training, measure detection + recovery time
5. **Runbook for the top 5 failure modes you've actually seen at Noetik** — this is the most valuable interview artifact in the whole plan

**Outputs:**

- Blog post #11 — *"GPU cluster SLOs for multimodal training: PCIe bandwidth, NVLink degradation, and the failure modes nobody writes about"*
- Blog post #12 — *"Building a DCGM-based health-check + auto-quarantine layer on Kubernetes"*

**Gate to advance:**

- [ ] Noetik cluster has a real DCGM → Prometheus → Grafana stack with alerts wired in
- [ ] At least one written post-mortem from a real Noetik incident with action items
- [ ] Can name the top 10 DCGM metrics that matter and what each one tells you

---

### Phase 6 — Capstone + Interview Prep (Months 17–18, ~60 hrs)

**Goal:** Convert everything into interview-ready stories and target the application.

**Capstone (pick one — portfolio-grade):**

- A non-trivial OSS contribution: a real perf improvement to vLLM/SGLang/Megatron/NCCL/PyTorch FSDP with benchmarks
- A long-form technical writeup: *"End-to-end profile of Noetik's multimodal training run, kernel by kernel"* — Nsight + DCGM data, the full stack, the full optimizations you've shipped
- A novel piece: implement disaggregated encoder/decoder inference at Noetik and write up the design + measured tradeoffs

**Interview prep:**

- ML systems design: "Design vLLM", "Design distributed training for a 70B model on 1024 H100s", "Why is my all-reduce slow — debug this with me"
- Practice the *vertical question*: pick any layer (NVLink → PCIe → HCA → NCCL → PyTorch → user code) and be able to talk at any depth
- LeetCode-style refresh (1–2 hrs/week, basic level — these roles have a coding round but it's not the bar)
- Read every Anthropic / OpenAI engineering blog post on inference and training
- Apply: Anthropic Performance Engineering, Inference, Infra Efficiency, AIRE; OpenAI Inference / Applied / Scaling

**Outputs:**

- Capstone artifact (PR or blog series)
- Updated resume + portfolio link to `building-ai-infra`

---

## Critical Files / Resources (one-stop reference)

**Courses:**

- [Stanford CS336 Spring 2025](https://cs336.stanford.edu/spring2025/) — foundation
- [GPU MODE Lectures](https://github.com/gpu-mode/lectures) — kernel writing

**Books:**

- [Programming Massively Parallel Processors (PMPP), 5th ed.](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1)

**Source code to read (priority order):**

- vLLM `vllm/core/scheduler.py`, `vllm/worker/` — inference scheduling
- [Mini-SGLang](https://github.com/sgl-project/mini-sglang) — minimal inference engine (~1k lines)
- PyTorch FSDP2 `torch/distributed/_composable/fsdp/`
- [Megatron-Core](https://github.com/NVIDIA/Megatron-LM) `megatron/core/parallel_state.py`
- [NCCL](https://github.com/NVIDIA/nccl) `src/transport/`, `src/collectives/`

**Must-read papers (10):**

1. [FlashAttention v1](https://arxiv.org/abs/2205.14135) / [v2](https://arxiv.org/abs/2307.08691) / [v3](https://arxiv.org/abs/2407.08608)
2. [Megatron-LM 3D](https://arxiv.org/abs/2104.04473)
3. [ZeRO](https://arxiv.org/abs/1910.02054)
4. [PyTorch FSDP](https://arxiv.org/abs/2304.11277)
5. [Reducing Activation Recomputation (Korthikanti)](https://arxiv.org/abs/2205.05198)
6. [Ring Attention](https://arxiv.org/abs/2310.01889)
7. [Orca Continuous Batching](https://www.usenix.org/conference/osdi22/presentation/yu)
8. [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
9. [SARATHI (chunked prefill)](https://arxiv.org/abs/2308.16369)
10. [Demystifying NCCL](https://arxiv.org/html/2507.04786v1)

**Reference docs (bookmark):**

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/)
- [ezyang PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/) + [Dispatcher](https://blog.ezyang.com/2020/09/lets-talk-about-the-pytorch-dispatcher/)

---

## Verification: How to Know This Is Working

**Monthly checkpoint** (15 min, end of month):

- Did I hit my reading targets?
- Did I produce a blog post or PR this month?
- Did I run something on real Noetik hardware that gave me a new mental model?
- If I had a 30-min phone screen tomorrow on this phase's topic, would I pass?

**Mid-plan reality check** (Month 9, end of Phase 2):

- Can I, *unprompted*, walk through the NVLink → NCCL → PyTorch distributed reasoning chain that JD-0 calls out?
- Do I have 5–6 blog posts up I'd link a recruiter to?
- Have I measurably improved something on the Noetik cluster I'd put on a resume?

If "no" to any of the above at Month 9: rebalance — drop scope on later phases, double down on the rate-limiter.

**Final readiness check** (Month 17):

- Mock interview with someone in the field (GPU MODE Discord has people willing)
- Apply to a non-target role first as a calibration interview
- If the capstone is OSS, get it merged before applying
