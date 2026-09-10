---
title: "GPU Hardware Spec Cheat Sheet for Kernel Developers"
date: 2026-07-11
tags: ["gpu-kernel", "cuda", "ampere", "hopper", "blackwell", "hardware"]
author: "Ryan H."
description: "A single-page, Excel-style lookup of NVIDIA data-center GPU specs — Ampere, Ada, Hopper, and Blackwell — from memory bandwidth and peak FLOPS down to SM counts, register files, shared-memory banks, and tensor-core details."
summary: "Fast lookup tables of hardware specs across A10, A100, L40, L40S, H100, H200, B200, and B300 — high-level throughput plus the microarchitecture numbers you actually need when writing kernels."
---

Every kernel-tuning decision eventually bottoms out in a hardware number: how many bytes per second the HBM can feed you, how much shared memory a block can claim before occupancy collapses, how many warps an SM can keep resident, what the tensor cores can actually retire per cycle. This page collects those numbers for the data-center GPUs you're most likely to target, in a form you can scan in a second.

## How to read these tables

- **Tensor throughput is quoted DENSE** (TFLOPS, or TOPS for integer). For **2:4 structured sparsity, multiply by 2** — *except B300 FP4, which is only ~1.33× (see note)*. NVIDIA's marketing datasheets usually print the sparse number; dense is what a normal GEMM achieves, so dense is the honest baseline here.
- **`—`** means the format is not supported by that generation's tensor cores.
- **Vector (non-tensor) FP32/FP64** are the plain CUDA-core rates — what your `float`/`double` arithmetic hits when you're *not* using tensor cores.
- Values are for the **flagship variant** of each SKU (A100 = 80 GB SXM4, H100 = SXM5). Variant deltas (PCIe, 40 GB, NVL) are in a separate table below.
- Register file, shared-memory banks, and warp size have been **constant since Volta** — 64 K 32-bit registers per SM, 32 banks × 4 bytes, 32-lane warps — so those rarely move. What moves is *shared-memory capacity, thread/warp caps, and tensor-core capability*, which is exactly where occupancy bugs come from.

---

## 1 · Platform & memory

| Spec | A100 | A10 | L40 | L40S | H100 | H200 | B200 | B300 |
|---|---|---|---|---|---|---|---|---|
| Architecture | Ampere | Ampere | Ada Lovelace | Ada Lovelace | Hopper | Hopper | Blackwell | Blackwell Ultra |
| Chip / die | GA100 | GA102 | AD102 | AD102 | GH100 | GH100 | 2× die (NV-HBI) | 2× die (NV-HBI) |
| Compute capability | 8.0 (`sm_80`) | 8.6 (`sm_86`) | 8.9 (`sm_89`) | 8.9 (`sm_89`) | 9.0 (`sm_90`) | 9.0 (`sm_90`) | 10.0 (`sm_100`) | 10.0 (`sm_100`)◆ |
| Process node | TSMC N7 | Samsung 8N | TSMC 4N | TSMC 4N | TSMC 4N | TSMC 4N | TSMC 4NP | TSMC 4NP |
| Transistors | 54.2 B | 28.3 B | 76.3 B | 76.3 B | 80 B | 80 B | 208 B | 208 B |
| Memory size | 80 GB | 24 GB | 48 GB | 48 GB | 80 GB | 141 GB | 192 GB✱ | 288 GB |
| Memory type | HBM2e | GDDR6 | GDDR6 ECC | GDDR6 ECC | HBM3 | HBM3e | HBM3e | HBM3e |
| **Bandwidth** | **2,039 GB/s** | **600 GB/s** | **864 GB/s** | **864 GB/s** | **3,350 GB/s** | **4,800 GB/s** | **8,000 GB/s** | **8,000 GB/s** |
| Memory bus | 5,120-bit | 384-bit | 384-bit | 384-bit | 5,120-bit | 6,144-bit | 8,192-bit | 8,192-bit |
| L2 cache | 40 MB | 6 MB | 96 MB | 96 MB | 50 MB | 50 MB | ~126 MB† | ~126 MB† |
| TDP | 400 W | 150 W | 300 W | 350 W | 700 W | 700 W | 1,000–1,200 W | 1,400 W |
| NVLink (per GPU) | 600 GB/s (v3) | — | — | — | 900 GB/s (v4) | 900 GB/s (v4) | 1,800 GB/s (v5) | 1,800 GB/s (v5) |
| PCIe | Gen4 (64 GB/s) | Gen4 | Gen4 | Gen4 | Gen5 (128 GB/s) | Gen5 | Gen5 (128 GB/s) | Gen6 (256 GB/s) |
| Boost clock | 1,410 MHz | 1,695 MHz | 2,490 MHz | 2,520 MHz | 1,980 MHz | 1,980 MHz | ~1,965 MHz‡ | ~1,965 MHz‡ |

<sub>✱ B200 raw capacity is 192 GB (8× 24 GB HBM3e). Shipped configs report **180 GB** (DGX/HGX B200) up to **186 GB** (GB200); 192 GB is the now-standard headline. &nbsp; † Blackwell L2 is not officially published; ~126 MB total is from third-party die analysis. &nbsp; ‡ Blackwell clocks are not officially published (derived). &nbsp; ◆ B300 is the same Blackwell DC family as B200; a newer toolkit may expose a distinct minor target — verify against your CUDA version.</sub>

---

## 2 · Execution resources

| Spec | A100 | A10 | L40 | L40S | H100 | H200 | B200 | B300 |
|---|---|---|---|---|---|---|---|---|
| **SMs** | **108** | **72** | **142** | **142** | **132** | **132** | **148** | **160** |
| FP32 (CUDA) cores | 6,912 | 9,216 | 18,176 | 18,176 | 16,896 | 16,896 | 18,944 | 20,480 |
| FP32 cores / SM | 64 | 128 | 128 | 128 | 128 | 128 | 128 | 128 |
| FP64 cores / SM | 32 | 2 | 2 | 2 | 64 | 64 | 64 | few (gutted)◊ |
| Tensor cores (total) | 432 | 288 | 568 | 568 | 528 | 528 | 592 | 640 |
| Tensor-core gen | 3rd | 3rd | 4th | 4th | 4th | 4th | 5th | 5th |
| Tensor cores / SM | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| Warp size (lanes) | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 32 |

<sub>◊ B300 deprioritizes FP64 to ~1.3 TFLOPS (roughly a 97% cut vs B200) and pushes doubles through tensor-core emulation — do **not** target B300 for FP64 HPC.</sub>

---

## 3 · Peak throughput by precision (DENSE)

TFLOPS unless noted; INT8 is TOPS. **Sparse (2:4) = 2× these values**, except B300 FP4 (see note under the table).

| Precision | A100 | A10 | L40 | L40S | H100 | H200 | B200 | B300 |
|---|---|---|---|---|---|---|---|---|
| FP4 tensor | — | — | — | — | — | — | **10,000** | **15,000** |
| FP6 tensor | — | — | — | — | — | — | 5,000 | 5,000 |
| FP8 tensor | — | — | 362 | 733 | 1,979 | 1,979 | 5,000 | 5,000 |
| FP16 / BF16 tensor | 312 | 125 | 181 | 362 | 989 | 989 | 2,500 | 2,500 |
| TF32 tensor | 156 | 62.5 | 90.5 | 183 | 494 | 494 | 1,250 | 1,250 |
| INT8 tensor (TOPS) | 624 | 250 | 362 | 733 | 1,979 | 1,979 | 5,000 | 5,000 |
| FP32 (vector) | 19.5 | 31.2 | 90.5 | 91.6 | 67 | 67 | ~80 | ~80 |
| FP64 tensor | 19.5 | — | — | — | 67 | 67 | 40 | ~1.3 |
| FP64 (vector) | 9.7 | 0.49 | ~1.4 | ~1.4 | 34 | 34 | 40 | ~1.3 |

> **The Blackwell FP4 sparsity trap.** For A100→Hopper, sparse is always 2× dense. For **B200** the same holds (FP4 = 10,000 dense / **20,000 sparse**). But **B300**'s FP4 is 15,000 dense / **20,000 sparse** — only **~1.33×**, *not* 2×. NVIDIA raised the dense rate but left the sparse peak where B200 was, so the naïve "sparse = 30,000" is wrong.

> **The Ada FP16-accumulate trap (L40 vs L40S).** These are the *same AD102 silicon* — identical 142 SMs, 18,176 CUDA cores, 96 MB L2, 864 GB/s — yet the L40S table shows exactly 2× the tensor throughput. That's a **reporting convention**, not hardware: the L40 datasheet quotes the FP32-accumulate rate, the L40S the full FP16-accumulate rate. On any Ada part, **FP16 tensor MMA with FP32 accumulate runs at half rate** — pick your accumulator precision deliberately.

---

## 4 · Occupancy & on-chip memory limits

These are fixed per **compute capability**, so they're grouped by CC. This is the table you reach for when a kernel won't hit target occupancy or a `__shared__` allocation won't launch.

| Limit | 8.0 (A100) | 8.6 (A10) | 8.9 (L40 / L40S) | 9.0 (H100 / H200) | 10.0 (B200 / B300) |
|---|---|---|---|---|---|
| Registers / SM | 65,536 (256 KB) | 65,536 | 65,536 | 65,536 | 65,536 |
| Max registers / thread | 255 | 255 | 255 | 255 | 255 |
| Combined L1 + shared / SM | 192 KB | 128 KB | 128 KB | 256 KB | 256 KB |
| **Max configurable shared / SM** | **164 KB** | **100 KB** | **100 KB** | **228 KB** | **228 KB** |
| Max shared / thread block | 163 KB | 99 KB | 99 KB | 227 KB | 227 KB |
| Shared-memory banks | 32 × 4 B | 32 × 4 B | 32 × 4 B | 32 × 4 B | 32 × 4 B |
| Tensor Memory (TMEM) / SM | — | — | — | — | 256 KB |
| **Max threads / SM** | **2,048** | **1,536** | **1,536** | **2,048** | **2,048** |
| Max warps / SM | 64 | 48 | 48 | 64 | 64 |
| Max thread blocks / SM | 32 | 16 | 24 | 32 | 32 |
| Max threads / block | 1,024 | 1,024 | 1,024 | 1,024 | 1,024 |
| Thread block clusters | No | No | No | Yes (≤8, 16 non-portable) | Yes (≤8, 16 non-portable) |
| TMA (async bulk copy) | No | No | No | Yes | Yes |
| Distributed shared mem (DSMEM) | No | No | No | Yes | Yes |

> **The `sm_86`/`sm_89` occupancy trap.** A10, L40, and L40S cap at **1,536 threads/SM (48 warps)**, not 2,048 — and **100 KB** shared/SM, not 164/228 KB. A block config that saturates an A100 or H100 can silently drop to 75% occupancy on these parts. Recompute your block dimensions per target, don't copy them across generations.
>
> To use **more than 48 KB** of shared memory per block on any of these GPUs you must opt in via `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` — static `__shared__` above 48 KB won't compile.

---

## 5 · Variant deltas

Same die, different memory/power/clock binning. Compute per SM is unchanged unless SMs are fused off.

### A100 (all GA100, CC 8.0)

| Variant | Memory | Bandwidth | TDP | NVLink |
|---|---|---|---|---|
| A100 40 GB PCIe | 40 GB HBM2 | 1,555 GB/s | 250 W | bridge 600 GB/s |
| A100 80 GB PCIe | 80 GB HBM2e | 1,935 GB/s | 300 W | bridge 600 GB/s |
| A100 40 GB SXM4 | 40 GB HBM2 | 1,555 GB/s | 400 W | 600 GB/s |
| **A100 80 GB SXM4** *(table above)* | 80 GB HBM2e | 2,039 GB/s | 400 W | 600 GB/s |

### H100 (all GH100, CC 9.0)

| Variant | SMs | Memory | Bandwidth | FP16 tensor (dense) | TDP | NVLink |
|---|---|---|---|---|---|---|
| **H100 SXM5** *(table above)* | 132 | 80 GB HBM3 | 3,350 GB/s | 989 | 700 W | 900 GB/s |
| H100 PCIe | 114 | 80 GB HBM2e | 2,000 GB/s | 756 | 350 W | 600 GB/s |
| H100 NVL (per card) | 132 | 94 GB HBM3 | 3,900 GB/s | 835 | 350–400 W | 600 GB/s |

<sub>Note the memory-type difference: **H100 PCIe uses HBM2e**, not HBM3. B200 also ships in a lower-power HGX/DGX tier (1,000 W): FP4 dense ~9,000, FP8 dense ~4,500, 180 GB — vs the 1,200 W GB200 numbers in the main table.</sub>

---

## 6 · Per-architecture notes for kernel authors

**Ampere (A100 / A10).** 3rd-gen tensor cores; **no FP8**. A100's async copy (`cp.async`) is the big lever — stage global→shared through the pipeline to overlap loads with math. A100 keeps the full 2,048 threads/SM and 164 KB shared budget; A10 (a GA102 graphics-derived part) drops to 1,536 threads and 100 KB, with tiny FP64 — treat it as an inference/light-training card, not an HPC part.

**Ada Lovelace (L40 / L40S).** Same occupancy caps as A10 (1,536 threads, 100 KB shared) but **4th-gen tensor cores with FP8** (E4M3/E5M2) and a Transformer Engine. GDDR6, not HBM — bandwidth (864 GB/s) is the usual ceiling, so these reward cache-friendly tiling and the large 96 MB L2. Mind the FP16-accumulate half-rate noted above.

**Hopper (H100 / H200).** The programming model changes here. To approach peak you need **`wgmma`** (warpgroup async MMA, operands fed from shared memory), **TMA** for bulk async global↔shared tiles with a single descriptor, **thread-block clusters** (co-scheduled blocks within a GPC) and **DSMEM** (one block reads another's shared memory over the SM-to-SM network). H200 is compute-identical to H100 SXM5 — only the memory changed (141 GB HBM3e, 4.8 TB/s), which directly lifts memory-bound and long-context workloads.

**Blackwell (B200 / B300).** 5th-gen tensor cores add **FP4/FP6** and microscaling (MXFP/NVFP4) formats via the 2nd-gen Transformer Engine. Operands are now staged in **Tensor Memory (TMEM)** — 256 KB/SM, *separate from shared memory* — and driven by the new **`tcgen05`** MMA instructions; the register-resident MMA model can't saturate these cores. Each GPU is two reticle-limited dies fused by a 10 TB/s NV-HBI link and presented to CUDA as one device. B300 (Blackwell Ultra) adds memory (288 GB) and dense FP4, doubles the special-function throughput for attention/softmax, moves to PCIe Gen6 — but **guts FP64**.

---

## 7 · Quick rules of thumb

- **Roofline ridge point** ≈ peak FLOPS ÷ bandwidth. It climbs steeply across generations: an A100 (BF16) sits near ~150 FLOP/byte, an H100 near ~300, a B200 (FP8/FP4) well past that — so a kernel that's compute-bound on A100 can be memory-bound on Blackwell. Recheck arithmetic intensity per target.
- **Registers cap occupancy first.** 65,536 regs/SM ÷ (regs/thread × threads/block) sets your resident blocks. At 255 regs/thread you get one block of 256 threads per SM. Watch `-maxrregcount` / `__launch_bounds__`.
- **Shared memory is the second gate.** On `sm_86`/`sm_89` you have 100 KB/SM; on Hopper/Blackwell up to 228 KB. A 227 KB single-block kernel means one block per SM — fine for a persistent-kernel design, fatal for a latency-bound one.
- **Bank conflicts are generation-independent:** 32 banks × 4 bytes everywhere. Pad leading dimensions (or use swizzling) so warp accesses hit distinct banks.
- **FP8 needs Ada+; FP4/FP6 need Blackwell.** Don't design a quantized kernel around a format the target can't retire in hardware.
- **Don't run FP64 on A10, L40, L40S, or B300.** Their double-precision rates are a rounding error; use A100, H100/H200, or B200.

---

<sub>Sources: NVIDIA A100/A10/L40/L40S/H100/H200 datasheets, the Ampere, Ada, Hopper, and Blackwell architecture whitepapers, and the CUDA C++ Programming Guide compute-capability tables. Tensor figures are shown dense; datasheet-published sparse figures are 2× (except B300 FP4). Blackwell L2 size and clocks are third-party/derived and marked accordingly; verify power/memory against the specific board (SXM vs PCIe, GB200 vs HGX) you deploy on.</sub>
