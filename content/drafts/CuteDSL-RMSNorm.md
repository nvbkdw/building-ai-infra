---
title: "RMSNorm Kernels in CuteDSL"
date: 2026-09-09
tags: ["GPU kernel", "LLM", "RMSNorm", "CuteDSL"]
author: "Ryan H."
description: "This blog post covers RMSNorm Kernels in CuteDSL"
summary: "This blog post covers RMSNorm Kernels in CuteDSL."
---

# RMSNorm
RMSNorm is a common operation using in transformer model, it's normalize any tensor along feature dimention.

For any tensor $U$ whose last dimension has width $J$, define

$$
\operatorname{RMSNorm}_{\gamma,\epsilon}(U)_{\ldots,j}
=\gamma_j\,
\frac{U_{\ldots,j}}
{\sqrt{\frac{1}{J}\sum_{r=0}^{J-1}U_{\ldots,r}^{\,2}+\epsilon}},
\qquad \gamma\in\mathbb R^J.
$$

For input $[B,T,D]$, compute one mean square per batch item and token, producing a
$[B,T,1]$ denominator that broadcasts over $D$. Do not reduce over tokens
or batch items. The output retains the input shape.

The denominator controls vector magnitude; the learned $\gamma$ restores
per-channel scale. $\epsilon$ belongs inside the square root.

# Pytorch Implementation

Assuming input and weight is BFloat16, RMSNorm casts its input to FP32 before squaring, uses `mean(-1, keepdim=True)`, and multiplies by the reciprocal square root. It casts the normalized vector back to the input dtype before multiplying by the learned weight

```
def rms_norm_ref(x, weight, eps):
    """FP32 numerical oracle before the kernel's final BF16 storage rounding."""
    x_fp32 = x.float()
    scale = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + eps)
    return x_fp32.to(weight.dtype) * scale * weight
```

# Kernel

Although it's a simple operator, RMSNorm torch implementation contains many small kernel launch: up cast, square, mean, add, rsqrt, down cast, multiple. It's common to fuse them into one kernel. 

TBD diagram


Cute DSL provide layout algebra to calcuate mapping from thread idx to physical address of data each thread need to process. meta-programming 

In this example, we rewrite elementwise kernel with two levels of tiling:

* the thread-block level
* the thread level with TV Layout and tiling

### CTA tiling

TBD

### TV layout
CuTe introduces TV layout to represent this mapping from thread index and value index (i.e., the 4 elements loaded per thread) to the logical coordinate space of a tensor.  By configuring different TV layouts, we can experiment with different memory access patterns with minimal code changes.

Definition: TV Layout is rank-2 layout which maps (thread_index, value_index) to logical coordinate of tensor.

We always have TV Layout with canonical form as (thread_domain, value_domain):(..., ...).

With TV Layout, each thread can find logical coordinates or indices of data partitioned to current thread.

### Wrap reduce
Reference: https://medium.com/@aditi.sikarwar25/parallel-reduction-in-cuda-warp-shuffle-instructions-register-level-communication-3f1e3dde4ae5

### Memory coalesing
GPU global memory is accessed via 32-byte memory transactions.
https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#coalesced-global-memory-access

```
from_dlpack(x2d.detach(), assumed_align=64)
        .mark_compact_shape_dynamic(mode=0),
```
An aligned allocation therefore does not prove that every row starts at a vector-aligned address. `mark_compact_shape_dynamic` make only the row count remains dynamic. Width and row stride stay known, and the weight layout is static. This allows CuTe to vectorize the existing .load() and store expressions.



### Complete Cute DSL kernel implementation:

```python
@cute.jit
def warp_sum(x):
    for delta in cutlass.range_constexpr(5):
        x = x + cute.arch.shuffle_sync_bfly(x, 1 << delta)
    return x


@cute.jit
def warp_sum_descending(x):
    for i in cutlass.range_constexpr(5):
        x = x + cute.arch.shuffle_sync_bfly(x, 16 >> i)
    return x

@cute.kernel
def rms_norm_kernel(mX, weight, mY, rows: cutlass.Int32, width: cutlass.Constexpr, eps: cutlass.Constexpr, tv_layout: cute.Layout):
    # kernel 
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    lane = cute.arch.lane_idx()
    
    # get CTA local tile
    blk_coord = (None, bidx)
    gX = mX[blk_coord]
    gY = mY[blk_coord]

    # get thread local vals
    w_tv_layout = cute.make_ordered_layout((32, cute.ceil_div(width, 32)), order=(1, 0))
    tidfrgX = cute.composition(gX, tv_layout)
    tidfrgY = cute.composition(gY, tv_layout)
    tidfrgW = cute.composition(weight, w_tv_layout)
    thr_coord = (tidx, None)
    thrX = tidfrgX[thr_coord]
    thrY = tidfrgY[thr_coord]
    # Each warp reuses the same weight vector; its thread coordinate is a lane.
    thrW = tidfrgW[(lane, None)]

    # A block owns four rows. Skip missing rows in the final block; the
    # predicate is identical for all lanes participating in a warp shuffle.
    if bidx * 4 + tidx // 32 < rows:
        # BF16 storage, FP32 arithmetic, and one rounding step at the store.
        v = thrX.load().to(cutlass.Float32)
        squared_sum = cutlass.Float32(0)
        for i in cutlass.range_constexpr(thrX.shape[0]):
            squared_sum += v[i] * v[i]

        scale = cute.rsqrt(warp_sum(squared_sum) / width + eps)
        # TODO: load weight to shared mem first
        w = thrW.load().to(cutlass.Float32)

        assert v.shape == w.shape
        thrY[None] = (v * scale * w).to(cutlass.BFloat16)


@cute.jit
def rms_norm(x, weight, y, width: cutlass.Constexpr, eps: cutlass.Constexpr, stream: cuda.CUstream):
    M = 4 # M rows per CTA
    # coalesced_bytes = 128
    thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    val_layout = cute.make_ordered_layout((1, cute.ceil_div(width, 32),), order=(1, 0))
    tiler_m, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    mX = cute.zipped_divide(x, tiler_m)
    mY = cute.zipped_divide(y, tiler_m)
    _rms_norm_kernel(
        mX, weight, mY, x.shape[0], width, eps, tv_layout
    ).launch(
        grid=((x.shape[0]+M-1)//M, 1, 1),
        block=(32*M, 1, 1), # one row per warp, M warps
        stream=stream
    )
```


# Performance Analysis and Benchmarking

TBD: roofline and performance analysis

Runing RMSNorm kernel (1024, 16, 128) on NVIDIA GB10, show 10x speech up:

```
RMSNorm (1024, 16, 128) on NVIDIA GB10; BF16 input/output, FP32 arithmetic
CUDA-event function latency: 10 warmups/repeat, 20 repeats, 100 calls/repeat

Backend                    Median (us)     Min (us)     Max (us)
torch_reference_bf16           228.683      228.143      240.033
cute_kernel_bf16                21.084       20.521       24.515
CuTe speedup (Torch / CuTe): 10.85x
```
