---
title: "Kernel 3 - Constant Memory"
date: 2026-06-20
tags: ["kernel","constant memory"]
author: "Ryan H."
description: "This blog post covers the constant memory."
summary: "This blog post covers the constant memory."
---

On NVIDIA GPUs, **constant cache is real per-SM hardware**, not just a CUDA compiler abstraction.

CUDA exposes a **constant memory space** with `__constant__`. PTX exposes the same idea as the `.const` state space, accessed with `ld.const` instructions. NVIDIA’s PTX documentation describes each multiprocessor as having “a read-only constant cache” shared by the scalar cores in that multiprocessor, used to speed reads from constant memory. ([NVIDIA Docs][1])

A simplified path is:

```cpp
__constant__ float coeff[1024];

__global__ void kernel(float *out) {
    float x = coeff[0];   // compiled as a constant-memory load
    out[threadIdx.x] = x;
}
```

Conceptually:

```text
CUDA __constant__ variable
        ↓
PTX .const object / ld.const instruction
        ↓
SM constant-load path
        ↓
per-SM constant cache
        ↓ miss: fetch from device memory hierarchy
```

The **storage** for user-visible constant memory is device memory, not the small on-chip cache. NVIDIA’s CUDA Best Practices Guide lists constant memory as off-chip, cached, read-only, visible to all threads and the host, with host-allocation lifetime. ([NVIDIA Docs][2]) The CUDA Programming Guide lists the constant memory size as **64 KB**, and the “cache working set per SM for constant memory” as **8 KB** for the currently documented compute capabilities. ([NVIDIA Docs][3])

The special part is not only caching; it is the **broadcast behavior**. Constant cache is optimized for the case where threads in a warp read the same address, or a small number of addresses. If all lanes in a warp read the same constant address, the hardware can serve/broadcast the value efficiently; NVIDIA says this can be as fast as a register access. If lanes read different constant addresses, those distinct addresses are serialized, so cost grows roughly with the number of unique addresses touched by the warp. ([NVIDIA Docs][2])

So **each SM has special constant-cache hardware.**
It is a small, read-only, per-SM cache for the constant address space. It is excellent for warp-uniform reads like coefficients, scalar parameters, lookup tables indexed uniformly across a warp, convolution masks, small model constants, etc. It is poor for “each thread reads a different table entry” patterns, because those accesses serialize.

A useful mental model:

```text
Same address across warp:
    32 lanes → one constant-cache request → broadcast to lanes

4 unique addresses across warp:
    32 lanes → about 4 serialized constant-cache requests

32 unique addresses across warp:
    32 lanes → about 32 serialized requests; often worse than normal global/read-only cache
```

This is why constant memory is not “faster global memory” in general. It is **fast for uniform or near-uniform access**. For per-thread random indexing, ordinary global memory, the read-only data path, texture memory, or shared-memory tiling may be better.

[1]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html "1. Introduction — PTX ISA 9.3 documentation"
[2]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html "CUDA Best Practices Guide — CUDA C++ Best Practices Guide 13.3 documentation"
[3]: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html "5.1. Compute Capabilities — CUDA Programming Guide"
