---
title: "Matmul Optimization Log Book"
date: 2026-06-16
tags: ["matmul","optimization"]
author: "Ryan H."
description: "This blog post covers the matmul optimization log book."
summary: "This blog post covers the matmul optimization log book."
cover:
    image: "matmul-optimization-log-book.png"
    alt: "Matmul Optimization Log Book"
    relative: true
---



## Hardware DGX Spark,
TBD: roofline model of DGX Spark, computation density of GEMM kernel.

Unified memory, GB10 GPU

### Compute Performance
* AI Compute: Up to 1,000 TOPS (trillion operations per second) inference and up to 1 PFLOP (petaFLOP) at FP4 precision with sparsity
* CUDA Cores: 6,144
* Copy Engines: 2 (enables simultaneous data transfers to and from GPU memory, improving throughput for AI workloads)
* Memory Bandwidth: 273 GB/s
* Memory Channels: 16 channels (256 bit) LPDDR5X 8533


### 2026-06-16 - native GEMM kernel
=== 512 x 512 x 512 (T = 16) ===
Kernel execution time: 0.217536 ms
=== 512 x 512 x 512 (T = 32) ===
Kernel execution time: 0.229344 ms
=== 1024 x 1024 x 1024 (T = 16) ===
Kernel execution time: 1.246656 ms
=== 1024 x 1024 x 1024 (T = 32) ===
Kernel execution time: 1.258464 ms
=== 2048 x 2048 x 2048 (T = 16) ===
Kernel execution time: 9.453440 ms
=== 2048 x 2048 x 2048 (T = 32) ===
Kernel execution time: 9.223680 ms
=== 4096 x 4096 x 4096 (T = 16) ===
Kernel execution time: 110.581406 ms
=== 4096 x 4096 x 4096 (T = 32) ===
Kernel execution time: 109.119873 ms
=== 8192 x 8192 x 8192 (T = 16) ===
Kernel execution time: 893.264099 ms
=== 8192 x 8192 x 8192 (T = 32) ===
Kernel execution time: 891.974365 ms

```
__global__
void matmulKernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int n, int m, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < k) {
        __nv_bfloat16 sum = __float2bfloat16(0.0f);
        for (int i = 0; i < m; i++) {
            sum = __hfma(A[row * m + i], B[i * k + col], sum);
        }
        C[row * k + col] = sum;
    }
}
```

### 2026-06-16 - tiled shared-memory GEMM kernel

Matrix multiplication with shared-memory and tiling.

```
__global__
void matmulKernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int n, int k, int m, int t) {

    extern __shared__ __nv_bfloat16 shared_mem[];
    __nv_bfloat16* a_ds = shared_mem;
    __nv_bfloat16* b_ds = shared_mem + t*t;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __nv_bfloat16 p = __float2bfloat16(0.0f);
    int numPhases = (k + t - 1) / t;
    for (int ph = 0; ph < numPhases; ++ph) {
        // load one tile of A into shared memory, zero-padding out-of-range elements
        int aCol = ph*t + threadIdx.x;
        if (row < n && aCol < k) {
            a_ds[threadIdx.y*t + threadIdx.x] = A[row*k + aCol];
        } else {
            a_ds[threadIdx.y*t + threadIdx.x] = __float2bfloat16(0.0f);
        }
        // load one tile of B into shared memory, zero-padding out-of-range elements
        int bRow = ph*t + threadIdx.y;
        if (bRow < k && col < m) {
            b_ds[threadIdx.y*t + threadIdx.x] = B[bRow*m + col];
        } else {
            b_ds[threadIdx.y*t + threadIdx.x] = __float2bfloat16(0.0f);
        }
        __syncthreads();

        for (int i = 0; i < t; i++) {
            p = __hfma(a_ds[threadIdx.y*t + i], b_ds[i*t + threadIdx.x], p);
        }
        __syncthreads();
    }
    if (row < n && col < m) {
        C[row * m + col] = p;
    }
}
```

Result is about 4X faster than the native GEMM kernel.

=== 512 x 512 x 512 (T = 16) ===
Kernel execution time: 0.228608 ms
=== 512 x 512 x 512 (T = 32) ===
Kernel execution time: 0.270880 ms
=== 1024 x 1024 x 1024 (T = 16) ===
Kernel execution time: 1.348512 ms
=== 1024 x 1024 x 1024 (T = 32) ===
Kernel execution time: 1.505696 ms
=== 2048 x 2048 x 2048 (T = 16) ===
Kernel execution time: 10.193824 ms
=== 2048 x 2048 x 2048 (T = 32) ===
Kernel execution time: 10.903456 ms
=== 4096 x 4096 x 4096 (T = 16) ===
Kernel execution time: 94.250717 ms
=== 4096 x 4096 x 4096 (T = 32) ===
Kernel execution time: 98.670815 ms
=== 8192 x 8192 x 8192 (T = 16) ===
Kernel execution time: 730.594116 ms
=== 8192 x 8192 x 8192 (T = 32) ===
Kernel execution time: 790.019409 ms


Does 4X improvement make sense?
within each tile of 16, native GEMM kernel each thread load 2k elements of A and B, and compute one elements of C. each element of A/B is loaded t times.
with tiling, each thread block load a tile, each element of A/B is loaded one times. if the kernel is memory-bound, the improvement should be t times.
but clearly the kernel is not memory-bound, because the speed up is less than t times.

TBD: performance modeling with roofline model, ncu profiling, etc.