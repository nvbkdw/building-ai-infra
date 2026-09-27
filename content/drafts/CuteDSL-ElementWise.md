---
title: "Elementwise Kernels in CuteDSL"
date: 2026-09-09
tags: ["GPU kernel", "LLM", "Elementwise kernel", "CuteDSL"]
author: "Ryan H."
description: "This blog post covers Elementwise Kernels in CuteDSL"
summary: "This blog post covers Elementwise Kernels in CuteDSL."
---







## vectorized LDS STORE

assumed_align changes the generated memory instructions, so the performance difference is real.

  It tells the compiler the pointer’s alignment in bytes; it does not move or realign the tensor. Without a hint, FP16 defaults to only 2-byte assumed alignment. NVIDIA documentation

  Your kernel assigns 8 consecutive FP16 elements (16 bytes) per thread. I inspected the generated PTX and SASS:

   assumed_align    Load/store width    Memory instructions per thread¹    Effective GB/s, current size    Effective GB/s, larger size
  ━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Default / 2               16 bits                                 24                             269                            228
  ───────────────  ──────────────────  ─────────────────────────────────  ──────────────────────────────  ─────────────────────────────
   4                         32 bits                                 12                             456                            237
  ───────────────  ──────────────────  ─────────────────────────────────  ──────────────────────────────  ─────────────────────────────
   8                         64 bits                                  6                             441                            241
  ───────────────  ──────────────────  ─────────────────────────────────  ──────────────────────────────  ─────────────────────────────
   16                       128 bits                                  3                             429                            241
  ───────────────  ──────────────────  ─────────────────────────────────  ──────────────────────────────  ─────────────────────────────
   32                       128 bits                                  3                             458                            239

  ¹Combined loads from A/B and stores to C. All variants passed correctness checks; timings are medians of repeated batches.

  For example, assumed_align=16 produces just:

  LDG.E.128   // eight FP16 values from A
  LDG.E.128   // eight FP16 values from B
  STG.E.128   // eight FP16 results to C

  Two effects explain the results:

  - Vectorization reduces instruction overhead. Without the hint, each thread executes many narrow memory operations.
  - Your current working set is 22.5 MiB, still below the GPU’s 24 MiB L2. Repeated accesses benefit from caching. Increasing it to 180 MiB brings all variants below the 273 GB/s memory specification, and the alignment benefit becomes much smaller.

  Wider instructions therefore don’t guarantee proportionally higher bandwidth—or strictly increasing timings across alignment settings.

  For this kernel, assumed_align=16 is sufficient to enable the full per-thread vector width. Verify the promise when creating the views:

  assert all(x.data_ptr() % 16 == 0 for x in (a, b, c))
  a_, b_, c_ = [from_dlpack(x, assumed_align=16) for x in (a, b, c)]

  A sliced tensor can have a different alignment, so check its actual pointer.


Almost—but `assumed_align=16` does **not create or realign the allocation**. It tells the CuTe compiler:

> “I, the caller, guarantee that this tensor’s base address is divisible by 16.”

Formally:

```python
tensor.data_ptr() % 16 == 0
```

`from_dlpack` is zero-copy, so it wraps the existing PyTorch allocation. The alignment becomes part of the generated pointer type, enabling 128-bit vectorized loads/stores; responsibility for satisfying it remains with the caller. [NVIDIA’s CuTe DSL documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/framework_integration.html)

In this notebook:

```python
a = torch.randn(..., dtype=torch.float16)
a_ = from_dlpack(a, assumed_align=16)
```

CuTe assumes `a`’s existing base pointer is 16-byte aligned. Since eight `float16` elements occupy \(8 \times 2 = 16\) bytes, this supports loading eight elements as one aligned vector, which is exactly what the notebook’s `(1, 8)` tiling is designed for. [Notebook source](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/notebooks/elementwise_add.ipynb)

You can verify the promise:

```python
assert a.data_ptr() % 16 == 0
assert b.data_ptr() % 16 == 0
assert c.data_ptr() % 16 == 0
```

Be careful with views:

```python
x = a.flatten()[1:]  # starts 2 bytes after a for float16
print(x.data_ptr() % 16)  # likely 2
```

Using `from_dlpack(x, assumed_align=16)` would make a false promise. Also, base-pointer alignment alone does not guarantee that every sliced address is aligned; strides and offsets must preserve the 16-byte alignment too.

so `assumed_align` is only a hint on tensor base address ptr, not for item alignment inside a tensor, is that correct?

Yes—with one wording correction: `assumed_align` is a **compiler contract**, not merely an optional hint.

```python
from_dlpack(a, assumed_align=16)
```

promises that:

```python
a.data_ptr() % 16 == 0
```

It does not promise that every element is 16-byte aligned. An element’s address is:

\[
\text{address} = \text{base} + \text{offset}
\]

A derived address remains 16-byte aligned only when its byte offset is a multiple of 16.

For `float16`:

- Each element is 2 bytes.
- Element 0 starts at a 16-byte boundary.
- Element 1 starts at base + 2, so it is not 16-byte aligned.
- Elements 0, 8, 16, … start at 16-byte boundaries.

That is why the notebook groups eight `float16` values:

```python
8 elements × 2 bytes = 16 bytes
```

The vector’s starting address is aligned; the eight individual elements do not each need 16-byte alignment. Strides, slices, and storage offsets must also preserve alignment for every vector access.


# TV layouts
