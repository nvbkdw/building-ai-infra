---
title: "Kernel 2: Shared Memory"
date: 2026-06-14
tags: ["kernel", "shared memory"]
author: "Ryan H."
description: "This blog post covers the shared memory."
summary: "This blog post covers the shared memory."
---

## Shared Memory

We can enable dynamic adjustment of the shared memory size with a different
style of declaration in CUDA by adding the extern keyword in front of the shared
memory declaration and omitting the size of the array in the declaration. Based on
this style, the declarations for Mds and Nds need to be merged into one dynamically
allocated array as follows:
```
extern __shared__ Mds_Nds[];
```

`extern __shared__` declares a dynamically sized shared-memory buffer.
Dynamically sized shared memory size is determined during kernel launch by the third kernel launch parameter `sharedMemBytes`, 

```
kernel<<<grid, block, sharedMemBytes>>>(...);
```

It is allocated once per block before the block runs. When kernel executes, the shared memory size is fixed and does not expand on demand.

Each CUDA thread block gets one dynamically sized shared-memory region for a given kernel launch.

But that does not mean you can only use it as one array. It means CUDA gives the block one contiguous byte buffer, and you can manually split that buffer into as many logical arrays as you want.