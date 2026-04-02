---
title: GPU DCGM Notes
description: Notes on using the GPU DCGM API
date: 2026-03-20
tags: [gpu, dcgm, notes]
---

# GPU DCGM Notes

`DCGM_FI_DEV_GPU_UTIL` is roughly equal to `DCGM_FI_PROF_GR_ENGINE_ACTIVE`. `DCGM_FI_PROF_GR_ENGINE_ACTIVE` is higher precision and works on MIG.

`DCGM_FI_DEV_MEM_COPY_UTIL` is utilization of the copy engine of the GPU. I would shy away from it as it may not capture all memory bandwidth. Sometimes cuda mem copies use cuda kernels rather than the copy engine, which would not be picked up by this metric. Also, this metric does not work on MIG.

`DCGM_FI_PROF_DRAM_ACTIVE` is dram bandwidth vs theoretical maximum. This metric is accurate and captures all transfers to and from the GPU's DRAM.

For bandwidth, use `DCGM_FI_PROF_DRAM_ACTIVE`. For allocation, use `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE`, `DCGM_FI_DEV_FB_TOTAL`.


Think of GPU SM usage as 3 dimensions of utilization:
`PROF_GR_ENGINE_ACTIVE` - Is any kernel running on any SM?
`PROF_SM_ACTIVE` - What ratio of SMs are active?
`PROF_SM_OCCUPANCY` - How many warps are running vs theoretical max (2048 per SM).

Rough estimate of SMs used would be `PROF_SM_ACTIVE` * `numSMs`.



