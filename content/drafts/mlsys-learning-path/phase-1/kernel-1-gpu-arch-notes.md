---
title: "Kernel 1: GPU Architecture"
date: 2026-06-06
tags: ["kernel", "gpu", "architecture"]
author: "Ryan H."
description: "This blog post covers the GPU architecture."
summary: "This blog post covers the GPU architecture."
---

## GPU Architecture

### Block waves within a grid
Grid is divided into blocks, which is further divided into threads. 
All threads in a block are scheduled to the same **Streaming Multiprocessor (SM)**.

Depends on HW resources, multiple blocks can be scheduled to the same SM at the same time.
However, due to limited SMs, not all blocks in a grid can be scheduled at the same time.
Blocks in a grid are scheduled in **waves**, which is a group of blocks that are scheduled at the same time.

### Warps within a block
Threads in a block are further divided into **warps** (like wave of threads), which is a unit of scheduling and execution.

An SM is designed to execute all threads in a warp following the Single Instruction, Multiple Data (SIMD) model. That is, at any instant in time, one instruction is
fetched and executed by all threads in the warp (see “Warps and SIMD Hardware”
sidebar). For this purpose, an SM is organized into multiple **processing blocks** and
the warps on an SM are distributed across the processing blocks. 

As a real example, the Hopper H100 SM, which has 128 streaming processors,
is organized into four processing blocks with 32 streaming processors each. Thus, a warp has 32 threads.

Threads in the same warp are assigned to the same processing block which fetches
the instruction for the warp and executes it for all threads in the warp. These threads apply the same instruction to their portions of the data. Because the CUDA thread
scheduling mechanism maps warps to the processing blocks and effectively restricts
all threads in a warp to execute the same instruction at any point in time (SIMD). 

The advantage of SIMD is that the cost of the control hardware, such as the
instruction fetch/dispatch unit, is shared across many execution units. 

**Warp programming**: Because threads in the same warp have a special relationship in the way that they
are scheduled, CUDA provides a collection of warp-level primitives, i.e. API func-
tions, that expose efficient data exchange and synchronization mechanisms across
threads in a warp. Programmers can use these warp-level primitives to improve the
speed and efficiency of computations that involve collaboration between threads.

### Control Divergence within a Warp


When threads in the same warp follow different control flow paths, we say that
these threads exhibit control divergence, i.e., they diverge in their execution control
flow paths. When threads within a warp take different control flow paths, the SIMD hardware will take multiple passes
through these paths, one pass for each path. The multi-pass approach to divergent warp execution extends the SIMD
hardware’s ability to implement the full semantics of CUDA threads.

This preserves the independence of threads while taking advantage of the reduced cost of SIMD hardware.
The cost of divergence, however, is the extra passes the hardware needs to take
in order to allow different threads in a warp to make their own decisions, as well as
the execution resources that are consumed by the inactive threads in each pass.

One can determine if a control construct can result in thread divergence by in-
specting its decision condition. If the decision condition is based on threadIdx values,
the control statement can potentially cause thread divergence. One common case is boundary condition control divergence.


### Wrap scheduling

SM contains multiple processing blocks, with each processing block executing one instruction for a warp at a time. 
With multiple processing blocks, SM can schedule multiple warps at the same time. 
However, there are usually more threads/warps assigned to an SM than streaming processors in the SM. 
At any point in time, the hardware can execute instructions only for a subset of all warps assigned to the SM.

Why we need to have so many warps assigned to an SM if it can only
execute a subset of them at any instant? ---> The answer is that this is how GPUs tolerate
long-latency operations such as global memory accesses.

When an instruction to be executed by a warp needs to wait for the result of a
previously initiated long-latency operation, the warp is not selected for execution.
Instead, another resident warp that is no longer waiting for results of previous instructions will be selected for execution. 
Warps that are not waiting for the results
of previous instructions are ready for execution. If more than one warp is ready for
execution, a priority logic is used to select one for execution. The ability for each
SM to select different ready warps for execution at each instant is called **fine-grained
multithreading**. and allows SMs to use the latency time of instructions from some
threads to execute instructions from other threads.

Context switching on traditional CPUs incurs such idle cycles because switching
the execution from one thread to another requires saving the execution state
(such as register contents of the out-going thread) to memory and loading the
execution state of the incoming thread from memory. GPU SMs achieves zero-
overhead scheduling by holding all the execution states for the assigned warps
in the hardware registers so there is no need to save and restore states when
switching from one warp to another.

### Resource partitioning and occupancy
The ratio of the number of warps/threads assigned to an SM to the maximum number it supports is referred to as **occupancy**.

Besides the streaming processors, the execution resources in an SM include:
* registers, 
* shared memory, 
* block slots,
* thread slots.
These resources are **dynamically partitioned** across threads to support their execution.
All the dynamically partitioned resources interact with each other in a complex manner. 
Saturation of one resource will affect the occupancy of other resources.

For example, if the registers are saturated, the occupancy will be affected. 
For example, the H100 GPU allows a maximum of 65,536
registers per SM. To run at full occupancy, each SM needs enough registers for
2048 threads which means that each thread should not use more than (65,536 reg-
isters)/(2048 threads) = 32 registers per thread. If, for example, a kernel uses 64
registers per thread, the maximum number of threads that can be supported with
65,536 registers is 1024 threads. In this case, the occupancy will be at
most 50%.




