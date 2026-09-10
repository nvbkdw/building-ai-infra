---
title: "GPU Architecture and Roofline Model"
date: 2026-08-26
tags: ["GPU Architecture", "NVIDIA GPU", "Roofline Model"]
author: "Ryan H."
description: "This blog post covers GPU Architecture and Roofline Model."
summary: "This blog post covers GPU Architecture and Roofline Model."
---

# GPU architecture

Comparison between GPU and CPU architecture

CPU: large silicon area to high latency on single thread, 
    * branch prediction
    * instruction level parallelism, out-of-order issues
    * SMT: simutanious multi-threading
    * cache coherence
    * with L1/L2/L3 caching to bring data closer


GPU: massive parallel thread, optimized for throughput, not latency

thread registers, low cost HW-level context swith, warp occupancy is key.
GPU use large parallelism to high mem latency, increase overall throughput.



# Roofline Modeling
Why?
GPU relies on high warp occupancy, HW level context switching to high system latency, deliver on high throughput on overall system. the latency of memory access (throw mem hierarchy) is overshadowed by computation from massive parallel thread/warp. So GPU kernel usually operate at two mode: compute bound, or memory bounded. 


From a single thread, mem access and computation happens sequentially, and the throughput is bound by `mem access latency` + `computation latency`. 

```
a = memA[i] # mem load A
b = memB[i] # mem load B
c = a * b   # computation
memC[i] = c # mem store C
```

However, with massive parralel warps/threads, when one thread waiting on memory access, other thread can run computation. From a whole system perspective, memory access and computation overlapese.


# Reference
Yi Wang blog
zartbot: https://zartbot.github.io/blog/arch/jalapeno/index.html
