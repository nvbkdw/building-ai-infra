---
title: "Attention Computation"
date: 2026-03-05
tags: ["attention"]
author: "Ryan H."
description: "This blog post covers the attention computation algorithm."
summary: "This blog post covers the attention computation algorithm."
cover:
    image: "attention-computation.png"
    alt: "Attention Computation Algorithm"
    relative: true
---

# Attention Computation Algorithm


### 1. The Standard Attention Formula (The Problem)

In standard attention, for a given Query matrix $Q$, Key matrix $K$, and Value matrix $V$, the output $O$ is:

$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

To calculate the softmax for a single query token (a single row in $Q$), you must compute the dot product with *every* key token, find the maximum value for numerical stability, exponentiate them, and sum them up to get the denominator. You cannot calculate the true denominator without seeing every Key.

### 2. The Solution: Online Softmax

To distribute this, we need to compute the softmax incrementally as new Keys and Values arrive over the network. We do this by keeping track of two running variables for each query token:

1. **$m$**: The running maximum attention score seen so far.
2. **$l$**: The running sum of the exponentiated scores (the unnormalized softmax denominator).

Let's say GPU 0 is holding a block of Queries ($Q_i$), and it is currently receiving block $j$ of Keys and Values ($K_j, V_j$) from the ring.

**Step A: Compute raw scores for the current block**
GPU 0 computes the unnormalized attention scores for the blocks it currently holds:


$$S_{ij} = \frac{Q_i K_j^T}{\sqrt{d}}$$

**Step B: Update the running maximum ($m$)**
It compares the maximum score in the new block ($m_{ij}$) with its historical maximum ($m^{(old)}$) to find the new global maximum:


$$m^{(new)} = \max(m^{(old)}, \max(S_{ij}))$$

**Step C: Update the running denominator ($l$)**
Because the maximum changed, the previous exponentiated sum ($l^{(old)}$) is now mathematically "out of date." We rescale the old sum using the difference between the old and new max, and add the sum of the new block:


$$l^{(new)} = l^{(old)} \cdot \exp(m^{(old)} - m^{(new)}) + \sum \exp(S_{ij} - m^{(new)})$$

**Step D: Update the Output block ($O_i$)**
Finally, we update the actual Output matrix. We rescale the old output, add the contribution from the new $V_j$ block, and divide by the new denominator:


$$O_i^{(new)} = \frac{l^{(old)} \cdot \exp(m^{(old)} - m^{(new)}) \cdot O_i^{(old)} + \exp(S_{ij} - m^{(new)}) V_j}{l^{(new)}}$$


## Ring Attention
Diving into the math behind Ring Attention is where things get really elegant. It solves a fundamental problem: standard Self-Attention requires seeing the *entire* sequence at once to compute the probabilities (the softmax denominator).

If GPU 0 only has tokens 0–48k, and GPU 1 has tokens 48k–96k, GPU 0 cannot compute the final attention scores for its tokens because it doesn't know how strongly they attend to GPU 1's tokens.

Ring Attention solves this using **Blockwise Computation** and a mathematical trick called **Online Softmax** (the same math that powers FlashAttention). Here is how the math unfolds during training.

### 3. The Ring Communication (Forward Pass)

Now that we have the math to update attention incrementally, we put the GPUs in a logical ring.

1. **Initialize:** Every GPU starts with its local chunk of $Q$, $K$, and $V$. It initializes $O = 0$, $m = -\infty$, and $l = 0$.
2. **Local Compute:** Every GPU computes the Online Softmax steps (above) using its local $Q$ and local $K, V$.
3. **Pass the Baton:** Every GPU simultaneously sends its local $K$ and $V$ to the *next* GPU in the ring, while receiving a new $K$ and $V$ from the *previous* GPU.
4. **Repeat:** The GPUs compute the Online Softmax update using their stationary local $Q$ and the newly arrived $K, V$.
5. **Finish:** After $C$ steps (where $C$ is the number of CP ranks), the $K$ and $V$ blocks have made a full circle. Every GPU's local $O_i$ matrix is mathematically exact, identical to if it had been computed on a single giant GPU.

### 4. The Backward Pass (Training Dynamics)

During training, you have to run this ring in reverse to compute gradients. This is where Ring Attention gets computationally intense.

To compute the gradients $dQ$, $dK$, and $dV$, the GPUs must form **two concurrent rings**:

1. **The Forward-Recompute Ring:** The GPUs pass $K$ and $V$ around the ring again to recompute the attention probabilities ($P$), because storing a 100K x 100K attention matrix would instantly OOM the GPU.
2. **The Gradient Ring:** At the same time, the GPUs pass the gradient of the keys ($dK$) and values ($dV$) around the ring. As GPU 0 computes the local gradients for the blocks it is currently holding, it accumulates those updates into the circulating $dK$ and $dV$ blocks.

By the time $dK$ and $dV$ complete their circle, they contain the full, exact gradients summed across the entire sequence, and the optimizer can step the model weights.

---

This combination of **Online Softmax** + **P2P Ring Communication** is a masterpiece of systems-level math, allowing sequence lengths to scale linearly with the number of GPUs you throw at them.

Would you like to look closer at how gradients ($dQ$, $dK$, $dV$) are mathematically derived in blockwise attention, or shift gears to look at how FSDP interacts with this CP mesh?





# References

- [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf). University of Washington CSE 599m, Spring 2023.
- Milakov, M., & Gimelshein, N. (2018). [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867). *arXiv preprint arXiv:1805.02867*.