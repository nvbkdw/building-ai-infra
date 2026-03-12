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


## Attention Formula (The Problem)

In standard attention, for a given Query matrix $Q$, Key matrix $K$, and Value matrix $V$, the output $O$ is:

$$O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

Due to the softmax function, we cannot compute the true denominator without seeing every Key.


### 1. The Naive Softmax

The standard softmax function converts a vector of raw scores (logits), $x$, into a probability distribution. For the $i$-th element, the formula is:

$$S(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$$

**The Problem:** The exponential function grows incredibly fast. If any element in $x$ is large (e.g., $x_i = 100$), $\exp(100)$ will cause a numerical overflow, resulting in `NaN` (Not a Number) in standard floating-point arithmetic.

---

### 2. Shift Invariance and Stable Softmax

To fix the overflow issue, we can exploit a mathematical property of softmax: **shift invariance**. Adding or subtracting a scalar constant $c$ from all input elements does not change the softmax output:

$$S(x-c)_i = \frac{\exp(x_i - c)}{\sum_j \exp(x_j - c)} = \frac{\exp(x_i)\exp(-c)}{\sum_j \exp(x_j)\exp(-c)} = \frac{\exp(-c) \cdot \exp(x_i)}{\exp(-c) \cdot \sum_j \exp(x_j)} = S(x)_i$$

**The Solution:** If we set $c$ to be the maximum value in our vector ($m = \max(x)$), then the largest value we will ever pass to the exponential function is $0$ (since $x_i - m \le 0$). $\exp(0) = 1$, strictly preventing overflow.

This gives us the **Stable Softmax**:


$$S(x)_i = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}$$

---

### 3. Defining Log-Sum-Exp (LSE)

In attention algorithms, we need to keep track of the denominator so we can combine chunks of data later. Storing the raw sum of exponentials is risky because it can still grow very large. Instead, we store it in log-space.

Let the denominator be $Z = \sum_j \exp(x_j)$.
To calculate this safely using our stable trick, we factor out the max value $m$:


$$Z = \exp(m) \cdot \sum_j \exp(x_j - m)$$

Taking the natural logarithm of both sides gives us the **Log-Sum-Exp (LSE)**:


$$LSE = \log(Z) = \log\left(\exp(m) \cdot \sum_j \exp(x_j - m)\right)$$

$$LSE = m + \log\left(\sum_j \exp(x_j - m)\right)$$

---

### 4. Rewriting Stable Softmax with LSE

Now that we have defined LSE as the log of the true denominator, we can rewrite the entire softmax calculation in a very clean way.

Since $Z = \exp(LSE)$, the softmax for element $i$ becomes:


$$S(x)_i = \frac{\exp(x_i)}{Z} = \frac{\exp(x_i)}{\exp(LSE)}$$

Using the exponent rule $\frac{\exp(A)}{\exp(B)} = \exp(A - B)$, we get:


$$S(x)_i = \exp(x_i - LSE)$$

---

### 5. Deriving Block-wise Softmax

If our input $x$ is too large to fit in memory, we split it into blocks. Let's say we have processed Block 1 and are now processing Block 2.

* **Block 1** has local output $S^{(1)}$ and local denominator log $LSE_1$.
* **Block 2** has local output $S^{(2)}$ and local denominator log $LSE_2$.

To combine them, we need a **new total denominator** ($LSE_{new}$) and **updated outputs**.

**Step 5a: Updating the LSE**
The combined raw denominator is the sum of the two block denominators: $Z_{new} = Z_1 + Z_2$.
In log-space:


$$LSE_{new} = \log(\exp(LSE_1) + \exp(LSE_2))$$

To prevent overflow here, we factor out $\exp(LSE_1)$:


$$LSE_{new} = \log\left(\exp(LSE_1) \cdot \left(1 + \frac{\exp(LSE_2)}{\exp(LSE_1)}\right)\right)$$

$$LSE_{new} = LSE_1 + \log(1 + \exp(LSE_2 - LSE_1))$$


**Step 5b: Updating the Softmax Outputs**
Our old output for Block 1 was $S^{(1)} = \frac{\exp(x^{(1)})}{\exp(LSE_1)}$.
It needs to be corrected to reflect the new global denominator: $S_{new}^{(1)} = \frac{\exp(x^{(1)})}{\exp(LSE_{new})}$.

We can isolate $\exp(x^{(1)})$ from the first equation: $\exp(x^{(1)}) = S^{(1)} \cdot \exp(LSE_1)$.
Substitute this into the target equation:


$$S_{new}^{(1)} = \frac{S^{(1)} \cdot \exp(LSE_1)}{\exp(LSE_{new})}$$

$$S_{new}^{(1)} = S^{(1)} \cdot \exp(LSE_1 - LSE_{new})$$

By applying the exact same logic to Block 2, we get:


$$S_{new}^{(2)} = S^{(2)} \cdot \exp(LSE_2 - LSE_{new})$$

**The Final Update Formula:**
To combine an old accumulated block with a new incoming block, the combined output is:


$$Out_{new} = Out_{old} \cdot \exp(LSE_{old} - LSE_{new}) + Out_{block} \cdot \exp(LSE_{block} - LSE_{new})$$


## Ring Attention
Diving into the math behind Ring Attention is where things get really elegant. It solves a fundamental problem: standard Self-Attention requires seeing the *entire* sequence at once to compute the probabilities (the softmax denominator).

If GPU 0 only has tokens 0–48k, and GPU 1 has tokens 48k–96k, GPU 0 cannot compute the final attention scores for its tokens because it doesn't know how strongly they attend to GPU 1's tokens.

Ring Attention solves this using **Blockwise Softmax Computation** described in the previous section.

![Ring Attention Algorithm](/static/ring-attention-algo.png)

### 1. The Standard Attention Formula

In standard attention, we compute the output for a set of queries ($Q$), keys ($K$), and values ($V$). For a single query vector $q$, the attention output $O$ is a weighted sum of the value vectors.

Let $x$ be the raw attention scores before softmax: $x = q K^T$.
The final output vector $O$ is the softmax of $x$ multiplied by the Values matrix $V$:


$$O = \text{softmax}(x) V$$

If we write this as a summation over all $N$ tokens in the sequence, where $v_i$ is the $i$-th row of $V$:


$$O = \sum_{i=1}^N \frac{\exp(x_i)}{Z_{total}} v_i$$


*(Remember from our previous section that $Z_{total} = \exp(LSE_{total})$, the global denominator).*

---

### 2. Deconstructing into Blocks

In Ring Attention, the sequences are too long to fit onto a single GPU. So, $Q$, $K$, and $V$ are distributed across multiple GPUs.

A specific GPU holds a block of Queries ($Q_{local}$) and a running accumulation of the Output ($O_{old}$) and Log-Sum-Exp ($LSE_{old}$).
While holding its $Q_{local}$ stationary, the GPU receives blocks of $K$ and $V$ from its neighbor in a "ring."

Let's say the GPU just received a new block of Keys ($K_{block}$) and Values ($V_{block}$).
It computes the local raw scores $x_{block} = Q_{local} K_{block}^T$.

Now, we compute the local attention output for just this block:


$$O_{block} = \sum_{j \in block} \frac{\exp(x_j)}{\exp(LSE_{block})} v_j$$

---

### 3. The Un-normalization Step (The "Ah-ha!" Moment)

We now have an old output ($O_{old}$) and a new block output ($O_{block}$). We cannot simply add them together because they were divided by different denominators ($LSE_{old}$ and $LSE_{block}$).

To combine them correctly, we need to mathematically "undo" their local divisions to get the raw, un-normalized weighted sums back.

From the formula above, the un-normalized sum of the old values is:


$$\text{Raw}_{old} = \sum_{i \in old} \exp(x_i) v_i = O_{old} \cdot \exp(LSE_{old})$$

And the un-normalized sum of the new block's values is:


$$\text{Raw}_{block} = \sum_{j \in block} \exp(x_j) v_j = O_{block} \cdot \exp(LSE_{block})$$

---

### 4. Combining and Re-normalizing

The true, combined new output ($O_{new}$) must be the sum of *all* the raw values, divided by the *new* combined denominator ($LSE_{new}$, which we calculated in Step 5a of the previous tutorial).

$$O_{new} = \frac{\text{Raw}_{old} + \text{Raw}_{block}}{\exp(LSE_{new})}$$

Let's substitute our formulas from Step 3 into this equation:


$$O_{new} = \frac{O_{old} \cdot \exp(LSE_{old}) + O_{block} \cdot \exp(LSE_{block})}{\exp(LSE_{new})}$$

Now, we separate the fraction and use the exponent division rule ($\frac{\exp(A)}{\exp(B)} = \exp(A - B)$):


$$O_{new} = O_{old} \cdot \frac{\exp(LSE_{old})}{\exp(LSE_{new})} + O_{block} \cdot \frac{\exp(LSE_{block})}{\exp(LSE_{new})}$$

$$O_{new} = O_{old} \cdot \exp(LSE_{old} - LSE_{new}) + O_{block} \cdot \exp(LSE_{block} - LSE_{new})$$

### Why this is brilliant for Ring Attention

Because of this math, a GPU in a Ring Attention setup never needs to see the full $K$ or $V$ matrix at once.

1. It computes $O_{block}$ and $LSE_{block}$ using whatever $K$ and $V$ it currently holds.
2. It uses this update formula to fold that data into its running `out` and running `lse`.
3. It passes the $K$ and $V$ block to the next GPU in the ring and receives a new block.
4. It repeats the process.

The exponents $(\dots - LSE_{new})$ act as **correction weights**. Because $LSE_{new}$ is guaranteed to be larger than or equal to both $LSE_{old}$ and $LSE_{block}$, the exponents are always negative or zero. This ensures the weights are always between **0 and 1**, safely scaling down the older accumulations as new, highly-attended tokens are discovered, completely preventing numerical overflow.

### Causal Masking
How causal masking (where tokens can't look at future tokens) is applied mathematically when computing these blocks in a ring topology?

Adding causal masking into the block-wise Ring Attention setup is a great next step. Causal masking ensures that when a model is predicting the next word, it cannot "cheat" by looking at future words.

Mathematically, this means token $i$ can only attend to token $j$ if $j \le i$. Let's break down how this is applied within the distributed block topology of Ring Attention.

#### 1. The Math of Causal Masking

In standard attention, we calculate the raw scores $x_{ij} = q_i k_j^T$.
To apply a causal mask, we force the attention weight for any future token ($j > i$) to be zero. However, we must apply this mask **before** the exponential in the softmax function.

If we simply set the raw score to $0$, $\exp(0) = 1$, which means the model would still pay attention to it! Instead, we set the raw score to negative infinity ($-\infty$):

$$x_{ij} = 
\begin{cases} 
q_i k_j^T & \text{if } j \le i \\
-\infty & \text{if } j > i 
\end{cases}$$

Because $\exp(-\infty) = 0$, these future tokens are completely zeroed out in the numerator, and they contribute exactly $0$ to the local denominator ($LSE$).

---

#### 2. Causal Masking in a Block Topology

In Ring Attention, the sequence is divided into chunks. Let's say your GPU holds a Query block ($Q_{block}$) that contains tokens 100 to 199.

As the Keys ($K$) and Values ($V$) circulate through the ring of GPUs, your GPU will encounter three distinct types of blocks:

**Case A: Past Blocks (e.g., $K, V$ for tokens 0–99)**
Since all the keys in this block happened *before* all the queries in your GPU's block, every query can attend to every key.

* **Math:** No masking is needed. You compute $x_{block} = Q_{local} K_{block}^T$ and use the standard update formula we derived earlier.

**Case B: Future Blocks (e.g., $K, V$ for tokens 200–299)**
Since all the keys in this block happen *after* all the queries in your GPU's block, no query is allowed to attend to any key here.

* **Math:** The entire block is masked to $-\infty$.
* **Optimization:** In practice, Ring Attention implementations are smart enough to recognize this and simply **skip the computation** for future blocks altogether. No math is performed, saving massive amounts of compute time.

**Case C: The Diagonal Block (e.g., $K, V$ for tokens 100–199)**
This is where the $Q$ block and the $K, V$ block overlap (they cover the exact same token indices). This requires **partial masking**.

* Token 100 can only look at key 100.
* Token 150 can look at keys 100 through 150, but not 151–199.
* **Math:** We compute the full $x_{block} = Q_{local} K_{block}^T$ and then apply a lower-triangular mask, setting everything above the diagonal to $-\infty$.

---

#### 3. How the Mask Affects the Block Update Formula

Let's look at what happens to our $LSE$ and Output formulas when a token score is masked to $-\infty$ during the "Diagonal Block" calculation.

Recall the local LSE calculation for the block:


$$LSE_{block} = m + \log\left(\sum_{j \in block} \exp(x_j - m)\right)$$

If a token $j$ is in the future, $x_j = -\infty$. Therefore, $\exp(-\infty - m) = 0$.
This means masked tokens contribute exactly $0$ to the sum. The $LSE_{block}$ only represents the sum of the valid, past tokens.

Similarly, recall the local Output calculation:


$$O_{block} = \sum_{j \in block} \frac{\exp(x_j)}{\exp(LSE_{block})} v_j$$

For future tokens where $x_j = -\infty$, the weight is $\frac{0}{\exp(LSE_{block})} = 0$. The value vector $v_j$ is multiplied by $0$ and completely excluded from the local output.

When this mathematically sound, masked $O_{block}$ and $LSE_{block}$ are fed into our brilliant Ring Attention update formula:


$$O_{new} = O_{old} \cdot \exp(LSE_{old} - LSE_{new}) + O_{block} \cdot \exp(LSE_{block} - LSE_{new})$$


...it merges seamlessly. The formula doesn't even need to know that a mask was applied; the zeros are already perfectly baked into $O_{block}$ and $LSE_{block}$.


### The Backward Pass (Training Dynamics)

During training, you have to run this ring in reverse to compute gradients. This is where Ring Attention gets computationally intense.

To compute the gradients $dQ$, $dK$, and $dV$, the GPUs must form **two concurrent rings**:

1. **The Forward-Recompute Ring:** The GPUs pass $K$ and $V$ around the ring again to recompute the attention probabilities ($P$), because storing a 100K x 100K attention matrix would instantly OOM the GPU.
2. **The Gradient Ring:** At the same time, the GPUs pass the gradient of the keys ($dK$) and values ($dV$) around the ring. As GPU 0 computes the local gradients for the blocks it is currently holding, it accumulates those updates into the circulating $dK$ and $dV$ blocks.

By the time $dK$ and $dV$ complete their circle, they contain the full, exact gradients summed across the entire sequence, and the optimizer can step the model weights.

---

This combination of **Online Softmax** + **P2P Ring Communication** is a masterpiece of systems-level math, allowing sequence lengths to scale linearly with the number of GPUs you throw at them.


TODO: Ring Attention algorithm in pytorch


TODO: overlapping computation and communication in ring attention

TODO: Ring Attention kernels ??? 


TODO: Ulysses Context Parallelism


TODO: how to fuse pre-attention operations into context parallel

TODO: Halix - context parallel for decoding
https://arxiv.org/abs/2507.07120




# References

- [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf). University of Washington CSE 599m, Spring 2023.
- Milakov, M., & Gimelshein, N. (2018). [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/1805.02867). *arXiv preprint arXiv:1805.02867*.
- [Lecture 13: Ring Attention](https://www.youtube.com/watch?v=ws7angQYIxI). GPU MODE