---
title: "Quantization"
date: 2025-11-24
tags: ["quantization", "deep learning"]
author: "Ryan H."
description: "This blog post covers the quantization in deep learning."
summary: "This blog post covers the quantization in deep learning."
cover:
    image: "quantization.png"
    alt: "Quantization"
    relative: true
---


Motivation for using smaller number format?
- reduce memory footprint
- reduce memory bandwidth pressure
- enables packing more math units into same silicon area
- consume less energy



### Linear Operations Can Cheaply Deal with Scales

Let:
- $x = (x_i | 1 \leq i \leq k)$ be a vector of reals.
- $x^q$ be the integer-quantized version of $x$.
- $x' = (x_i^q * dscale_x | 1 \leq i \leq k)$ be the approximation of $x$ after dequantization.
- Define $y$ and $y'$ similarly.

Then,

$$
z' = \sum_i x'_i * y'_i = dscale_x * dscale_y \sum_i x_i^q * y_i^q
$$

Most of the math is cheap, only the dequantization at the end is expensive.

what is the effect of quantization on the performance of the model?

- reduce the range, scaling factor clips the value to the range [min, max]
- reduce the precision, rounding the value to the nearest integer

Scaling Factor Granularity:
- per-block quantization: $x^q = (x_{i,j,k}^q | 1 \leq i \leq k, 1 \leq j \leq m, 1 \leq k \leq n)$
- per-tensor quantization: $x^q = (x_i^q | 1 \leq i \leq k)$
- per-channel quantization: $x^q = (x_{i,j}^q | 1 \leq i \leq k, 1 \leq j \leq m)$
- per-layer quantization: $x^q = (x_{i,j,k}^q | 1 \leq i \leq k, 1 \leq j \leq m, 1 \leq k \leq n)$

Quantization Design Space:

- How to choose quantization scheme? (TBD)
    
- How to compensatre for for quantization error?


## Applications in inference, training and RL

### Training
FP16 training, with scaling factor
why scaling factor? otherwise gradient does not fall into the range of FP16

BF16 training

FP8 training??


OCP MX block format:
- metadata + data blocks, with hardware support

Recover accuracy:
AWQ: Activation Weights Quantization
QAT: Quantization aware training


FP4 training?
- MXFP4:
    - 32-element block, E8M0
- NVFP4:
    - 16-element block, E4M3



## Reference

[1] Numerics and AI: https://www.youtube.com/watch?v=ua2NhlenIKo