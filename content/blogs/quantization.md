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

## Quantization Techniques

### K-Mean Quantization
Quantization by clustering: Use k-means to find the best set of centroids for the given tensor. 
As a result, a tensor is compressed into low-bits index and a cookbook. 



![K-Mean Quantization](/static/k-mean-quantization.png)

Weights are decoded by "cookbook" table lookup. Reduce memory footprint and memory bandwidth pressure, all computation is still in full FP.

![K-Mean Quantization Decode](/static/k-mean-quantization-decode.png)

To recover model accuracy, weights in K-mean quantization are optimized by gradient descent to update the centroids.

![K-Mean Quantization Gradient Descent](/static/k-mean-quantization-backprop.png)

### Linear Quantization


Affine mapping between integer and real numbers:

$$
A \approx (A^q - z ) * s
$$

- $A \in \mathbf{R}^{m \times n}$
- $A^q \in \mathbf{Z}^{m \times n}$ where $Z$ is N-bit integer 
- $z \in \mathbf{Z}$ is the zero-point
- $s \in \mathbf{R}$ is the scaling factor

***How to calcuate $z$ and $s$?***

$r_{min}$ and $r_{max}$ are the minimum and maximum values of the real numbers in the matrix $A$.
$q_{min}$ and $q_{max}$ are the minimum and maximum values of the N-bit integer numbers.

$$
r_{max} = s(q_{max} - z) \\
r_{min} = s(q_{min} - z)
$$

then we have:

$$
s = \frac{r_{max} - r_{min}}{q_{max} - q_{min}}

z = q_{max} - \frac{r_{max}}{s}
\\

$$

$s$ is the ratio between dynamic range of the real numbers to the N-bit integer numbers. 

Given $s$, we can calculate $z$ as follows:

$$
z = q_{max} - \frac{r_{max}}{s}
$$

#### Matrix Multiplication with Linear Quantization

$$
\begin{aligned}
Y &= WX \\
&\Rightarrow s_y(Y^q - z_y) = s_w(W^q - z_w) * s_x(X^q - z_x) \\
&\Rightarrow q_y = \frac{s_w s_x}{s_y} (W^q - z_w)(X^q - z_x) + z_y \\
&\Rightarrow q_y = \frac{s_w s_x}{s_y} (W^q X^q - z_w X^q - z_x W^q  + z_w z_x) + z_y
\end{aligned}
$$


Empirically, weights distribution is symmetric around 0, so $z_w$ is set to 0. Then we simplify the equation as follows:


$$
q_y = \frac{s_w s_x}{s_y} (W^q X^q - z_x W^q) + z_y \\
$$


At inference time, $W^q$ and $z_x$ are constant (pre-determined), thus $z_x W^q$ can be pre-computed. The bulk of runtime computation is the integer matrix multiplication of $W^qX^q$.

Similarly, for matrix product with bias, we have:
$$
\begin{aligned}
Y &= WX + b \\
&\Rightarrow s_y(Y^q - z_y) = s_w(W^q - z_w) * s_x(X^q - z_x) + s_b(\mathbf{b}^q - z_b) \\
&\Rightarrow q_y = \frac{s_w s_x}{s_y} (W^q X^q - z_x W^q) + \frac{s_b}{s_y} (\mathbf{b}^q - z_b) + z_y \\
\end{aligned}
$$

Since weight and bias follows the same distribution, we can force $s_b = s_ws_x$, and $z_b = 0$, then we have:

$$
q_y = \frac{s_w s_x}{s_y} (W^q X^q + B^q) + z_y \\
$$

where $B^q = -z_x W^q + \mathbf{b}^q$ is pre-computed and does not change at runtime.

#### Convolution with Linear Quantization

Convolution is linear operation, so it can be quantized using the same technique as matrix multiplication.

$$
q_y = \frac{s_w s_x}{s_y} (Conv(W^q X^q) + B^q) + z_y \\ 
$$

![Convolution with Linear Quantization](/static/conv-quantization.png)


#### Self-Attention with Linear Quantization ??


Use both weights is stored in integer format, and computation is performed in integer domain.



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



Questions?
- When to use which quantization technique?
- How to make trade-offs between quantization accuracy and performance? on different models?


## Reference

[1] EfficientML.ai, Lecture 5: Quantization, Song Han: https://www.dropbox.com/scl/fi/qc2s9opsa2mnqfithvwz1/Lec05-Quantization-I.pdf

[2] Deep Compression, Song Han, et al.: https://arxiv.org/abs/1510.00149