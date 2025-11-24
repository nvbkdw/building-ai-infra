---
title: "Math for Deep Learning"
date: 2025-11-19
tags: ["deep learning","math"]
author: "Ryan H."
description: "This blog post covers the math behind deep learning."
summary: "This blog post covers the math behind deep learning."
cover:
    image: "math-for-deep-learning.png"
    alt: "Math for Deep Learning"
    relative: true
---

This blog post provides a refresher on the essential math for deep learning. We will cover the following topics:
- Linear Algebra
- Calculus
- Probability

## Table of Contents
- [Introduction](#introduction)
- [Linear Algebra](#linear-algebra)
  - [Sets, Scalars, Vectors, Matrices, Tensors](#sets-scalars-vectors-matrices-tensors)
    - [Sets](#sets)
    - [Scalars](#scalars)
    - [Vectors](#vectors)
    - [Matrices](#matrices)
    - [Tensors](#tensors)
    - [Transpose](#transpose)
  - [Addition and Multiplication](#addition-and-multiplication)
    - [Addition](#addition)
    - [Multiplication](#multiplication)
  - [Mutiplication](#mutiplication)
    - [Matrix Multiplication Properties](#matrix-multiplication-properties)

## Linear Algebra

### Sets, Scalars, Vectors, Matrices, Tensors

#### Sets
A set $\mathbb{S}$ is a collection of different values. 
- A set can empty, i.e. $\mathbb{S} = \emptyset$.
- A set can contain discrete elements, i.e. $\mathbb{S} = \{red, green, blue\}$.
- A set can contain continuous elements, i.e. $\mathbb{S} = \{x \in \mathbb{R} \}$.

A set defines a "virtual space" in which many mathmatical computation can be performed.

Set properties:
- **Cardinality**: $|\mathbb{S}|$ is the number of elements in a set
- **Union**: $\mathbb{S}_1 \cup \mathbb{S}_2$ is the set of all elements that are in either of the two sets
- **Intersection**: $\mathbb{S}_1 \cap \mathbb{S}_2$ is the set of all elements that are in both of the two sets
- **Cartesian Product**: $\mathbb{S}_1 \times \mathbb{S}_2$ is the set of all ordered pairs of elements from the two sets

#### Scalars
A scalar is a single number. It's call scalar because it often used to scale other values, e.g., vectors, matrices, tensors.

#### Vectors
A vector is a array of numbers with specific order. 
- If $x \in \mathbb{R}$, then $\mathbf{x} = [x_1, x_2, \ldots, x_n] $ is a vector of $n$ real numbers. 
- $\mathbf{x}$ is in set $\mathbb{R}^n$ which is the cartesian product of $\mathbb{R} \times \mathbb{R} \times \ldots \times \mathbb{R}$. 
- We can think of vector $\mathbf{x}$ as a point in an $n$-dimensional space.

#### Matrices
A matrix is a 2D array of numbers. If a real-valued matrics has M rows and N columns, we write $A \in \mathbb{R}^{M \times N}$. 
- $a_{ij}$ is the element in the $i$-th row and $j$-th column of $A$. 
- $\mathbf{a}_{j} = A_{:j}$ is the $j$-th column of $A$. By default, a vector $a_{ij}$ is column vector.
- $\mathbf{a}_{i}^T = A_{i:}$ is the $i$-th row of $A$. Tranpose sign means it represents a row vector.
- Matrix can be thought of as a collection of column vectors, i.e. $\mathbf{A} = [\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_N]$.
- Alternatively, Matrix can be represented as a vector, a.k.a. vectorization. $M \times N = T, A \in  \mathbb{R}^{M \times N} = \mathbb{R}^T$ $\Leftrightarrow$ Matrix $A$ is a point in an $T$-dimensional space.

#### Tensors
A tensor is an array with more than 2 axes, i.e. $A \in \mathbb{R}^{M \times N \times K}$.
- Similar to matrix, tensor can be represented as a vector, a.k.a. vectorization. $M \times N \times K = T, A \in  \mathbb{R}^{M \times N \times K} = \mathbb{R}^T$ $\Leftrightarrow$ Tensor $A$ is a point in an $T$-dimensional space.

#### Transpose
The transpose of a matrix $A$ is a matrix $A^T$ such that $A^T_{ij} = A_{ji}$. Mirroring it at its main diagonal.
- a standard column vector $\mathbf{x}$ can be transposed to a row vector $\mathbf{x}^T$. 
- for scaler, $x = x^T$

### Addition and Multiplication
#### Addition
- Vector or matrix addition is defined as the element-wise addition.
    - **Commutative**: $A + B = B + A$.
    - **Associative**: $(A + B) + C = A + (B + C)$.
- We can also add (or multiply) a scaler to a vector or matrix. This corresponds to performing the operation on each element of the vector or matrix.
    - **Distributive**: $a \times (B + C) = (a \times B) + (a \times C)$.

#### Multiplication
- Matrix multiplication is defined as the dot product of the rows of the first matrix and the columns of the second matrix.
- Matrix multiplication is not commutative, i.e. $A \times B \neq B \times A$.
- Matrix multiplication is associative, i.e. $(A \times B) \times C = A \times (B \times C)$.

In deep learning, we allow addition of a matrix and a vector:

$$
A + \mathbf{b} = [a_{11} + b_1, a_{12} + b_2, \ldots, a_{1N} + b_N]
$$

This is called "broadcasting" by implictly adding a column vector $\mathbf{b}$ to each row of matrix $A$.

Broadcasting rules applies to any tensor $A$ and $B$ with different number of dimensions:
- Compare shapes of tensor A and B, starting from the last shape dimension $dim$ (right to left)
- If $dim_A = dim_B$, move to the next $dim$
- If one of the dimensions is 1, then broadcast the tensor to the same shape as the other tensor
    - If $dim_A \neq dim_B$, and $dim_A$ is 1, then broadcast $A$ to the same shape as $B$
    - If $dim_A \neq dim_B$, and $dim_B$ is 1, then broadcast $B$ to the same shape as $A$
- If $dim_A \neq dim_B$, and $dim_A$ and $dim_B$ are not 1, then error

### Mutiplication
Two matrix can be multiplied if the number of columns of the first matrix equals the number of rows of the second matrix.

For example, if $A \in \mathbb{R}^{M \times N}$ and $B \in \mathbb{R}^{N \times P}$, then $C = A B \in \mathbb{R}^{M \times P}$. 

The **matrix product** if defined as:

$$
C_{ij} = \sum_{k=1}^N A_{ik} B_{kj}
$$

Inner product of two vectors, a.k.a. vector dot product:

$$
\mathbf{x}^T \mathbf{y} = \sum_{i=1}^N x_i y_i
$$

Outer product of two vectors:

$$
\mathbf{x} \mathbf{y}^T = \begin{bmatrix} x_1 y_1 & x_1 y_2 \\ x_2 y_1 & x_2 y_2 \end{bmatrix}
$$

#### Matrix Multiplication Properties
- **Associative**: $(A B) C = A (B C)$.
- **Distributive**: $A (B + C) = A B + A C$.
- **Transpose**: $(A B)^T = B^T A^T$. Hence, the vector dot procut is commutative, i.e. $\mathbf{x}^T \mathbf{y} = \mathbf{y}^T \mathbf{x}$.
- **NOT Commutative**: $A B \neq B A$.