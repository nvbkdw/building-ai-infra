---
title: "Math for Deep Learning - Part 1"
date: 2025-11-19
tags: ["deep learning","math"]
author: "Ryan H."
description: "This blog post covers the math behind deep learning - part 1."
summary: "This blog post covers the math behind deep learning - part 1."
cover:
    image: "math-for-deep-learning.png"
    alt: "Math for Deep Learning - Part 1"
    relative: true
---

This blog post provides a refresher on the essential math for deep learning. We will cover the following topics:
- Part 1: Linear Algebra 
- Part 2: Calculus
- Part 3: Probability

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
- By default, a vector $a_{i}$ is column vector. $a_{i}^T$ with tranpose sign means it represents a row vector.

#### Matrices
A matrix is a 2D array of numbers. If a real-valued matrics has M rows and N columns, we write $A \in \mathbb{R}^{M \times N}$. 
- $a_{ij}$ is the element in the $i$-th row and $j$-th column of $A$. 
- $\mathbf{a}_{j} = A_{:j}$ is the $j$-th column of $A$. 
- $\mathbf{a}_{i}^T = A_{i:}$ is the $i$-th row of $A$. 
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

### Identity and Inverse Matrix

Matrix-vector product $A \mathbf{x}$ can be thought of as a linear transformation. If the transformation does not change the vector, it is called the identity matrix $I$.

$$
I \mathbf{x} = \mathbf{x}
$$

The identity matrix is a square matrix with ones on the main diagonal and zeros elsewhere.

$$
I = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

The matrix inverser $A^{-1}$ is a matrix that when multiplied by $A$ gives the identity matrix.

$$
A A^{-1} = I
$$

The matrix inverse is not always exist. If it exists, it is unique.

$$
A^{-1} = \frac{1}{\det(A)} \text{adj}(A)
$$

The matrix inverse is not always exist. More on this later.

### Linear Dependence and Span

Matrix-vector product $A \mathbf{x} = b$ is a compact representation of system of linear equations:

$$
a_1^Tx=b_1; \quad a_2^Tx=b_2; \quad \ldots; \quad a_n^Tx=b_n
$$

It is possible for $A \mathbf{x} = b$ to have 0, 1, or infinite solutions.


We define **span** of a set of vectors as the set of all points obtained by linear combination of the vectors. 

If we break $A$ into column vectors, $A = [\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_n]$, then $A \mathbf{x} = \sum_{i=1} x_i \mathbf{a}_i$ is a span of the column vectors, a.k.a. the *column space* or *range* of $A$.

If $b$ is in the column space of $A$, then $A \mathbf{x} = b$ has a solution.

If the columns are linearly independent, then the column space is a line. 
- If $b$ is not on this line, there is no solution.
- If $b$ is on this line, there are infinite solutions.

The **rank** of a matrix is the number of linearly independent columns. For $A \in \mathbb{R}^{N \times N}$, to have its span emcompass full space of $\mathbb{R}^N$, it must have $N$ linearly independent columns, a.k.a. **full rank**. If any two columns are linearly dependent, the matrix is called singular.

Every square matrix with full rank is invertible.

### Norms

We define **norm** of a vector as a measure of its size. 
A norm function $ f: \mathbb{R}^N \to \mathbb{R}$ satisfies the following properties:
- **Non-negativity**: $f(\mathbf{x}) \geq 0$
- **Zero**: $f(\mathbf{x}) = 0$ if and only if $\mathbf{x} = 0$
- **Homogeneity**: $f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x})$
- **Triangle inequality**: $f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y})$



The $\ell_p$ norm of a vector $\mathbf{x}$ is defined as:

$$
\|\mathbf{x}\|_p = \left( \sum_{i=1}^N |x_i|^p \right)^{\frac{1}{p}}
$$


The most common norms are:
- $\ell_1$ norm: $\|\mathbf{x}\|_1 = \sum_{i=1}^N |x_i|$
- $\ell_2$ norm: $\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^N x_i^2}$

- $\ell_\infty$ norm: $\|\mathbf{x}\|_\infty = \max_{i=1}^N |x_i|$

The dot product of two vectors $\mathbf{x}$ and $\mathbf{y}$ can be written in term of norms:

$$
\mathbf{x}^T \mathbf{y} = \sum_{i=1}^N x_i y_i = \|\mathbf{x}\|_2 \|\mathbf{y}\|_2 \cos(\theta)
$$

where $\theta$ is the angle between the two vectors.

The "size" of a matrix can be measured by the Frobenius norm:

$$
\|\mathbf{x}\|_F = \sqrt{\sum_{i=1}^N x_i^2}
$$


The Frobenius norm is a special case of the $\ell_2$ norm for matrices.

### Special Vector and Matrix

**Diagonal Matrix** - non-zero elements are only on the main diagonal:
$$
D = diag(\mathbf{d}) = \begin{bmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{bmatrix}
$$

* Multiplying with diagnal matrics $diag(\mathbf{d})\mathbf{x} = \mathbf{d} \odot \mathbf{x}$
* Inverting diagnal matrics $diag(\mathbf{d})^{-1} = diag((1/d_1, 1/d_2, 1/d_3, \ldots, 1/d_N)^T)$

**Symmetric Matrix** - matrix that is equal to its transpose:
$$
A = A^T
$$

**Unit vector** - vector with norm 1:
$$
\|\mathbf{x}\|_2 = 1
$$

A vector $\mathbf{x}$ is orthogonal to another vector $\mathbf{y}$ if their dot product is 0:
$$
\mathbf{x}^T \mathbf{y} = 0
$$

A set of unit vectors that are orthogonal to each other are called **orthonormal**.

A orthogonal matrix is a square matrix whose rows/columns are orthonormal.
$$
A^T A = A A^T = I \Rightarrow A^T = A^{-1}
$$


### Eigenvalues and Singular Value Decomposition

Many math problems can be solve by breaking them into parts. An **eigendecomposition** is a way to break a matrix into its eigenvalues and eigenvectors.

An **eigenvector** of square matrix $A$ is a non-zero vector $\mathbf{v}$ such that matrix-eigenvector product only alters its scale, known as **eigenvalue**. 

$$
A \mathbf{v} = \lambda \mathbf{v}
$$

**Eigenvalue Decomposition (EVD)**:
$$
A = V \Lambda V^{-1}
$$

where $Q$ is the matrix of eigenvectors and $\Lambda$ is the diagonal matrix of eigenvalues.

A real symmetric matrix $A$ can be decomposed into:

$$A = Q \Lambda Q^T$$

Where $Q$ is an orthogonal matrix of eigenvectors and $\Lambda$ is a diagonal matrix of eigenvalues. This can be view as three step transformation:

- $Q^T$ rotation on to the eigenvectors space
- $\Lambda$ scaling along the eigenvectors
- $Q$ rotation back to the original space

This can be view as $A$ distorts space along the eigenvectors, and scales the magnitude by the eigenvalues.

The **rank** of a matrix = the number of non-zero eigenvalues.

**Singular Value Decomposition (SVD)**:
Eigenvalue decomposition is only defined for square matrices. SVD is a generalization of EVD to non-square matrices.

$$
A = U \Sigma V^T
$$

where $U \in \mathbb{R}^{M \times M}$ and $V \in \mathbb{R}^{N \times N}$ are orthogonal matrices and $\Sigma \in \mathbb{R}^{M \times N}$ is a diagonal matrix.

**Covariance Matrix and Principal Components**:

A covariance matrix is a square matrix that captures the pairwise covariances between multiple variables in a dataset. For a data matrix with $n$ features, the covariance matrix is an $n \times n$ matrix where:

- The element at position $(i, j)$ represents the covariance between feature $i$ and feature $j$
- Diagonal elements represent the variance of each feature
- The matrix is symmetric (i.e., $\text{Cov}(X_i, X_j) = \text{Cov}(X_j, X_i)$)

Mathematically, for a data matrix $\mathbf{X}$ with centered columns (mean subtracted), the covariance matrix is:

$$
\text{Cov} = \frac{1}{n-1} \mathbf{X}^T \mathbf{X}
$$

**Connection to PCA**:
- The **eigenvectors** of the covariance matrix point in the directions of maximum variance in the data (these are called **principal components**)
- The **eigenvalues** indicate the magnitude of variance in those directions
- This is the foundation of **PCA**, which uses eigenvalue decomposition of the covariance matrix to find the most important directions in the data

Since covariance matrices are real symmetric matrices, they can be decomposed using eigenvalue decomposition: $\text{Cov} = Q \Lambda Q^T$, where $Q$ contains the eigenvectors and $\Lambda$ contains the eigenvalues.



