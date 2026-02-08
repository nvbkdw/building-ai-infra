---
title: "Math for Deep Learning - Part 3"

date: 2025-11-19
tags: ["deep learning","math"]
author: "Ryan H."
description: "This blog post covers the math behind deep learning."
summary: "This blog post covers the math behind deep learning."
cover:
    image: "math-for-deep-learning.png"
    alt: "Math for Deep Learning - Part 3"
    relative: true
---

This blog post covers the math behind deep learning. We will cover the following topics:
- Part 1: Linear Algebra
- Part 2: Calculus
- Part 3: Probability


## Probability

Probability is a measure of the likelihood that an event will occur. It is a number between 0 and 1, where 0 means the event is impossible and 1 means the event is certain.

$$
P(A) = \frac{n(A)}{n(S)}
$$

Set theory, vene diagram, and probability space.

Probability as "measure" or "volume" of a set.

- discrete set: probability mass function: $P(x) = \frac{n(x)}{n(S)}$
- continuous set: probability density function: $P(x) = \frac{1}{n(S)} \int_{x} f(x) dx$

Probability density function is a function that gives the probability of a continuous random variable taking a specific value.

Probability properties:
- **Addition Rule**: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- **Multiplication Rule**: $P(A \cap B) = P(A) P(B|A)$
- **Conditional Probability**: $P(A|B) = \frac{P(A \cap B)}{P(B)}$
- **Independence**: $P(A \cap B) = P(A) P(B)$
- **Total Probability**: $P(A) = \sum_{i=1}^n P(A|B_i) P(B_i)$
- **Bayes' Theorem**: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$

## Bayes' Theorem

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

