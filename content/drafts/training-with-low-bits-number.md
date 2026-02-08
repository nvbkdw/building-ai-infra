---
title: "Training with Low-Bits Number"
date: 2025-11-27
tags: ["quantization", "deep learning"]
author: "Ryan H."
description: "This blog post covers the training with low-bits number in deep learning."
summary: "This blog post covers the training with low-bits number in deep learning."
cover:
    image: "training-with-low-bits-number.png"
    alt: "Training with Low-Bits Number"
    relative: true
---



FP16 training, with scaling factor
why scaling factor? otherwise gradient does not fall into the range of FP16

BF16 training

FP8 training??


FP4 training?
- MXFP4:
    - 32-element block, E8M0
- NVFP4:
    - 16-element block, E4M3





Non-deterministic in LLM inference:
https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/


Defeating the Training-Inference Mismatch via FP16
https://arxiv.org/pdf/2510.26788


SGLang FP8 training as RL: https://lmsys.org/blog/2025-11-25-fp8-rl/