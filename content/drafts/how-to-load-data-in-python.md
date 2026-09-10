---
title: "How to load data into GPU as fast as possible"
date: 2026-08-26
tags: ["python", "asyncio"]
author: "Ryan H."
description: "This blog post covers python asyncio, dataloading, and GPU memory copy."
summary: "This blog post covers python asyncio, dataloading, and GPU memory copy."
---


# The Problem
in ML system, we often need to load data from external storage and feed data to GPU. before any GPU operation, the data should be ready on the host memory, dataloading speed is critical to avoid idle GPU time. 
In this blog we will study a few techniques for loading data as fast a possible.


# Python GIL & `asyncio`
Dataloading involve amount of parallel IO, however Python’s Global Interpreter Lock (GIL) will only allow one thread of Python code running at once. 
This means if you are just parallelizing Python code, you won’t get true parallelism.

Because of python GIL, creating many thread pool in python for I/O is not efficient, it introduce a lot of thread context switch overhead, and gain nothing for parallel I/O.

Asyncio is excellent for many operations waiting for IO. It does not parallelize Python code, instead it use one thread (or event loop) to concurrently zaggle between many different "tasks".

TBD: IO using asyncio

# Release GIL
Run IO outside python GIL and truely running threads in parallel

use Rust lib


TBD: passing result back to python with zero-copy


# Load data into GPU

host to device memory copy
GPU stream
CUDA pinned memory, pinned memory allocation
Async HtoD mem copy














