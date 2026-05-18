# Nvidia engineering interview with Sental Kumar Gopal for ML infrastructure role

Wed, 11 Mar 26

### Interview Overview

- Phone screen with Santal Kumar Gopal, Engineering Manager at Nvidia
- Role: Dynamo team - open source software for data center scale LLM inferencing
  - Handles KV caching, block manager, routing, planner deployment
  - Focus on runtime inference engines (VLAN, TensorRT integration)
  - Scaling workers, data transmission, performance bottlenecks, fault tolerance
- Team building phase (Santal joined 1.5 months ago)
  - Supporting more models, system robustness, new features

### Technical Discussion - Infrastructure & Optimization

- Current role: Building ML infrastructure at AI research company
  - Kubernetes + Ray orchestration for full ML cycle
  - Ad hoc experiments, long-term training, batch + real-time inference
- Distributed caching solution built on Curve (Rust-based)
  - Added RDMA support for AWS EFA cards
  - 10-20x speedup for data loading
  - Caches data close to cluster vs. slow S3 TCP network
  - Two use cases: multimodal data preprocessing, KV cache via SGLang’s high cache interface
- Model parallelism approach:
  - Context parallelism for long sequences (millions of tokens)
  - Ring attention with Nvidia Transformer Engine
  - Combined with DDP, FSDP for training
  - Tensor parallelism for inference (smaller models)
- Operator fusion optimizations:
  - Pointwise operations (normalization, residual connections)
  - Rope calculation integration
  - Reduces memory round trips and kernel launches

### LLM Inference Architecture Deep Dive

- Request flow walkthrough:
  1. API tokenization
  2. Router with prefix tree structure for cache matching
  3. Load balancing based on cache hit rate and real-time load
  4. Inference engine (VLLM/SGLang) execution
  5. Page table management in GPU for non-contiguous caching
- Prefill phase:
  - Model weights loaded, layer-by-layer computation
  - KV cache written as layer rows, token columns
  - Cache transformation for CPU offload (token rows, layer columns)
- Decode phase:
  - Fetch existing KV cache to GPU
  - Generate one token at a time, update cache
- Prefix caching optimization:
  - Dedicated worker pools for common system prompts
  - Pre-computed cache chunks for known prefixes
- Heavy Rust codebase with Python bindings for Dynamo

### Team Structure & Collaboration

- Works closely with VLLM/SGLang open source maintainers
  - Pull requests for new model preprocessing/postprocessing
  - Integration between Dynamo and open source engines
- Collaboration areas:
  - MOE model expert parallelism and hybrid 2D/3D parallelism
  - Scheduling improvements in VLLM/SGLang
  - Performance optimization with TensorRT team (internal Nvidia)
  - GPU vs CPU workload distribution decisions

---

Chat with meeting transcript: https://notes.granola.ai/t/12b9eaf9-aa82-419e-afb4-c74cb97b3ce4-00demib2