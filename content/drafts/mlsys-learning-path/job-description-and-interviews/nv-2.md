# Nvidia 3

Fri, 10 Apr 26

### Interview Overview

- Technical interview with Pasha from Nvidia (3rd interview of the day)
- Focus on ML infrastructure background and debugging experience
- Non-coding interview format with deep-dive technical questions

### Current Role & Experience

- ML Platform Engineer at Omnivore
  - Works with research teams on training and inference
  - Focus on batch inference (not real-time)
  - Primary responsibilities: cluster optimization, job performance, debugging failures
- Experience with 200 H100 nodes (1,600 GPUs total)
- Technical lead role (individual contributor, not management)

### Complex Debugging Examples

- GPU kernel performance optimization
  - Identified NCCL conflicts with matrix multiplication operations
  - Used tracing to find resource conflicts between NCCL streams and tensor cores
  - Solution: Tuned NCCL flags to reduce SM resource usage
  - Required correlating traces across multiple captures to identify patterns
- Data loading optimization
  - Implemented distributed KV cache system using RDMA
  - Addressed slow external storage access for multimodal workloads
  - Used AWS EFA fabric for higher bandwidth node-to-node transfers

### Debugging Methodology

- Collaborative approach with research teams to understand workloads
- Theoretical performance modeling (MFU calculations) vs actual traces
- Infrastructure failure triage using DCGM metrics
- Distributed training timeout debugging
  - Log analysis across workers
  - Isolation and reproduction in smaller environments
  - Network partition testing and MPI validation

### Nvidia Team & Role Discussion

- Nvidia’s Efficiency organization focuses on software stack optimization
- Team handles failure attribution and performance debugging
- Mix of manual debugging for new issues + automation tools for known problems
- Supports internal research (recent NIMO 3 model training) and external customers
- Tools and learnings shared with cloud service providers

### Career Motivation

- Seeking deeper low-level AI infrastructure knowledge
- Current role too research-focused, wants more engineering-oriented environment
- Interest in latest hardware optimization techniques

---

Chat with meeting transcript: https://notes.granola.ai/t/c9b9fd55-959e-4f98-a4a1-e0256d9b5717-00demib2