# ML infrastructure interview with Ajay

Wed, 18 Mar 26

### Interview Overview

- Technical interview with Ajay from AI efficiency team at NVIDIA
- Interview format: intro → background → technical questions → Q&A
- Ajay works on optimizing workloads from infrastructure perspective

### Ryan’s Background & Experience

- Backend engineer specializing in distributed systems
- Previous roles at AWS and Uber building internal ML infrastructure platforms
  - Created unified APIs for ML workflows (training, evaluation, data pipelines)
  - Built high-level orchestration abstracting job schedulers (Kubernetes, Slurm)
  - Enabled product teams to define end-to-end workflows without infrastructure management
- Current role at AI research startup
  - Built entire ML infrastructure solo on Kubernetes
  - Orchestrates multiple GPU clusters using Ray framework
  - Integrated PyTorch with CLI tools for simplified job submission
  - Manages 3-layer storage: blob storage (data lake), persistent volumes, distributed cache

### Technical Deep Dives

- **Data Loading Optimization Project**
  - Challenge: Accelerating multimodal medical data loading (very large images)
  - Solutions implemented:
    1. Deployed distributed KV cache across GPU cluster with RDMA networking
    2. Used local H100 SSD for cache overflow when memory insufficient
    3. Rewrote PyTorch data loader in Rust to eliminate extra data copies
    4. Avoided FUSE protocol overhead with in-application data loader
  - Performance: Chose RDMA over Ethernet due to bandwidth limitations
  - Trade-off: Shared RDMA bandwidth with existing jobs, but data loading was primary bottleneck
- **Multi-cloud Integration & Topology Optimization**
  - Built global queuing system using Skypilot for cross-cloud job dispatch
  - Added custom rack-aware scheduling for H100 and GB200 clusters
    - GB200: 18 nodes per rack requiring topology awareness
    - H100: Availability zone placement on AWS
  - Performance gain: 3x faster training when jobs packed within same rack (NVLink vs cross-rack communication)
  - Cluster scale: 50-150 nodes (200-400 GPUs total)
  - Jobs limited to single cloud vendor (no cross-cloud job splitting)
- **Debugging Approach & Common Issues**
  - Performance debugging workflow:
    1. Check GPU utilization across all nodes
    2. Analyze CPU, memory, network throughput metrics
    3. Identify anomalies (temperature, power, single slow GPU dragging fleet)
  - Common failure modes:
    - CUDA library version conflicts
    - GPU thermal/power failures
    - Network congestion causing slowness
  - Historical example: Fixed memory leak in Weaviate vector database
    - Used Go profiler during load testing
    - Found tenant mode bug causing memory accumulation
    - Optimized REST API serialization overhead by switching to gRPC

### Next Steps

- Ryan asked about DGX Cloud product details
- Ajay clarified AI efficiency team works at infrastructure level to improve workload performance/throughput

---

Chat with meeting transcript: https://notes.granola.ai/t/5ebba30c-8be6-49eb-87b7-7477b326b267-00demib2