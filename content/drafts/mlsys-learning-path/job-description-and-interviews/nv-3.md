# Nvidia AI infrastructure interview with Jason

Fri, 10 Apr 26

### Interview Overview

- Two-part technical interview for NVIDIA AI infrastructure team
- Focus on failure attribution and analysis for AI training workloads
- Team works on understanding why AI training jobs fail across NVIDIA clusters

### Current Role at Omniva

- AI startup infrastructure work with ML research team
- Built Kubernetes GPU cluster running PyTorch workloads
- Responsibilities include:
  - Infrastructure setup and management
  - Training workload optimization
  - Failure debugging and cluster monitoring
  - Profiling tools development for offline analysis
  - CPU-side optimization (memory, data, networking)
  - Model parallelism implementation

### Failure Detection and Debugging Approach

- Current monitoring uses DCGM metrics
  - GPU temperature correlation with error metrics
  - Utilization vs error rate analysis
- Automation criteria for failure patterns:
  - Repeated occurrences warrant automation
  - Known failure modes (network, CPU, GPU components)
  - New software components require ad-hoc debugging initially
- Hierarchical debugging methodology:
  1. High-level metrics comparison (current vs baseline)
  2. Identify straggler nodes or components
  3. Layer-by-layer analysis (application → kernel → hardware)
  4. Use multiple tracing tools when data conflicts

### Technical Debugging Scenarios

- Noisy/contradictory data handling:
  - Linux memory maps as source of truth
  - Multiple profiling approaches when tools conflict
  - Deeper layer analysis (Python → C++ → kernel)
- Root cause vs contributing factors:
  - Reproduction testing to validate fixes
  - Cascaded failure analysis in distributed systems
  - Hypothesis-driven approach with scientific method
- Ray distributed inference debugging:
  - Start with high-level metrics overview
  - Compare against known-good baselines
  - Isolate straggler nodes through utilization analysis
  - Use tracing for detailed bottleneck identification

### Performance Optimization Experience

- GPU utilization maximization as primary goal
- MFU (Model FLOPs Utilization) benchmarking against published results
- Custom kernel work:
  - Primarily integration of off-shelf kernels (Transformer Engine, FlashAttention)
  - Simple fusion for non-standard attention mechanisms
  - Roofline analysis for compute vs memory bound identification
- Data loading pipeline optimization:
  - Caching systems for data loading acceleration
  - CPU-GPU copy optimization
  - Disk bandwidth and I/O bottleneck analysis

### Platform and Infrastructure Preferences

- Kubernetes orchestration with custom scheduling
- Requirements for AI training platforms:
  - GPU topology awareness
  - Network topology consideration for node allocation
  - Resource isolation between workloads
  - Dynamic environment variable setup for distributed coordination
- Observability priorities:
  1. Hardware-level metrics (GPU, CPU, network utilization)
  2. Error rates and hardware health signals
  3. Application-level metrics (token rate, MFU)
  4. Workload-specific failure mode tracking

---

Chat with meeting transcript: https://notes.granola.ai/t/7a9708ee-dd70-4fb3-823a-f86c7b9ce02a-00demib2