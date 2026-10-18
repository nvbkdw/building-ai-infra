---
title: "GPU Provisioner"
description: "A mini-SkyPilot in Rust. Provisions a GPU VM on the cheapest cloud offer with capacity across AWS, GCP, and Prime Intellect, fails over on capacity errors without leaking VMs, and can turn the node into a single-node k3s cluster."
summary: "Multi-cloud GPU VM provisioner in Rust with cost-ordered failover"
tags: ["rust", "gpu", "cloud", "kubernetes"]
externalUrl: "https://github.com/DumbAI/gpu-provisioner"
weight: 6
hidemeta: true
---

A mini-SkyPilot in Rust. `gpuprov` provisions a GPU virtual machine on the cheapest cloud offer that has capacity, waits until SSH works, and records the VM locally. Optionally, it turns the VM into a single-node k3s Kubernetes cluster.

```sh
gpuprov provision --name dev --gpus L4:1 --cloud gcp \
  --region us-central1 --region us-east1
gpuprov ssh dev -- nvidia-smi
gpuprov down dev
```

## Key Features

- **One interface, many clouds** — AWS (EC2), GCP (Compute Engine), Prime Intellect, and a local Docker "cloud" for testing, all behind one provider abstraction
- **Cost-ordered failover** — offers are tried cheapest first; a capacity error tears down the partial attempt and moves on, a quota error skips the region, and a VM whose teardown can't be confirmed is marked `LEAKED` instead of forgotten
- **Durable operations** — every operation is journaled in local SQLite, so Ctrl-C cleans up what the run created and a crash resumes from the last recorded state
- **SSH-native access** — a local ed25519 key and a per-VM `~/.ssh/config` entry make plain `ssh`, VS Code Remote, and `rsync` just work
- **Optional Kubernetes** — `--k8s` installs k3s over SSH, sets up the NVIDIA device plugin, validates the cluster, and writes a kubeconfig (direct API endpoint where ports can be opened, SSH tunnel otherwise)
- **No server** — a single static binary, with state in `~/.gpu-provisioner`

## Architecture

Each cloud implements a `CloudProvider` trait of six idempotent primitives. Provisioning runs as state machines that observe the cloud before every action, so a resumed run never creates a second instance.

**Tech stack**: Rust, SQLite, SSH, k3s, NVIDIA device plugin

[View documentation](https://dumbai.github.io/gpu-provisioner/) · [View on GitHub](https://github.com/DumbAI/gpu-provisioner)
