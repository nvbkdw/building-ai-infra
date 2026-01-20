---
title: "AWS Hyperpod Development Notes"
date: 2026-01-14
tags: ["aws", "hyperpod"]
author: "Ryan H."
description: "This blog post covers the development notes for AWS Hyperpod."
summary: "This blog post covers the development notes for AWS Hyperpod."
cover:
    image: "aws-hyperpod-dev-notes.png"
    alt: "AWS Hyperpod Development Notes"
    relative: true
---

# AWS Hyperpod Development Notes


EKS Cluster and control plane:

* Hyperpod VPC, subnets, security groups, etc.

Hyperpod be default installs EFA device plugin to register EFA resource (`vpc.amazonaws.com/efa`) to k8s. If the plugin is not installed, you can install it manually by installing the EFA helm chart:
``` bash
name: aws-efa-k8s-device-plugin
repo: https://aws.github.io/eks-charts
```
Also, make sure the request `vpc.amazonaws.com/efa` in pod spec, so that your container runtime can attach EFA NIC to the container:
``` yaml
resources:
  requests:
    vpc.amazonaws.com/efa: "1"
```

If everything is configured correctly, you should be able to see the EFA NIC in the container using `fi_info`:

```
# which fi_info
/opt/amazon/efa/bin/fi_info
# fi_info -p efa
provider: efa
    fabric: efa-direct
    domain: rdmap49s0-rdm
    version: 201.0
    type: FI_EP_RDM
    protocol: FI_PROTO_EFA
provider: efa
    fabric: efa
    domain: rdmap49s0-rdm
    version: 201.0
    type: FI_EP_RDM
    protocol: FI_PROTO_EFA
provider: efa
    fabric: efa
    domain: rdmap49s0-dgrm
    version: 201.0
    type: FI_EP_DGRAM
    protocol: FI_PROTO_EFA
```



Hyperpod Node Group:
* On-demand, Training Plan, etc.
* Node group AZ override


EFA network:
* Intra-cluster EFA networking
* FSx with EFA, expensive
* EFA network benchmarking, troubleshooting



