---
title: "RDMA Dev Notes"
date: 2026-01-03
tags: ["RDMA", "dev"]
author: "Ryan H."
description: "This blog post covers the RDMA dev notes."
summary: "This blog post covers the RDMA dev notes."
cover:
    image: "RDMA-dev-notes.png"
    alt: "RDMA Dev Notes"
    relative: true
---


## Introduction



Get basic inoformation about EFA NIC:

``` bash
$./build/libfabric/bin/fi_info --fabric efa --verbose
---
fi_info:
    caps: [ FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, FI_READ, FI_WRITE, FI_RECV, FI_SEND, FI_REMOTE_READ, FI_REMOTE_WRITE, FI_MULTI_RECV, FI_LOCAL_COMM, FI_REMOTE_COMM, FI_SOURCE, FI_DIRECTED_RECV ]
    mode: [ FI_MSG_PREFIX ]
    addr_format: FI_ADDR_EFA
    src_addrlen: 32
    dest_addrlen: 0
    src_addr: fi_addr_efa://[fe80::816:eff:fe0c:9ba3]:0:0
    dest_addr: (null)
    handle: (nil)
    fi_tx_attr:
        caps: [ FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, FI_READ, FI_WRITE, FI_SEND ]
        mode: [ FI_MSG_PREFIX ]
        op_flags: [  ]
        msg_order: [  ]
        inject_size: 4096
        size: 4096
        iov_limit: 4
        rma_iov_limit: 1
        tclass: 0x0
    fi_rx_attr:
        caps: [ FI_MSG, FI_RMA, FI_TAGGED, FI_ATOMIC, FI_RECV, FI_REMOTE_READ, FI_REMOTE_WRITE, FI_MULTI_RECV, FI_SOURCE, FI_DIRECTED_RECV ]
        mode: [ FI_MSG_PREFIX ]
        op_flags: [  ]
        msg_order: [  ]
        size: 8192
        iov_limit: 4
    fi_ep_attr:
        type: FI_EP_RDM
        protocol: FI_PROTO_EFA
        protocol_version: 4
        max_msg_size: 18446744073709551615
        msg_prefix_size: 176
        max_order_raw_size: 0
        max_order_war_size: 0
        max_order_waw_size: 0
        mem_tag_format: 0xaaaaaaaaaaaaaaaa
        tx_ctx_cnt: 1
        rx_ctx_cnt: 1
        auth_key_size: 0
    fi_domain_attr:
        domain: 0x0
        name: rdmap70s0-rdm
        threading: FI_THREAD_SAFE
        progress: FI_PROGRESS_AUTO
        resource_mgmt: FI_RM_ENABLED
        av_type: FI_AV_TABLE
        mr_mode: [ FI_MR_LOCAL, FI_MR_VIRT_ADDR, FI_MR_ALLOCATED, FI_MR_PROV_KEY, FI_MR_HMEM ]
        mr_key_size: 4
        cq_data_size: 4
        cq_cnt: 512
        ep_cnt: 256
        tx_ctx_cnt: 256
        rx_ctx_cnt: 256
        max_ep_tx_ctx: 1
        max_ep_rx_ctx: 1
        max_ep_stx_ctx: 0
        max_ep_srx_ctx: 0
        cntr_cnt: 0
        mr_iov_limit: 1
        caps: [ FI_LOCAL_COMM, FI_REMOTE_COMM ]
        mode: [  ]
        auth_key_size: 0
        max_err_data: 0
        mr_cnt: 262144
        tclass: 0x0
    fi_fabric_attr:
        name: efa
        prov_name: efa
        prov_version: 200.0
        api_version: 2.0
    nic:
        fi_device_attr:
            name: rdmap70s0
            device_id: 0xefa1
            device_version: 6
            vendor_id: 0x1d0f
            driver: efa
            firmware: 0.0.0.0
        fi_bus_attr:
            bus_type: FI_BUS_PCI
            fi_pci_attr:
                domain_id: 0
                bus_id: 70
                device_id: 0
                function_id: 0
        fi_link_attr:
            address: EFA-fe80::816:eff:fe0c:9ba3
            mtu: 8760
            speed: 25000000000
            state: FI_LINK_UP
            network_type: Ethernet
---
fi_info:
    caps: [ FI_MSG, FI_RECV, FI_SEND, FI_LOCAL_COMM, FI_REMOTE_COMM, FI_SOURCE ]
    mode: [ FI_MSG_PREFIX ]
    addr_format: FI_ADDR_EFA
    src_addrlen: 32
    dest_addrlen: 0
    src_addr: fi_addr_efa://[fe80::816:eff:fe0c:9ba3]:0:0
    dest_addr: (null)
    handle: (nil)
    fi_tx_attr:
        caps: [ FI_MSG, FI_SEND ]
        mode: [ FI_MSG_PREFIX ]
        op_flags: [  ]
        msg_order: [  ]
        inject_size: 0
        size: 4096
        iov_limit: 2
        rma_iov_limit: 0
        tclass: 0x0
    fi_rx_attr:
        caps: [ FI_MSG, FI_RECV, FI_SOURCE ]
        mode: [ FI_MSG_PREFIX ]
        op_flags: [  ]
        msg_order: [  ]
        size: 8192
        iov_limit: 3
    fi_ep_attr:
        type: FI_EP_DGRAM
        protocol: FI_PROTO_EFA
        protocol_version: 1
        max_msg_size: 8928
        msg_prefix_size: 40
        max_order_raw_size: 0
        max_order_war_size: 0
        max_order_waw_size: 0
        mem_tag_format: 0x0000000000000000
        tx_ctx_cnt: 1
        rx_ctx_cnt: 1
        auth_key_size: 0
    fi_domain_attr:
        domain: 0x0
        name: rdmap70s0-dgrm
        threading: FI_THREAD_DOMAIN
        progress: FI_PROGRESS_AUTO
        resource_mgmt: FI_RM_DISABLED
        av_type: FI_AV_UNSPEC
        mr_mode: [ FI_MR_LOCAL, FI_MR_VIRT_ADDR, FI_MR_ALLOCATED, FI_MR_PROV_KEY ]
        mr_key_size: 4
        cq_data_size: 4
        cq_cnt: 512
        ep_cnt: 256
        tx_ctx_cnt: 256
        rx_ctx_cnt: 256
        max_ep_tx_ctx: 1
        max_ep_rx_ctx: 1
        max_ep_stx_ctx: 0
        max_ep_srx_ctx: 0
        cntr_cnt: 0
        mr_iov_limit: 1
        caps: [ FI_LOCAL_COMM, FI_REMOTE_COMM ]
        mode: [  ]
        auth_key_size: 0
        max_err_data: 0
        mr_cnt: 262144
        tclass: 0x0
    fi_fabric_attr:
        name: efa
        prov_name: efa
        prov_version: 200.0
        api_version: 2.0
    nic:
        fi_device_attr:
            name: rdmap70s0
            device_id: 0xefa1
            device_version: 6
            vendor_id: 0x1d0f
            driver: efa
            firmware: 0.0.0.0
        fi_bus_attr:
            bus_type: FI_BUS_PCI
            fi_pci_attr:
                domain_id: 0
                bus_id: 70
                device_id: 0
                function_id: 0
        fi_link_attr:
            address: EFA-fe80::816:eff:fe0c:9ba3
            mtu: 8760
            speed: 25000000000
            state: FI_LINK_UP
            network_type: Ethernet
```


In RDMA, unlike traditional sockets, the receiver must have a RECV operation posted before the sender sends data. If no RECV is waiting when data arrives, the data is lost. This is critical for bidirectional communication.

```
Client                              Server
  |                                   |
  |  ----[CONNECT + address]---->     |  (RECV 1 waiting)
  |                                   |  (RECV 2 waiting)
  |  (RECV waiting for response)      |
  |  ----[DATA: "Hello"]-------->     |  (reverses, sends back)
  |                                   |
  |  <---[DATA: "olleH"]---------     |
  |                                   |
```




Check NIC link speed

``` bash
ethtool eth0 | grep Speed

```







## References
Harnessing 3200 Gbps Network, Lequn Chen, 2024: https://le.qun.ch/en/blog/2024/12/25/libfabric-efa-0-intro/