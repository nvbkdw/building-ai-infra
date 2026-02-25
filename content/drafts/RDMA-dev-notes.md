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



To obtain the system topology on Linux, you can use the hwloc library, which includes a command-line tool lstopo that can generate a system topology diagram. 

``` bash
sudo apt update
sudo apt install hwloc
lstopo
```
Another way to obtain system topology is to use the lspci -tv command. We can see the addresses of each PCIe switch and the devices mounted under each PCIe switch. 

``` bash
lspci -tv
```



Calculating PCIe bandwidth
```
$sudo lspci -vv
...
01:00.0 PCI bridge: Amazon.com, Inc. Device 0200 (prog-if 00 [Normal decode])
        Physical Slot: 32
        Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0
        NUMA node: 0
        Region 0: Memory at 84000000 (32-bit, non-prefetchable) [size=1K]
        Bus: primary=01, secondary=02, subordinate=43, sec-latency=0
        I/O behind bridge: 00000000-00000fff [size=4K]
        Memory behind bridge: 84100000-882fffff [size=66M]
        Prefetchable memory behind bridge: 0000030090000000-0000030093ffffff [size=64M]
        Secondary status: 66MHz- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- <SERR- <PERR-
        BridgeCtl: Parity- SERR+ NoISA- VGA- VGA16- MAbort- >Reset- FastB2B-
                PriDiscTmr- SecDiscTmr- DiscTmrStat- DiscTmrSERREn-
        Capabilities: [40] Subsystem: Amazon.com, Inc. Device 0200
        Capabilities: [48] MSI-X: Enable+ Count=1 Masked-
                Vector table: BAR=0 offset=00000000
                PBA: BAR=0 offset=00000200
        Capabilities: [54] Express (v2) Root Port (Slot+), MSI 00
                DevCap: MaxPayload 128 bytes, PhantFunc 0
                        ExtTag- RBE+
                DevCtl: CorrErr- NonFatalErr- FatalErr- UnsupReq-
                        RlxdOrd- ExtTag- PhantFunc- AuxPwr- NoSnoop-
                        MaxPayload 128 bytes, MaxReadReq 128 bytes
                DevSta: CorrErr- NonFatalErr- FatalErr- UnsupReq- AuxPwr- TransPend-
                LnkCap: Port #0, Speed 16GT/s, Width x16, ASPM not supported
                        ClockPM- Surprise- LLActRep+ BwNot- ASPMOptComp-
                LnkCtl: ASPM Disabled; RCB 64 bytes, Disabled- CommClk-
                        ExtSynch- ClockPM- AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 16GT/s (ok), Width x16 (ok)
                        TrErr- Train- SlotClk- DLActive+ BWMgmt- ABWMgmt-
                SltCap: AttnBtn+ PwrCtrl+ MRL- AttnInd+ PwrInd+ HotPlug+ Surprise-
                        Slot #0, PowerLimit 0.000W; Interlock- NoCompl+
                SltCtl: Enable: AttnBtn+ PwrFlt- MRL- PresDet- CmdCplt- HPIrq+ LinkChg+
                        Control: AttnInd Off, PwrInd On, Power- Interlock-
                SltSta: Status: AttnBtn- PowerFlt- MRL- CmdCplt- PresDet+ Interlock-
                        Changed: MRL- PresDet- LinkState-
                RootCap: CRSVisible-
                RootCtl: ErrCorrectable- ErrNon-Fatal- ErrFatal- PMEIntEna+ CRSVisible-
                RootSta: PME ReqID 0000, PMEStatus- PMEPending-
                DevCap2: Completion Timeout: Not Supported, TimeoutDis- NROPrPrP- LTR-
                         10BitTagComp- 10BitTagReq- OBFF Not Supported, ExtFmt- EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS- LN System CLS Not Supported, TPHComp- ExtTPHComp- ARIFwd-
                         AtomicOpsCap: Routing- 32bit- 64bit- 128bitCAS-
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis- LTR- OBFF Disabled, ARIFwd-
                         AtomicOpsCtl: ReqEn- EgressBlck-
                LnkCap2: Supported Link Speeds: 2.5-16GT/s, Crosslink- Retimer- 2Retimers- DRS-
                LnkCtl2: Target Link Speed: 16GT/s, EnterCompliance- SpeedDis-
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance De-emphasis: -6dB
                LnkSta2: Current De-emphasis Level: -6dB, EqualizationComplete- EqualizationPhase1-
                         EqualizationPhase2- EqualizationPhase3- LinkEqualizationRequest-
                         Retimer- 2Retimers- CrosslinkRes: unsupported
        Kernel driver in use: pcieport
...
```

This number comes from a standard calculation based on the PCIe Generation 4.0 specification shown in your lspci output (Speed 16GT/s).Here is the exact math of how 16 GT/s becomes 31.5 GB/s.

1. The Raw Speed (GT/s)Your output shows Speed 16GT/s.GT/s stands for GigaTransfers per second.This refers to the raw number of electrical state changes happening on the wire every second.$16 \text{ GT/s} = 16,000,000,000 \text{ transfers/second per lane}$.

2. The Encoding Overhead (The "Tax")PCIe is a serial protocol. It doesn't send 100% user data; it has to mix in some "overhead" bits to keep the signal synchronized (clock recovery).PCIe 4.0 uses "128b/130b" encoding.This means for every 130 bits sent over the wire, 128 bits are actual data.Efficiency: $128 / 130 \approx 0.985$ (or 98.5% efficient).

3. The CalculationTo get the bandwidth in Gigabytes per second (GB/s): Convert Transfers to Bits:$$16 \text{ GT/s} \times (128/130 \text{ encoding}) = 15.754 \text{ Gigabits/s (per lane)}$$Convert Bits to Bytes:$$15.754 \text{ Gbps} / 8 \text{ bits per Byte} = 1.969 \text{ GB/s (per lane)}

$$Multiply by Lane Width:Your output shows Width x16.$$1.969 \text{ GB/s} \times 16 \text{ lanes} = \mathbf{31.508 \text{ GB/s}}

$$Important Context: Unidirectional vs. Bidirectional 31.5 GB/s is Unidirectional: This is the speed at which you can send data to the device OR receive data from the device.63.0 GB/s is Bidirectional: Since PCIe is full-duplex, you can technically send 31.5 GB/s and receive 31.5 GB/s simultaneously.However, when benchmarking storage or GPU loading, we usually care about the unidirectional speed (e.g., "How fast can I load this texture into VRAM?"), which is why 31.5 GB/s is the "speed limit" number you should keep in mind.




# Disk volume mapping

list all block devices:

``` bash
lsblk
```

list all logical filesystems mount:

``` bash
mount
df -h
```






What is LVM (Logical Volume Manager)?
LVM is a device mapper framework that provides logical volume management for Linux. It adds a layer of abstraction between your physical storage devices and your filesystems.
The hierarchy is:

```
Physical Disks (nvme0n1, nvme1n1, nvme2n1)
    ↓
Physical Volumes (PV) - raw disks initialized for LVM
    ↓
Volume Groups (VG) - pools of storage from multiple PVs
    ↓
Logical Volumes (LV) - virtual partitions you can mount
    ↓
Filesystems (ext4, xfs, etc.) mounted to directories
```

``` bash
sudo apt-get update && sudo apt-get install lvm2
```


## References
Harnessing 3200 Gbps Network, Lequn Chen, 2024: https://le.qun.ch/en/blog/2024/12/25/libfabric-efa-0-intro/