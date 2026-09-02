---
title: "Deepseek V4 Deep Dive"
date: 2026-08-30
tags: ["Deepseek", "LLM", "Inference"]
author: "Ryan H."
description: "This blog post covers Deepseek V4 model architecture and inference"
summary: "This blog post covers Deepseek V4 model architecture and inference."
---


# DeepSeek-V4
[DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/abs/2606.19348) containes efficient model architeture (CSA, HCA, mHC), MoE training, Muon optimizer, post training, inference, evaluation, etc. It does amazing job to make model more efficient, even though longer context length. DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2. 



![DeepSeek-V4 models](/static/deepseek-v4.webp)

DeepSeek-V4 published two models: Flash (284B-13BA) and Pro (1.6T-49BA)

In the rest of post, I will follow symbol to describe model architecture:
| Symbol | Meaning | DeepSeek-V4-Flash | DeepSeek-V4-Pro |
|---|---|---|---|
| $B$ | \(B\)batch size | runtime | runtime |
| $S$ | tokens in this call | prompt length, or 1 in decode | prompt length, or 1 in decode |
| $P$ | tensor-parallel world size | runtime | runtime |
| $V$ | vocabulary size | 129,280 | 129,280 |
| $D$ | hidden dimension | 4,096 | 7,168 |
| $H_c$ | hyper-connection multiplicity | 4 | 4 |
| $L$ | Transformer blocks | 43 | 61 |
| $H_q$ | attention query heads | 64 | 128 |
| $D_h$ | attention head dimension | 512 | 512 |
| $D_r$ | RoPE dimensions per head | 64 | 64 |
| $R_q$ | query low-rank dimension | 1,024 | 1,536 |
| $G$ | output projection groups | 8 | 16 |
| $R_o$ | output rank per group | 1,024 | 1,024 |
| $W$ | sliding-window size | 128 | 128 |
| $E$ | routed experts per layer | 256 | 384 |
| $K_e$ | routed experts activated/token | 6 | 6 |
| $D_{\text{ff}}$ | expert intermediate size | 2,048 | 3,072 |

Layer attention layout is:
Layers 0–1: HCA, compression ratio 128.
Layers 2–60: CSA and HCA alternate.
Even layers 2, 4, …, 60: CSA, ratio 4.
Odd layers 3, 5, …, 59: HCA, ratio 128.
Separate MTP layer 61: pure sliding-window attention.

DeepSeek also open source [reference inference code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py). This post use this reference and tech report, and dive into details into serving DeepSeek-V4.

# mHC connection

mHC is introduced in [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/html/2512.24880v2)

[Hyper-Connections](https://arxiv.org/html/2409.19606v3) expands the width of the residual stream and enhancing connection complexity, HC has show performance advantage by significantly without increase too much FLOPs overhead. 

In the HC formulation, the input to the $l$-th layer, $x_l \in \mathbb{R}^{1 \times C}$, is expanded by a factor of $n$ into a hidden matrix $\mathbf{x}_l = (x_{l,0}^{\top}, \ldots, x_{l,n-1}^{\top})^{\top} \in \mathbb{R}^{n \times C}$ — an $n$-stream residual. Each layer learns three mappings over these streams: $\mathcal{H}_l^{\mathrm{pre}}$ gathers the $n$ streams into a single layer input, $\mathcal{H}_l^{\mathrm{post}}$ scatters the layer output back across the streams, and $\mathcal{H}_l^{\mathrm{res}}$ mixes the streams among themselves. In effect, the identity shortcut of a vanilla residual is replaced by a learned $n \times n$ connection.

That substitution is exactly where the instability comes from. The identity shortcut is what keeps deep networks trainable: signal and gradient pass through every layer with their magnitude untouched. An unconstrained $\mathcal{H}_l^{\mathrm{res}}$ re-scales the residual at every layer, and compounded over $L = 61$ layers this surfaces as exploding or vanishing gradients. mHC's fix is geometric: project $\mathcal{H}_l^{\mathrm{res}}$ onto the manifold of doubly stochastic matrices (rows and columns sum to 1), so each mixing step is a convex combination of streams — free to route information between streams, but unable to amplify or attenuate the total signal. The gather/scatter mappings are likewise bounded by Sigmoid gates. The next subsection walks through how these mappings are computed and constrained.

## Parameterization and manifold projection

Given the input hidden matrix $\mathbf{x}_l \in \mathbb{R}^{n \times C}$ at the $l$-th layer, mHC first flattens it into a vector $\vec{\mathbf{x}}_l = \mathrm{vec}(\mathbf{x}_l) \in \mathbb{R}^{1 \times nC}$ to preserve full context information. Then, following the original HC formulation, the dynamic and static mappings are:

$$
\begin{cases}
\vec{\mathbf{x}}_l^{\prime} = \mathrm{RMSNorm}(\vec{\mathbf{x}}_l) \\
\tilde{\mathcal{H}}_l^{\mathrm{pre}} = \alpha_l^{\mathrm{pre}} \cdot \left(\vec{\mathbf{x}}_l^{\prime} \varphi_l^{\mathrm{pre}}\right) + \mathbf{b}_l^{\mathrm{pre}} \\
\tilde{\mathcal{H}}_l^{\mathrm{post}} = \alpha_l^{\mathrm{post}} \cdot \left(\vec{\mathbf{x}}_l^{\prime} \varphi_l^{\mathrm{post}}\right) + \mathbf{b}_l^{\mathrm{post}} \\
\tilde{\mathcal{H}}_l^{\mathrm{res}} = \alpha_l^{\mathrm{res}} \cdot \mathrm{mat}\left(\vec{\mathbf{x}}_l^{\prime} \varphi_l^{\mathrm{res}}\right) + \mathbf{b}_l^{\mathrm{res}},
\end{cases}
$$

where $\varphi_l^{\mathrm{pre}}, \varphi_l^{\mathrm{post}} \in \mathbb{R}^{nC \times n}$ and $\varphi_l^{\mathrm{res}} \in \mathbb{R}^{nC \times n^2}$ are linear projections for the dynamic mappings, and $\mathrm{mat}(\cdot)$ is a reshape function from $\mathbb{R}^{1 \times n^2}$ to $\mathbb{R}^{n \times n}$. 

As a result, $\tilde{\mathcal{H}}_l^{\mathrm{pre}} \in  \mathbb{R}^{1 \times n}$, $\tilde{\mathcal{H}}_l^{\mathrm{post}} \in  \mathbb{R}^{1 \times n}$, and $\tilde{\mathcal{H}}_l^{\mathrm{res}} \in  \mathbb{R}^{n \times n}$.

The final constrained mappings are obtained via:

$$
\begin{cases}
\mathcal{H}_l^{\mathrm{pre}} = \sigma\left(\tilde{\mathcal{H}}_l^{\mathrm{pre}}\right) \\
\mathcal{H}_l^{\mathrm{post}} = 2\,\sigma\left(\tilde{\mathcal{H}}_l^{\mathrm{post}}\right) \\
\mathcal{H}_l^{\mathrm{res}} = \text{Sinkhorn-Knopp}\left(\tilde{\mathcal{H}}_l^{\mathrm{res}}\right),
\end{cases}
$$

where $\sigma(\cdot)$ denotes the Sigmoid function. The $\text{Sinkhorn-Knopp}(\cdot)$ operator first makes all elements positive via an exponent operator, then alternately rescales rows and columns to sum to 1. Starting from the positive matrix $\mathbf{M}^{(0)} = \exp\left(\tilde{\mathcal{H}}_l^{\mathrm{res}}\right)$, the normalization iteration proceeds as:

$$
\mathbf{M}^{(t)} = \mathcal{T}_r\left(\mathcal{T}_c\left(\mathbf{M}^{(t-1)}\right)\right),
$$

where $\mathcal{T}_r$ and $\mathcal{T}_c$ denote row and column normalization. This converges to a doubly stochastic matrix $\mathcal{H}_l^{\mathrm{res}} = \mathbf{M}^{(t_{\max})}$ as $t_{\max} \to \infty$; DeepSeek-V4 uses $t_{\max} = 20$ (`hc_sinkhorn_iters: 20` in the model config).

The Sigmoid gates keep $\mathcal{H}_l^{\mathrm{pre}}$ and $\mathcal{H}_l^{\mathrm{post}}$ bounded, and the doubly stochastic $\mathcal{H}_l^{\mathrm{res}}$ makes stream mixing a convex combination — **signal magnitude is preserved across layers, which is exactly the manifold constraint that fixes HC's training instability**.

The full computation graph of one mHC residual connection:

To be concise, we removed subtitle $l$, i.e. $\mathbf{x}_l$ display as $\mathbf{x}$ in the diagram. Rounded nodes are tensors (with shapes), rectangles are operations annotated as `input shapes → output shape`:

```mermaid
flowchart TD
    XL(["x : (n × C)<br/>n-stream residual"])

    subgraph MAPS["mapping computation"]
        FLAT["vec + RMSNorm<br/>(n × C) → (1 × nC)"]
        XP(["x′ : (1 × nC)"])
        PPRE["x′ · φ_pre : (1 × nC) · (nC × n) → (1 × n)<br/>α · ( ) + b, then σ"]
        PPOST["x′ · φ_post : (1 × nC) · (nC × n) → (1 × n)<br/>α · ( ) + b, then 2σ"]
        PRES["x′ · φ_res : (1 × nC) · (nC × n²) → (1 × n²)<br/>mat → (n × n), α · ( ) + b<br/>then Sinkhorn-Knopp × 20"]
        HPRE(["H_pre : (1 × n)"])
        HPOST(["H_post : (1 × n)"])
        HRES(["H_res : (n × n)<br/>doubly stochastic"])
    end

    subgraph APPLY["residual application"]
        GATHER["gather : H_pre · x <br/>(1 × n) · (n × C) → (1 × C)"]
        FW["F( · , W) : attention / MoE<br/>(1 × C) → (1 × C)"]
        SCATTER["scatter : H_postᵀ · F<br/>(n × 1) · (1 × C) → (n × C)"]
        MIX["stream mixing : H_res · x<br/>(n × n) · (n × C) → (n × C)"]
        SUM(("+"))
    end

    XNEXT(["x_out : (n × C)<br/>next-layer residual"])

    XL --> FLAT --> XP
    XP --> PPRE --> HPRE
    XP --> PPOST --> HPOST
    XP --> PRES --> HRES
    XL --> GATHER
    HPRE --> GATHER
    GATHER --> FW --> SCATTER
    HPOST --> SCATTER
    XL --> MIX
    HRES --> MIX
    SCATTER --> SUM
    MIX --> SUM
    SUM --> XNEXT
```

Ablation study shows residual stream mixing $\mathcal{H}_l^{\mathrm{res}}$ yields the most significant performance gain. This finding underscores the critical importance of effective information exchange within the residual stream.

## mHC Connection Inference

mHC connection is applied in each layer, around both attention and MLP.
Code snippet from [DeepSeek V4 reference code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py):

```python
class Block(nn.Module):
    """Transformer block with Hyper-Connections (HC) mixing.
    Instead of a simple residual, HC maintains `hc_mult` copies of the hidden state.
    hc_pre: reduces hc copies -> 1 via learned weighted sum (pre-weights from Sinkhorn).
    hc_post: expands 1 -> hc copies via learned post-weights + combination matrix."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        # hc_mult is the mHC stream expansion factor n
        # There are three learnt stream mapping parameter: 
        # - H_pre, of size (n), 
        # - H_post, of size (n)
        # - H_res, of size (n^2)
        # thus, total size is (2+n)*n
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b,s,hc,d]
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        
        # RMSNorm and mHC mapping parameter
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        
        # mHC stream projection
        # use one kernel to fuse three projections
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x

        # mHC stream projection
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)

        # attention
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)

        # residual stream mixing
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


```

```python
@tilelang.jit(pass_configs=pass_configs)
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    threads = 64

    @T.prim_func
    def hc_split_sinkhorn_kernel_(
        mixes: T.Tensor[(n, mix_hc), FP32],
        hc_scale: T.Tensor[(3,), FP32],
        hc_base: T.Tensor[(mix_hc,), FP32],
        pre: T.Tensor[(n, hc), FP32],
        post: T.Tensor[(n, hc), FP32],
        comb: T.Tensor[(n, hc, hc), FP32],
    ):
        with T.Kernel(n, threads=threads) as i:
            mixes_shared = T.alloc_shared(mix_hc, FP32)
            comb_frag = T.alloc_fragment((hc, hc), FP32)
            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = mixes_shared[j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2]

            row_sum = T.alloc_fragment(hc, FP32)
            col_sum = T.alloc_fragment(hc, FP32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(hc, FP32)
            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
            T.reduce_sum(comb_frag, row_sum, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(comb_frag, col_sum, dim=0)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb[i, :, :])

    return hc_split_sinkhorn_kernel_
```



Per token and per layer, counting activation elements only (projection weights $\varphi$ and the internal I/O of the layer function $F$ are excluded; weight reads are amortized across the tokens of a batch). FLOPs use the $2mnk$ GEMM convention:

| Method | Operation | Read (elements) | Write (elements) | FLOPs |
|---|---|---|---|---|
| Residual connection | Residual merge | $2C$ | $C$ | $C$ |
| | **Total** | $2C$ | $C$ | $C$ |
| HC / mHC | Calculate $\tilde{\mathcal{H}}^{\mathrm{pre}}, \tilde{\mathcal{H}}^{\mathrm{post}}, \tilde{\mathcal{H}}^{\mathrm{res}}$ | $nC$ | $n^2 + 2n$ | $2n^3C + 4n^2C$ |
| | Apply $\mathcal{H}^{\mathrm{pre}}$ (gather) | $nC + n$ | $C$ | $2nC$ |
| | Apply $\mathcal{H}^{\mathrm{post}}$ (scatter) | $C + n$ | $nC$ | $2nC$ |
| | Apply $\mathcal{H}^{\mathrm{res}}$ (mix) | $nC + n^2$ | $nC$ | $2n^2C$ |
| | Residual merge | $2nC$ | $nC$ | $nC$ |
| | **Total** | $(5n+1)C + n^2 + 2n$ | $(3n+1)C + n^2 + 2n$ | $2n^3C + 6n^2C + 5nC$ |

Where the FLOPs come from:

- **Calculate mappings**: $\vec{\mathbf{x}}^{\prime}\varphi^{\mathrm{pre}}$ and $\vec{\mathbf{x}}^{\prime}\varphi^{\mathrm{post}}$ are $(1 \times nC)(nC \times n)$ GEMVs, $2n^2C$ FLOPs each; $\vec{\mathbf{x}}^{\prime}\varphi^{\mathrm{res}}$ is $(1 \times nC)(nC \times n^2)$, $2n^3C$ FLOPs
- **Gather**: $(1 \times n)(n \times C)$, $2nC$ FLOPs; **scatter**: $(n \times 1)(1 \times C)$, $2nC$ FLOPs; **mix**: $(n \times n)(n \times C)$, $2n^2C$ FLOPs
- **Residual merge**: $nC$ additions
- Omitted small terms: RMSNorm $O(nC)$, the $\alpha, \mathbf{b}$ affine and Sigmoid gates $O(n^2)$, and Sinkhorn-Knopp $O(t_{\max} n^2)$ — all stay in registers in the fused kernel above

Plugging in DeepSeek-V4-Pro numbers ($n = 4$, $C = 7168$): the FLOPs overhead is $2n^3C + 6n^2C + 5nC \approx 1.75$ MFLOPs per token per layer — roughly $0.1\%$ of the $\approx 1.6$ GFLOPs the layer itself costs ($\approx 49\text{B}/61$ activated parameters). The memory side is a different story: reads grow from $2C$ to $(5n+1)C + n^2 + 2n \approx 21C$, a $10.5\times$ blow-up in residual-stream traffic. Since decode is memory-bound, this — not compute — is the real serving cost of mHC, and it is why the mappings and residual merge must be fused rather than launched as separate elementwise kernels.

# Efficient Attention Kernels & KV Cache




# MoE serving



# Speculative Decoding 

## MTP

## DSpark

## Quantization
NVFP4
EXL3???

TBD: quantization kernels

# Reference

1. DeepSeek-AI. "DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence." *arXiv preprint* [arXiv:2606.19348](https://arxiv.org/abs/2606.19348), 2026.
2. DeepSeek-AI. *DeepSeek-V4-Pro: reference inference implementation* [Computer software]. [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference), 2026.
3. Xie, Z., Wei, Y., Cao, H., Zhao, C., Deng, C., Li, J., Dai, D., Gao, H., Chang, J., Yu, K., Zhao, L., Zhou, S., Xu, Z., Zhang, Z., Zeng, W., Hu, S., Wang, Y., Yuan, J., Wang, L., and Liang, W. "mHC: Manifold-Constrained Hyper-Connections." *arXiv preprint* [arXiv:2512.24880](https://arxiv.org/abs/2512.24880), 2025.