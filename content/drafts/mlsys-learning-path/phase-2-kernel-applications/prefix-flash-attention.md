

## The workload

Shared prefix, large batch of independent queries.

Per decode forward:

- **Q** — `[N, H, Sq, D]`: `N` independent query groups, `Sq` query tokens each, `H` heads, `D` head_dim, bf16.
- **Context prefix** — `[H, P, D]`: the shared context, `P` prefix token length, the **same for all `N` query groups**.
- **Local block** — each query group also attends to its own `Sq` decode tokens under a 2×2 block-diagonal / prompt-causal mask.
- **Dimentions** — `P >> Sq` long prefix, short local context, `N >> P` is large batch size.



The decode mask is *all-attend over the prefix* plus a *tiny block-diagonal tail over the local
tokens*. Q and K are already `LayerNormNoAffine`-normed before attention, so the kernel only does
`softmax(Q·Kᵀ·scale)·V` with `scale = head_dim**-0.5`.

## Roofline analysis

Computation complexity:

- `N` query groups
- `H` heads
- `Sq` query tokens each
- `P` prefix tokens
- `D` head_dim

Total computation:

- `Q·Kᵀ` ~ `2H·N·D·Sq·P` flops
- `A·V` ~ `2H·N·D·Sq·P` flops
- Total ~ `4H·N·D·Sq·P` flops

Memory I/O: Assuming bf16
- `Q` ~ `2H·N·Sq·D` bytes
- `K` ~ `2H·P·D` bytes
- `V` ~ `2H·P·D` bytes
- `O` ~ `2H·N·Sq·D` bytes
- Total ~ `2H·N·Sq·D + 4H·P·D + 2H·N·Sq·D` bytes = `6H·N·Sq·D + 4H·P·D` bytes

Computational intensity = flops / bytes = `4H·N·D·Sq·P / (6H·N·Sq·D + 4H·P·D)` = `2NSqP / (3NSq + 2P)` 


Roofline model of L40S:
- BF16	362.05 TFLOPS
- Memory bw	864 GB/s

Roofline mem to compute turning point = 422.52 FLOPS/B

Given Sq=1, P=1000, N=10000, compute density = 625 FLOPS/B > 422.52 FLOPS/B, so it's computational bounded kernel.