import torch
import triton
import triton.language as tl

# ==============================================================================
# 1. ORIGAMI V2 KERNEL: CONTIGUOUS 8KB TILES + SINGLE-LAUNCH SPLIT-K REDUCTION
# ==============================================================================
@triton.jit
def origami_v2_matmul_kernel(
    X_ptr,            # Input activations [M, K] (BF16)
    W_hi_ptr,         # Packed weights [N // BLOCK_N, K, BLOCK_N] (BF16)
    W_lo_ptr,         # Residual weights for FP32 router [N // BLOCK_N, K, BLOCK_N] (BF16)
    Out_ptr,          # Final Output [M, N] (BF16 or FP32)
    Partials_ptr,     # Split-K workspace [SPLIT_K, M_MAX, N_MAX] (FP32)
    Locks_ptr,        # Atomic completion counters per (pid_m, pid_n) tile (INT32)
    M, N, K,
    stride_xm, stride_xk,
    stride_wt, stride_wk,
    IS_ROUTER: tl.constexpr,
    SPLIT_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    num_pid_n = tl.num_programs(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m[:, None] < M

    # Start of this fold's K slice
    k_start = pid_k * K_PER_SPLIT
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k[None, :]) * stride_xk

    # W is pre-packed as [N // BLOCK_N, K, BLOCK_N] -> innermost stride is 1 (100% coalesced)
    w_base_offset = pid_n * stride_wt + (k_start + offs_k[:, None]) * stride_wk + tl.arange(0, BLOCK_N)[None, :]
    w_hi_ptrs = W_hi_ptr + w_base_offset
    if IS_ROUTER:
        w_lo_ptrs = W_lo_ptr + w_base_offset

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Software-pipelined hot loop: zero branches, zero indirection
    for _ in range(0, K_PER_SPLIT, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m, other=0.0)
        w_hi = tl.load(w_hi_ptrs)
        acc += tl.dot(x, w_hi)

        if IS_ROUTER:
            w_lo = tl.load(w_lo_ptrs)
            acc += tl.dot(x, w_lo)
            w_lo_ptrs += BLOCK_K * stride_wk

        x_ptrs += BLOCK_K * stride_xk
        w_hi_ptrs += BLOCK_K * stride_wk

    # Fast Path: Single fold writes directly to Out_ptr
    if SPLIT_K == 1:
        out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
        if IS_ROUTER:
            tl.store(out_ptrs, acc, mask=mask_m)
        else:
            tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m)
        return

    # Split-Fold Path: Store partial crease into L2 cache (.cg bypasses per-SM L1)
    partial_stride_s = M * N
    partial_offset = offs_m[:, None] * N + offs_n[None, :]
    my_partial_ptrs = Partials_ptr + pid_k * partial_stride_s + partial_offset
    tl.store(my_partial_ptrs, acc, mask=mask_m, cache_modifier=".cg")

    # Ensure all warps in this block finish writing to L2 before signaling completion
    tl.debug_barrier()

    # The last fold team to arrive at (pid_m, pid_n) unfolds and writes the final output
    lock_idx = pid_m * num_pid_n + pid_n
    arrived = tl.atomic_add(Locks_ptr + lock_idx, 1, sem="acq_rel")

    if arrived == (SPLIT_K - 1):
        total_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for s in range(0, SPLIT_K):
            s_ptrs = Partials_ptr + s * partial_stride_s + partial_offset
            total_acc += tl.load(s_ptrs, mask=mask_m, other=0.0, cache_modifier=".cg")

        out_ptrs = Out_ptr + partial_offset
        if IS_ROUTER:
            tl.store(out_ptrs, total_acc, mask=mask_m)
        else:
            tl.store(out_ptrs, total_acc.to(tl.bfloat16), mask=mask_m)

        # Wait for all warps to finish reading Partials_ptr before resetting the lock
        tl.debug_barrier()
        tl.atomic_xchg(Locks_ptr + lock_idx, 0, sem="release")


# ==============================================================================
# 2. UNTIMED PREPARE PHASE (Up to 300s — Supports up to 32-Way Split-K)
# ==============================================================================
def prepare(weights: dict) -> dict:
    BLOCK_N = 64
    state = {"layers": {}, "partials": None, "locks": None}
    device = None

    for name, tensor in weights.items():
        if tensor is None:
            continue
        device = tensor.device
        N, K = tensor.shape

        # Pack [N, K] -> [N // 64, K, 64] so innermost N-tile stride is 1
        w_packed = tensor.view(N // BLOCK_N, BLOCK_N, K).permute(0, 2, 1).contiguous()

        if name == "router":
            # Lossless BF16 High + Low split for exact FP32 accuracy on Tensor Cores
            w_hi = w_packed.to(torch.bfloat16).contiguous()
            w_lo = (w_packed - w_hi.to(torch.float32)).to(torch.bfloat16).contiguous()
        else:
            w_hi = w_packed.contiguous()
            w_lo = w_hi

        state["layers"][name] = {
            "w_hi": w_hi,
            "w_lo": w_lo,
            "K": K,
            "max_N": N,
            "stride_wt": w_hi.stride(0),
            "stride_wk": w_hi.stride(1),
        }

    if device is not None:
        # 32-split scratchpad allows router (4 tiles * 32 = 128 blocks) to fill 128/132 SMs
        state["partials"] = torch.empty((32, 256, 8192), dtype=torch.float32, device=device)
        state["locks"] = torch.zeros((4096,), dtype=torch.int32, device=device)

    return state


# ==============================================================================
# 3. TIMED EVALUATION PHASE (Layer-Specific 128-Block Wave Quantization)
# ==============================================================================
def kernel_fn(data: dict, state: dict) -> torch.Tensor:
    x = data["x"]
    config = data["config"]
    layer = config["layer"]

    cap = state["layers"][layer]
    M, K = x.shape
    N = config.get("n", cap["max_N"])
    is_router = (layer == "router")

    # Keep M <= 32 in a single M-tile (num_pid_m = 1) so N-tiles * SPLIT_K == 128
    BLOCK_M = 16 if M <= 16 else (32 if M <= 32 else 64)
    BLOCK_N = 64
    BLOCK_K = 64

    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = N // BLOCK_N
    base_tiles = num_pid_m * num_pid_n

    # Target ~128 thread blocks (1 full wave across 132 H200 SMs)
    if base_tiles <= 4:          # router (4 tiles) at M <= 32 -> 4 * 32 = 128 blocks
        SPLIT_K = 32
    elif base_tiles <= 8:        # router at M = 128 (8 tiles) -> 8 * 16 = 128 blocks
        SPLIT_K = 16
    elif base_tiles <= 16:       # sliced unpooler (16 tiles) or router at M=256 -> 128 blocks
        SPLIT_K = 8
    elif base_tiles <= 36:       # qkv_gate (36 tiles) at M <= 16 -> 144 blocks (or 72 at M=32)
        SPLIT_K = 4 if M <= 16 else 2
    elif base_tiles <= 64:       # pooler (64 tiles) at M <= 32 -> 64 * 2 = 128 blocks
        SPLIT_K = 2
    else:                        # o (128 tiles), full unpooler (128 tiles), and verify batches
        SPLIT_K = 1

    out_dtype = torch.float32 if is_router else torch.bfloat16
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)

    grid = (num_pid_m, num_pid_n, SPLIT_K)

    origami_v2_matmul_kernel[grid](
        x,
        cap["w_hi"],
        cap["w_lo"],
        out,
        state["partials"],
        state["locks"],
        M, N, K,
        x.stride(0), x.stride(1),
        cap["stride_wt"], cap["stride_wk"],
        IS_ROUTER=is_router,
        SPLIT_K=SPLIT_K,
        K_PER_SPLIT=K // SPLIT_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    return out
