###Yet to test 10/24/2026

import torch
import triton
import triton.language as tl

# ==============================================================================
# 1. ORIGAMI FOLDED KERNEL (3D Parallel Crease Execution + Crease Map)
# ==============================================================================
@triton.jit
def origami_folded_matmul_kernel(
    X_ptr,            # Input activations [M, K] (BF16)
    W_origami_ptr,    # Folded Capsule: [NUM_FOLDS, FOLD_K, N_TILES, TILE_SIZE]
    FoldMap_ptr,      # Crease Map: [NUM_FOLDS, 2] -> (physical_fold_id, sign_flag)
    Out_ptr,          # Output or FP32 Accumulator [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wf, stride_wk, stride_wt, stride_ws,
    NUM_FOLDS: tl.constexpr,
    FOLD_K: tl.constexpr,
    SPLIT_FOLDS: tl.constexpr,
    IS_ATOMIC: tl.constexpr,
    OUT_FP32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_f = tl.program_id(2)  # Which Origami Fold group this worker team owns

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Map N coordinates into bank-conflict-free 32-lane micro-tiles
    tile_idx = offs_n // TILE_SIZE
    intra_tile_offset = offs_n % TILE_SIZE
    w_tile_base = tile_idx[None, :] * stride_wt + intra_tile_offset[None, :] * stride_ws

    # Determine which range of the 64 Origami Folds this worker team processes
    folds_per_split = NUM_FOLDS // SPLIT_FOLDS
    start_fold = pid_f * folds_per_split
    end_fold = start_fold + folds_per_split

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Walk only this team's assigned Origami creases
    for f in range(start_fold, end_fold):
        # Consult the Origami Crease Map: [physical_fold_idx, mirror_sign]
        phys_f = tl.load(FoldMap_ptr + f * 2)
        sign_flag = tl.load(FoldMap_ptr + f * 2 + 1)

        if sign_flag != 0:
            fold_sign = sign_flag.to(tl.float32)
            # Global K offset for input X vs. Physical Fold offset for Weight Capsule
            k_fold_base = f * FOLD_K
            w_fold_ptr = W_origami_ptr + phys_f * stride_wf + w_tile_base

            # Each fold of 128 elements is digested in steps of BLOCK_K (64)
            for k_sub in tl.static_range(0, FOLD_K, BLOCK_K):
                offs_k = tl.arange(0, BLOCK_K) + k_sub
                global_k = k_fold_base + offs_k

                x_ptrs = X_ptr + offs_m[:, None] * stride_xm + global_k[None, :] * stride_xk
                w_ptrs = w_fold_ptr + offs_k[:, None] * stride_wk

                k_mask = global_k < K
                x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
                w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)

                # Multiply in Tensor Cores (cast FP32 router weights to BF16 for Tensor Core dot, or native)
                acc += tl.dot(x, w.to(x.dtype)) * fold_sign

    out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Unfold / Reduce across parallel teams
    if IS_ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=out_mask, sem="relaxed")
    else:
        if OUT_FP32:
            tl.store(out_ptrs, acc, mask=out_mask)
        else:
            tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


# ==============================================================================
# 2. UNTIMED PREPARE PHASE: Build the 64-Fold Capsules & Crease Maps
# ==============================================================================
def prepare(weights: dict) -> dict:
    """
    Folds flat [N, K] weight matrices into 4D Origami Capsules:
    [NUM_FOLDS, FOLD_K (128), N // 32, 32] and builds the Crease Map.
    """
    state = {"layers": {}, "scratch_fp32": None}
    TILE_SIZE = 32
    FOLD_K = 128  # K=8192 -> Exactly 64 Origami Folds of 128 elements

    device = None
    for name, tensor in weights.items():
        if tensor is None:
            continue
        device = tensor.device
        N, K = tensor.shape
        num_folds = K // FOLD_K

        # 1. Transpose from [N, K] to [K, N] so K is the folding axis
        t_kn = tensor.T.contiguous()

        # 2. Fold K into [num_folds, FOLD_K, N] and swizzle N into 32-lane micro-tiles
        folded = t_kn.view(num_folds, FOLD_K, N // TILE_SIZE, TILE_SIZE)
        # Permute inside each fold so adjacent threads hit distinct shared-memory banks
        w_origami = folded.permute(0, 1, 3, 2).contiguous()

        # 3. Construct the Origami Crease Map: [num_folds, 2] -> (physical_fold_id, sign)
        # Detects zero-folds or exact mirror reflections along K to reuse L2 cache lines
        fold_map = torch.zeros((num_folds, 2), dtype=torch.int32, device=device)
        flat_folds = w_origami.view(num_folds, -1)
        
        for f in range(num_folds):
            cur = flat_folds[f]
            if torch.all(cur == 0):
                fold_map[f, 0] = f
                fold_map[f, 1] = 0  # Skip empty zero-crease
                continue
            
            matched = False
            # Check mirror partner (f_mirror = num_folds - 1 - f) if earlier in stack
            mirror_idx = num_folds - 1 - f
            if mirror_idx < f:
                ref = flat_folds[mirror_idx]
                if torch.equal(cur, ref):
                    fold_map[f, 0] = mirror_idx
                    fold_map[f, 1] = 1
                    matched = True
                elif torch.equal(cur, -ref):
                    fold_map[f, 0] = mirror_idx
                    fold_map[f, 1] = -1
                    matched = True

            if not matched:
                fold_map[f, 0] = f
                fold_map[f, 1] = 1

        state["layers"][name] = {
            "w_origami": w_origami,
            "fold_map": fold_map,
            "num_folds": num_folds,
            "K": K,
            "max_N": N,
        }

    # Pre-allocate a reusable FP32 reduction workspace (up to M=256, N=8192) for Split-Fold atomics
    if device is not None:
        state["scratch_fp32"] = torch.zeros((256, 8192), dtype=torch.float32, device=device)

    return state


# ==============================================================================
# 3. TIMED EVALUATION PHASE: Zero-Branch Capsule Dispatch
# ==============================================================================
def kernel_fn(data: dict, state: dict) -> torch.Tensor:
    x = data["x"]
    config = data["config"]
    layer = config["layer"]

    capsule = state["layers"][layer]
    w_origami = capsule["w_origami"]
    fold_map = capsule["fold_map"]
    num_folds = capsule["num_folds"]

    M, K = x.shape
    # Telescoping unpooler: config["n"] is 1024 for m <= 8, or 8192 otherwise
    N = config.get("n", capsule["max_N"])

    is_router = (layer == "router")
    BLOCK_M = 32 if M <= 32 else 64
    BLOCK_N = 64
    BLOCK_K = 64
    TILE_SIZE = 32
    FOLD_K = 128

    # Choose parallel Origami Fold split factor:
    # For small M or narrow N (router/o), split the 64 folds across SMs to fill all 132 H200 cores
    tiles_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    if tiles_mn <= 16 and num_folds >= 8:
        SPLIT_FOLDS = 8
    elif tiles_mn <= 64 and num_folds >= 4:
        SPLIT_FOLDS = 4
    else:
        SPLIT_FOLDS = 1

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
        SPLIT_FOLDS,
    )

    if SPLIT_FOLDS == 1:
        out_dtype = torch.float32 if is_router else torch.bfloat16
        out = torch.empty((M, N), dtype=out_dtype, device=x.device)
        origami_folded_matmul_kernel[grid](
            x, w_origami, fold_map, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w_origami.stride(0), w_origami.stride(1), w_origami.stride(3), w_origami.stride(2),
            NUM_FOLDS=num_folds, FOLD_K=FOLD_K, SPLIT_FOLDS=1,
            IS_ATOMIC=False, OUT_FP32=is_router,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, TILE_SIZE=TILE_SIZE,
        )
        return out
    else:
        # Parallel 3D Fold Reduction: teams atomically fold into FP32 buffer
        if is_router:
            out = torch.zeros((M, N), dtype=torch.float32, device=x.device)
            target_buf = out
        else:
            target_buf = state["scratch_fp32"][:M, :N]
            target_buf.zero_()

        origami_folded_matmul_kernel[grid](
            x, w_origami, fold_map, target_buf,
            M, N, K,
            x.stride(0), x.stride(1),
            w_origami.stride(0), w_origami.stride(1), w_origami.stride(3), w_origami.stride(2),
            NUM_FOLDS=num_folds, FOLD_K=FOLD_K, SPLIT_FOLDS=SPLIT_FOLDS,
            IS_ATOMIC=True, OUT_FP32=True,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, TILE_SIZE=TILE_SIZE,
        )
        return out if is_router else target_buf.to(torch.bfloat16)
