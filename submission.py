import torch
import triton
import triton.language as tl

# ==============================================================================
# 1. TRITON KERNEL: SWIZZLED MICRO-TILE MATMUL
# ==============================================================================
@triton.jit
def swizzled_weight_matmul_kernel(
    X_ptr,          # Input activations [M, K]
    W_swizzled_ptr, # Swizzled weights [Rows, Cols // TILE_SIZE, TILE_SIZE]
    Out_ptr,        # Output matrix [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wt, stride_wts,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    tile_idx = offs_n // TILE_SIZE
    intra_tile_offset = offs_n % TILE_SIZE

    w_ptrs = W_swizzled_ptr + (
        offs_k[:, None] * stride_wn + 
        tile_idx[None, :] * stride_wt + 
        intra_tile_offset[None, :] * stride_wts
    )

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < (K - k), other=0.0)
        acc += tl.dot(x, w)
        
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wn

    out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ==============================================================================
# 2. UNTIMED PREPARE PHASE (Up to 300 seconds)
# ==============================================================================
def prepare(weights: dict) -> dict:
    """
    Losslessly swizzles weight matrices into micro-tiles to eliminate 
    shared-memory bank conflicts during kernel execution.
    """
    optimized_weights = {}
    TILE_SIZE = 32
    
    for name, tensor in weights.items():
        if tensor is None:
            continue
            
        t = tensor.contiguous()
        
        # Apply layout swizzling to 2D weight tensors
        if t.ndim == 2:
            rows, cols = t.shape
            if cols % TILE_SIZE == 0:
                blocked = t.view(rows, cols // TILE_SIZE, TILE_SIZE)
                t_swizzled = blocked.permute(0, 2, 1).contiguous()
                optimized_weights[name] = t_swizzled
            else:
                optimized_weights[name] = t
        else:
            optimized_weights[name] = t
            
    return optimized_weights


# ==============================================================================
# 3. TIMED EVALUATION PHASE (CUDA Graph Capturable)
# ==============================================================================
def kernel_fn(data: dict) -> torch.Tensor:
    """
    Executes the swizzled kernel against incoming runtime inputs.
    Must avoid host synchronizations (.item(), .cpu(), etc.) to support CUDA graphs.
    """
    # Extract inputs and pre-swizzled weights
    x = data.get("x")              # Input tensor [M, K]
    w = data.get("weights")        # Pre-swizzled weights dictionary from prepare()
    target_weight_name = "weight"  # Adjust key based on problem schema
    
    # Fallback if weights are passed dynamically or stored differently in the task harness
    w_swizzled = w.get(target_weight_name) if isinstance(w, dict) else data.get("weight")

    M, K = x.shape
    # If swizzled, shape is [Rows, Num_Tiles, Tile_Size] -> N = Rows * Tile_Size (or derived columns)
    TILE_SIZE = 32
    N = w_swizzled.shape[0] * TILE_SIZE if w_swizzled.ndim == 3 else w_swizzled.shape[1]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    swizzled_weight_matmul_kernel[grid](
        x, w_swizzled, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w_swizzled.stride(0), w_swizzled.stride(1), w_swizzled.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        TILE_SIZE=TILE_SIZE,
    )
    
    return out
