#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Name: origami_keyblade.py
Author: Hector Rodriguez
Description: Brief description of what your program does.

Copyright 2026 Hector E. Rodriguez

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton
import triton.language as tl

# ==============================================================================
# 1. ORIGAMI KEYBLADE: GEOMETRIC CRYSTAL STREAMING + L2 CACHE HINTS
# ==============================================================================
@triton.jit
def origami_v4_crystal_kernel(
    X_ptr,            # Input activations [M, K] (BF16)
    W_hi_ptr,         # Folded Crystal [N_TILES, K, BLOCK_N] (BF16)
    W_lo_ptr,         # Residual Crystal for FP32 router (BF16)
    Out_ptr,          # Pre-allocated Output [M, N] (BF16 or FP32)
    Partials_ptr,     # Tightly-Packed L2 Workspace [SPLIT_K, M * N] (FP32)
    Locks_ptr,        # Atomic completion counters per 1D tile (INT32)
    M, N, K,
    stride_xm,
    stride_wn,        # Stride between N-tiles in the crystal (K * 64)
    IS_ROUTER: tl.constexpr,
    SPLIT_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # 1. Diagonal Crease Swizzling: keeps X hot in L2 across neighboring SMs
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = N // BLOCK_N

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m[:, None] < M

    # 2. Constant-Step Crystal Streaming (Zero runtime stride multiplications in loop)
    k_start = pid_k * K_PER_SPLIT
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k[None, :])

    CRYSTAL_STEP: tl.constexpr = BLOCK_K * BLOCK_N  # Constant 4096-element jump!
    w_offset = pid_n * stride_wn + k_start * BLOCK_N + offs_k[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    w_hi_ptrs = W_hi_ptr + w_offset
    if IS_ROUTER:
        w_lo_ptrs = W_lo_ptr + w_offset

    tl.multiple_of(offs_k, 16)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Pin X in L2 ("evict_last") while streaming W through once ("evict_first")
    for _ in range(0, K_PER_SPLIT, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m, other=0.0, eviction_policy="evict_last")
        w_hi = tl.load(w_hi_ptrs, eviction_policy="evict_first")
        acc += tl.dot(x, w_hi)

        if IS_ROUTER:
            w_lo = tl.load(w_lo_ptrs, eviction_policy="evict_first")
            acc += tl.dot(x, w_lo)
            w_lo_ptrs += CRYSTAL_STEP

        x_ptrs += BLOCK_K
        w_hi_ptrs += CRYSTAL_STEP

    out_offset = offs_m[:, None] * N + offs_n[None, :]

    # Fast Path: Single-wave layers (o, full unpooler, verify M>=128) write directly
    if SPLIT_K == 1:
        out_ptrs = Out_ptr + out_offset
        if IS_ROUTER:
            tl.store(out_ptrs, acc, mask=mask_m)
        else:
            tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask_m)
        return

    # Split-Fold Path: Tightly-packed L2 slice (M * N elements per split)
    mn_stride = M * N
    my_partial_ptrs = Partials_ptr + pid_k * mn_stride + out_offset
    tl.store(my_partial_ptrs, acc, mask=mask_m, cache_modifier=".cg")

    tl.debug_barrier()
    arrived = tl.atomic_add(Locks_ptr + pid, 1, sem="acq_rel")

    if arrived == (SPLIT_K - 1):
        total_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for s in range(0, SPLIT_K):
            s_ptrs = Partials_ptr + s * mn_stride + out_offset
            total_acc += tl.load(s_ptrs, mask=mask_m, other=0.0, cache_modifier=".cg")

        out_ptrs = Out_ptr + out_offset
        if IS_ROUTER:
            tl.store(out_ptrs, total_acc, mask=mask_m)
        else:
            tl.store(out_ptrs, total_acc.to(tl.bfloat16), mask=mask_m)

        tl.debug_barrier()
        tl.atomic_xchg(Locks_ptr + pid, 0, sem="release")


# ==============================================================================
# 2. UNTIMED PREPARE PHASE: Builds 4D Geometric Crystals & Static Dispatch Table
# ==============================================================================
def prepare(weights: dict) -> dict:
    BLOCK_N = 64
    state = {"dispatch_table": {}, "partials": None, "locks": None}
    device = None

    layers = {}
    for name, tensor in weights.items():
        if tensor is None:
            continue
        device = tensor.device
        N, K = tensor.shape

        # Fold [N, K] into contiguous [N // 64, K, 64] crystal bricks
        w_crystal = tensor.view(N // BLOCK_N, BLOCK_N, K).permute(0, 2, 1).contiguous()

        if name == "router":
            w_hi = w_crystal.to(torch.bfloat16).contiguous()
            w_lo = (w_crystal - w_hi.to(torch.float32)).to(torch.bfloat16).contiguous()
        else:
            w_hi = w_crystal.contiguous()
            w_lo = w_hi

        layers[name] = {
            "w_hi": w_hi,
            "w_lo": w_lo,
            "K": K,
            "max_N": N,
            "stride_wn": w_hi.stride(0),
        }

    if device is not None:
        state["partials"] = torch.empty((32 * 256 * 8192,), dtype=torch.float32, device=device)
        state["locks"] = torch.zeros((4096,), dtype=torch.int32, device=device)

        out_bf16_pool = torch.empty((256 * 8192,), dtype=torch.bfloat16, device=device)
        out_fp32_pool = torch.empty((256 * 256,), dtype=torch.float32, device=device)

        for name, cap in layers.items():
            is_router = (name == "router")
            K = cap["K"]
            for m in (1, 8, 16, 32, 128, 256):
                n_choices = (1024, 8192) if name == "unpooler" else (cap["max_N"],)
                for N in n_choices:
                    BLOCK_M = 16 if m <= 16 else (32 if m <= 32 else 64)
                    num_pid_m = (m + BLOCK_M - 1) // BLOCK_M
                    num_pid_n = N // BLOCK_N
                    base_tiles = num_pid_m * num_pid_n

                    if base_tiles <= 4:
                        SPLIT_K = 32
                    elif base_tiles <= 8:
                        SPLIT_K = 16
                    elif base_tiles <= 16:
                        SPLIT_K = 8
                    elif base_tiles <= 36:
                        SPLIT_K = 4 if m <= 16 else 2
                    elif base_tiles <= 64:
                        SPLIT_K = 2
                    else:
                        SPLIT_K = 1

                    pool = out_fp32_pool if is_router else out_bf16_pool
                    out_view = pool[:m * N].view(m, N)
                    K_PER_SPLIT = K // SPLIT_K
                    num_stages = 4 if K_PER_SPLIT >= 512 else 3

                    state["dispatch_table"][(name, m, N)] = (
                        cap["w_hi"],
                        cap["w_lo"],
                        out_view,
                        K,
                        cap["stride_wn"],
                        is_router,
                        SPLIT_K,
                        K_PER_SPLIT,
                        BLOCK_M,
                        (base_tiles, SPLIT_K),
                        num_stages,
                    )

    return state


# ==============================================================================
# 3. TIMED EVALUATION PHASE: O(1) Tuple Unpack, Zero Torch Overhead
# ==============================================================================
def kernel_fn(data: dict, state: dict) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]

    (
        w_hi, w_lo, out, K, stride_wn,
        is_router, SPLIT_K, K_PER_SPLIT, BLOCK_M, grid, num_stages
    ) = state["dispatch_table"][(cfg["layer"], cfg["m"], cfg["n"])]

    origami_v4_crystal_kernel[grid](
        x,
        w_hi,
        w_lo,
        out,
        state["partials"],
        state["locks"],
        cfg["m"], cfg["n"], K,
        x.stride(0),
        stride_wn,
        IS_ROUTER=is_router,
        SPLIT_K=SPLIT_K,
        K_PER_SPLIT=K_PER_SPLIT,
        BLOCK_M=BLOCK_M,
        BLOCK_N=64,
        BLOCK_K=64,
        GROUP_M=8,
        num_warps=4,
        num_stages=num_stages,
    )

    return out
