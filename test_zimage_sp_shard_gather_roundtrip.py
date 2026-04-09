#!/usr/bin/env python3
"""Test that ZImagePipelineConfig.shard_latents_for_sp -> gather_latents_for_sp
is an identity (round-trip preserves the original tensor) under SP=2.

Spawns 2 ranks via torch.multiprocessing and uses NCCL.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_repo_python = Path(__file__).resolve().parent / "python"
if _repo_python.is_dir():
    sys.path.insert(0, str(_repo_python))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _worker(rank: int, world_size: int, seed: int, H: int, W: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29512"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Initialize sglang's SP parallel state to match the runtime usage.
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    initialize_model_parallel(
        tensor_parallel_degree=1,
        sequence_parallel_degree=world_size,
        ulysses_degree=world_size,
        ring_degree=1,
    )

    from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
        ZImagePipelineConfig,
    )

    cfg = ZImagePipelineConfig()

    # Construct a fake "batch" providing the attributes that the SP plan needs.
    class _Batch:
        pass

    batch = _Batch()
    batch.height = H
    batch.width = W
    batch.raw_latent_shape = (1, 16, 1, H // 8, W // 8)

    full_shape = (1, 16, 1, H // 8, W // 8)
    # Build full tensor identically on every rank.
    g = torch.Generator(device="cuda").manual_seed(seed)
    full = torch.randn(full_shape, generator=g, device=f"cuda:{rank}", dtype=torch.float32)

    # Shard then gather, then compare against original.
    sharded, did = cfg.shard_latents_for_sp(batch, full)
    if rank == 0:
        print(
            f"[rank0] full={tuple(full.shape)} sharded={tuple(sharded.shape)} did_shard={did}",
            flush=True,
        )
    gathered = cfg.gather_latents_for_sp(sharded, batch=batch)
    if rank == 0:
        print(f"[rank0] gathered={tuple(gathered.shape)}", flush=True)

    same_shape = tuple(gathered.shape) == tuple(full.shape)
    max_abs = (gathered - full).abs().max().item() if same_shape else float("inf")
    if rank == 0:
        print(f"[rank0] same_shape={same_shape} max_abs_diff={max_abs}", flush=True)
        if not same_shape or max_abs > 0:
            print("[rank0] FAIL roundtrip", flush=True)
        else:
            print("[rank0] PASS roundtrip", flush=True)

    # Also test 4D and packed [B*T, C, F, H, W] shapes used by debug gather.
    full4 = torch.randn((2, 16, H // 8, W // 8), generator=g, device=f"cuda:{rank}", dtype=torch.float32)
    sharded4, _ = cfg.shard_latents_for_sp(batch, full4.unsqueeze(2))  # 5D
    sharded4 = sharded4.squeeze(2)  # back to 4D
    gathered4 = cfg.gather_latents_for_sp(sharded4, batch=batch)
    if rank == 0:
        ok4 = tuple(gathered4.shape) == tuple(full4.shape)
        d4 = (gathered4 - full4).abs().max().item() if ok4 else float("inf")
        print(f"[rank0] 4D roundtrip same_shape={ok4} max_abs_diff={d4}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    # Use heights that exercise odd token splits (33 H tokens at 1024).
    mp.spawn(_worker, args=(2, 42, 1024, 1024), nprocs=2, join=True)
