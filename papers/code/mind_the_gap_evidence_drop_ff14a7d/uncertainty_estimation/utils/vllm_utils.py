"""Shared vLLM tensor-parallel / NCCL setup utilities."""
from __future__ import annotations

import os
import socket
from argparse import ArgumentParser

import torch


def setup_vllm_distributed_and_tp(args) -> int:
    """
    Set MASTER_PORT, NCCL env; return resolved tensor_parallel_size.
    Expects args to have optional: master_port, tensor_parallel_size, gpu_memory_utilization.
    """
    if getattr(args, "master_port", None) is not None:
        os.environ["MASTER_PORT"] = str(args.master_port)
        print(f"Using master port: {args.master_port}")
    else:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            free_port = s.getsockname()[1]
        os.environ["MASTER_PORT"] = str(free_port)
        print(f"Auto-selected master port: {free_port}")

    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No CUDA GPUs visible; vLLM inference requires at least one GPU.")

    if getattr(args, "tensor_parallel_size", None) is not None:
        tensor_parallel = args.tensor_parallel_size
        if tensor_parallel > num_gpus:
            print(
                f"Warning: tensor_parallel_size ({tensor_parallel}) > "
                f"available GPUs ({num_gpus}), using {num_gpus}"
            )
            tensor_parallel = num_gpus
        if tensor_parallel < 1:
            print("Warning: tensor_parallel_size < 1, using 1")
            tensor_parallel = 1
    else:
        tensor_parallel = num_gpus

    print(f"Using tensor_parallel_size={tensor_parallel} (visible GPUs: {num_gpus})")
    return tensor_parallel


def add_vllm_tp_args(parser: ArgumentParser, default_gpu_memory_utilization: float) -> None:
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=None,
        help="Number of GPUs for tensor parallelism; defaults to all visible GPUs",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=default_gpu_memory_utilization,
        help="vLLM GPU memory utilization ratio",
    )
    parser.add_argument(
        "--master_port",
        type=int,
        default=None,
        help="Master port for multi-GPU communication; auto-selected if not specified",
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=20,
        help="Maximum number of concurrent sequences for scheduler (default 20)",
    )
    parser.add_argument(
        "--swap_space",
        type=int,
        default=16,
        help="CPU swap space in GiB to relieve KV cache pressure (default 16)",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Maximum context length for KV planning (default 8192)",
    )
    parser.add_argument(
        "--disable_prefix_caching",
        action="store_true",
        default=False,
        help="Disable prefix caching (enabled by default)",
    )


def llm_common_kwargs(args, tensor_parallel: int) -> dict:
    kw: dict = {
        "model": args.model,
        "tensor_parallel_size": tensor_parallel,
        "disable_custom_all_reduce": tensor_parallel > 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tokenizer": args.model,
        "enable_prefix_caching": not getattr(args, "disable_prefix_caching", False),
        "swap_space": getattr(args, "swap_space", 16),
        "max_num_seqs": getattr(args, "max_num_seqs", 20),
        "max_model_len": getattr(args, "max_model_len", 8192),
    }
    return kw
