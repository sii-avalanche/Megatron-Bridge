#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run Qwen3.5-35B-A3B SFT from a packed-data root."""

from __future__ import annotations

import argparse
import math
import os
from functools import partial
from pathlib import Path

import torch
from megatron.core._rank_utils import safe_get_rank as get_rank_safe
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.data.datasets.packed_sequence import PackedSequenceSpecs
from megatron.bridge.models.qwen_vl.qwen3_vl_step import forward_step as qwen3_vl_forward_step
from megatron.bridge.recipes.qwen_vl.qwen35_vl import qwen35_vl_35b_a3b_sft_config
from megatron.bridge.training.chunked_linear_cross_entropy import chunked_lm_head_output_processor
from megatron.bridge.training.config import FinetuningDatasetConfig
from megatron.bridge.training.finetune import finetune


SEQ_LENGTH = 262144
GLOBAL_BATCH_SIZE = 64


def parse_args() -> argparse.Namespace:
    """Parse the locations and W&B names that vary between SFT experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-exp-name", required=True)
    parser.add_argument("--packed-sequence-size", type=int, default=SEQ_LENGTH)
    parser.add_argument("--pad-seq-to-mult", type=int, default=16)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--train-iters", type=optional_int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=GLOBAL_BATCH_SIZE)
    parser.add_argument("--save-interval", type=optional_int, default=None)
    parser.add_argument("--lr-warmup-iters", type=optional_int, default=None)
    parser.add_argument("--lr-decay-iters", type=optional_int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--sequence-parallel", type=str_to_bool, default=True)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=8)
    parser.add_argument("--expert-model-parallel-size", type=int, default=8)
    parser.add_argument("--expert-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--rotary-base", type=float, default=10_000_000.0)
    parser.add_argument("--mtp-num-layers", type=int, default=1)
    parser.add_argument("--mtp-use-repeated-layer", type=str_to_bool, default=False)
    parser.add_argument("--mtp-loss-scaling-factor", type=float, default=0.2)
    parser.add_argument("--mtp-loss-scaling-decay-factor", type=float, default=0.8)
    parser.add_argument("--moe-token-dispatcher-type", default="flex")
    parser.add_argument("--moe-flex-dispatcher-backend", default="deepep")
    parser.add_argument("--moe-router-load-balancing-type", default="aux_loss")
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--moe-router-dtype", default="float32")
    parser.add_argument("--moe-router-fusion", type=str_to_bool, default=True)
    parser.add_argument("--recompute-granularity", default="full")
    parser.add_argument("--recompute-method", default="block")
    parser.add_argument("--recompute-num-layers", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--min-learning-rate", type=float, default=5e-6)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--lr-decay-style", default="cosine")
    parser.add_argument("--start-weight-decay", type=float, default=0.1)
    parser.add_argument("--end-weight-decay", type=float, default=0.1)
    parser.add_argument("--weight-decay-incr-style", default="constant")
    parser.add_argument("--calculate-per-token-loss", type=str_to_bool, default=True)
    parser.add_argument("--hf-model-path", required=True)
    parser.add_argument("--megatron-checkpoint-path", required=True)
    parser.add_argument("--checkpoint-load", default=None)
    parser.add_argument("--checkpoint-save-optim", type=str_to_bool, default=False)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    for name in ("packed_sequence_size", "pad_seq_to_mult", "global_batch_size", "micro_batch_size",
                 "tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size",
                 "expert_model_parallel_size", "expert_tensor_parallel_size", "mtp_num_layers",
                 "recompute_num_layers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("train_iters", "save_interval", "lr_warmup_iters", "lr_decay_iters"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive when provided")
    if not args.checkpoint_load:
        args.checkpoint_load = None
    return args


def str_to_bool(value: str) -> bool:
    """Parse booleans passed through the workflow shell command."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def optional_int(value: str) -> int | None:
    """Treat an empty CLI value as an omitted optional integer."""
    return None if value == "" else int(value)


def count_packed_samples(data_dir: Path) -> int:
    """Return the number of packed training sequences without loading their contents."""
    from pyarrow.parquet import ParquetFile

    paths = sorted(data_dir.glob("shard*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No packed training data found in {data_dir}")
    return sum(ParquetFile(path).metadata.num_rows for path in paths)


def checkpoint_interval(train_iters: int, epochs: int) -> int:
    """Return a step interval that saves approximately four checkpoints per epoch."""
    return max(1, math.ceil(train_iters / (epochs * 4)))


def build_config(args: argparse.Namespace, *, train_iters: int, save_interval: int):
    """Build the fixed Qwen3.5-35B-A3B SFT recipe around experiment paths."""
    data_dir = args.data_root / "packed" / f"seq-{args.packed_sequence_size}"
    checkpoint_dir = args.exp_dir / "checkpoints"
    tensorboard_dir = args.exp_dir / "tb_logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("NVTE_USE_CUTLASS_GROUPED_GEMM", "1")
    os.environ.setdefault("WANDB_DIR", str(args.exp_dir / "wandb"))

    cfg = qwen35_vl_35b_a3b_sft_config(hf_path=args.hf_model_path)
    cfg.model.seq_length = args.packed_sequence_size
    cfg.model.tensor_model_parallel_size = args.tensor_model_parallel_size
    cfg.model.sequence_parallel = args.sequence_parallel
    cfg.model.pipeline_model_parallel_size = args.pipeline_model_parallel_size
    cfg.model.context_parallel_size = args.context_parallel_size
    cfg.model.expert_model_parallel_size = args.expert_model_parallel_size
    cfg.model.expert_tensor_parallel_size = args.expert_tensor_parallel_size
    cfg.model.rotary_base = args.rotary_base
    cfg.model.mtp_num_layers = args.mtp_num_layers
    cfg.model.mtp_use_repeated_layer = args.mtp_use_repeated_layer
    cfg.model.mtp_loss_scaling_factor = args.mtp_loss_scaling_factor
    cfg.model.mtp_loss_scaling_decay_factor = args.mtp_loss_scaling_decay_factor
    cfg.model.moe_token_dispatcher_type = args.moe_token_dispatcher_type
    cfg.model.moe_flex_dispatcher_backend = args.moe_flex_dispatcher_backend
    cfg.model.moe_router_load_balancing_type = args.moe_router_load_balancing_type
    cfg.model.attention_backend = getattr(AttnBackend, args.attention_backend)
    cfg.model.moe_router_dtype = torch.float32 if args.moe_router_dtype in {"float32", "fp32"} else args.moe_router_dtype
    cfg.model.moe_router_fusion = args.moe_router_fusion
    cfg.model.recompute_granularity = args.recompute_granularity
    cfg.model.recompute_method = args.recompute_method
    cfg.model.recompute_num_layers = args.recompute_num_layers

    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = args.global_batch_size
    cfg.train.micro_batch_size = args.micro_batch_size
    cfg.optimizer.lr = args.learning_rate
    cfg.optimizer.min_lr = args.min_learning_rate
    cfg.optimizer.adam_beta1 = args.adam_beta1
    cfg.optimizer.adam_beta2 = args.adam_beta2
    cfg.optimizer.weight_decay = args.weight_decay
    cfg.optimizer.clip_grad = args.clip_grad
    cfg.scheduler.lr_decay_style = args.lr_decay_style
    cfg.scheduler.lr_warmup_iters = args.lr_warmup_iters or min(500, max(1, math.ceil(train_iters * 0.1)))
    cfg.scheduler.lr_decay_iters = args.lr_decay_iters or train_iters
    cfg.scheduler.start_weight_decay = args.start_weight_decay
    cfg.scheduler.end_weight_decay = args.end_weight_decay
    cfg.scheduler.weight_decay_incr_style = args.weight_decay_incr_style
    cfg.model.calculate_per_token_loss = args.calculate_per_token_loss
    cfg.ddp.average_in_collective = False
    cfg.tokenizer.tokenizer_model = args.hf_model_path
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.dataset = FinetuningDatasetConfig(
        dataset_root=str(data_dir),
        seq_length=args.packed_sequence_size,
        seed=args.train_seed,
        dataloader_type="batch",
        do_validation=False,
        do_test=False,
        packed_sequence_specs=PackedSequenceSpecs(
            packed_sequence_size=args.packed_sequence_size,
            pad_seq_to_mult=args.pad_seq_to_mult,
            packed_train_data_path=str(data_dir / "train" / "shard*.parquet"),
        ),
        dataset_kwargs={
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
            "pad_to_max_length": True,
        },
    )
    cfg.checkpoint.pretrained_checkpoint = args.megatron_checkpoint_path
    cfg.checkpoint.save = str(checkpoint_dir)
    cfg.checkpoint.load = args.checkpoint_load
    cfg.checkpoint.save_interval = save_interval
    cfg.checkpoint.save_optim = args.checkpoint_save_optim
    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_dir = str(tensorboard_dir)
    cfg.logger.wandb_entity = "avalanche"
    cfg.logger.wandb_project = args.wandb_project
    cfg.logger.wandb_exp_name = args.wandb_exp_name
    cfg.logger.wandb_save_dir = str(args.exp_dir)
    cfg.rng.seed = args.train_seed
    return cfg


def main() -> None:
    """Configure and run distributed fine-tuning."""
    args = parse_args()
    data_dir = args.data_root / "packed" / f"seq-{args.packed_sequence_size}"
    train_iters = args.train_iters or math.ceil(
        count_packed_samples(data_dir / "train") * args.epochs / args.global_batch_size
    )
    cfg = build_config(
        args,
        train_iters=train_iters,
        save_interval=args.save_interval or checkpoint_interval(train_iters, args.epochs),
    )
    if get_rank_safe() == 0:
        cfg.print_yaml()
    forward_step_func = partial(
        qwen3_vl_forward_step,
        output_processor=partial(
            chunked_lm_head_output_processor,
            vocab_chunk_size=31_040,
            mtp_vocab_chunk_size=31_040,
        ),
    )
    finetune(config=cfg, forward_step_func=forward_step_func)


if __name__ == "__main__":
    main()
