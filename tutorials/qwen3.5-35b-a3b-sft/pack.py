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

"""Pack one arbitrary JSONL shard for Qwen3.5-35B-A3B SFT."""

from __future__ import annotations

import argparse
from pathlib import Path

from megatron.bridge.data.datasets.packed_sequence import prepare_packed_sequence_data
from megatron.bridge.training.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse the input and output locations for one packing task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True, help="One JSONL input shard.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output Parquet file.")
    parser.add_argument("--metadata-path", type=Path, required=True, help="Output packing metadata JSON file.")
    parser.add_argument("--hf-model-path", required=True, help="Hugging Face tokenizer/model directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--packed-sequence-size", type=int, default=262144)
    parser.add_argument("--max-seq-length", type=int, default=262144)
    parser.add_argument("--pad-seq-to-mult", type=int, default=16)
    parser.add_argument("--num-tokenizer-workers", type=int, default=60)
    args = parser.parse_args()
    if args.packed_sequence_size < 1 or args.max_seq_length < 1:
        parser.error("sequence lengths must be positive")
    if args.pad_seq_to_mult < 1 or args.num_tokenizer_workers < 1:
        parser.error("--pad-seq-to-mult and --num-tokenizer-workers must be positive")
    return args


def main() -> None:
    """Pack the requested shard without assigning it a train or validation role."""
    args = parse_args()
    if not args.input_path.is_file():
        raise FileNotFoundError(f"Input shard does not exist: {args.input_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=args.hf_model_path))
    prepare_packed_sequence_data(
        input_path=args.input_path,
        output_path=args.output_path,
        output_metadata_path=args.metadata_path,
        packed_sequence_size=args.packed_sequence_size,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        dataset_kwargs={"chat": True, "use_hf_tokenizer_chat_template": True},
        pad_seq_to_mult=args.pad_seq_to_mult,
        num_tokenizer_workers=args.num_tokenizer_workers,
    )


if __name__ == "__main__":
    main()
