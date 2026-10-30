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

"""Verify Qwen3.8 chat rendering and packed loss-mask roundtrip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pack import CONTRACT_SCHEMA_VERSION, tokenizer_contract_fingerprint


REASONING_SENTINEL = "QWEN38_REASONING_SENTINEL_7A91"
TOOL_SENTINEL = "qwen38_contract_probe"


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded tokenizer-contract smoke CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-path", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    return parser


def main() -> None:
    """Exercise the real tokenizer, generic packer, and Parquet loss mask."""
    from pyarrow.parquet import read_table

    from megatron.bridge.data.datasets.packed_sequence import (
        prepare_packed_sequence_data,
        tokenize_dataset,
    )
    from megatron.bridge.training.config import TokenizerConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

    args = build_parser().parse_args()
    model_path = args.hf_model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Immutable local tokenizer/model directory does not exist: {model_path}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.evidence_path.parent.mkdir(parents=True, exist_ok=True)

    tools = [
        {
            "type": "function",
            "function": {
                "name": TOOL_SENTINEL,
                "description": "Contract probe only",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Use the available probe.", "use_loss": False},
        {
            "role": "assistant",
            "reasoning_content": REASONING_SENTINEL,
            "content": "Probe complete.",
            "use_loss": True,
        },
    ]
    record = {"messages": messages, "tools": tools}
    source_path = args.work_dir / "contract-probe.jsonl"
    output_path = args.work_dir / "contract-probe.parquet"
    metadata_path = args.work_dir / "contract-probe.metadata.json"
    source_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    tokenizer = build_tokenizer(
        TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=str(model_path))
    )
    hf_tokenizer = tokenizer._tokenizer
    rendered = hf_tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    if REASONING_SENTINEL not in rendered:
        raise RuntimeError("Qwen3.8 chat template discarded reasoning_content")
    if TOOL_SENTINEL not in rendered:
        raise RuntimeError("Qwen3.8 chat template discarded the tools schema")

    dataset_kwargs = {"chat": True, "use_hf_tokenizer_chat_template": True}
    tokenized = tokenize_dataset(
        source_path,
        tokenizer,
        max_seq_length=512,
        seed=42,
        dataset_kwargs=dict(dataset_kwargs),
        pad_seq_to_mult=1,
        num_tokenizer_workers=1,
    )
    if len(tokenized) != 1:
        raise RuntimeError(f"Expected one tokenized contract record, received {len(tokenized)}")
    tokenized_record = tokenized[0]
    input_ids = tokenized_record["input_ids"].tolist()
    assistant_mask = [bool(value) for value in tokenized_record["loss_mask"].tolist()]
    if not any(assistant_mask) or all(assistant_mask):
        raise RuntimeError("Assistant mask must contain both supervised assistant and masked context tokens")

    prepare_packed_sequence_data(
        input_path=source_path,
        output_path=output_path,
        output_metadata_path=metadata_path,
        packed_sequence_size=512,
        tokenizer=tokenizer,
        max_seq_length=512,
        seed=42,
        dataset_kwargs=dict(dataset_kwargs),
        pad_seq_to_mult=1,
        num_tokenizer_workers=1,
    )
    table = read_table(output_path, columns=["input_ids", "loss_mask"])
    if table.num_rows != 1:
        raise RuntimeError(f"Expected one packed contract row, received {table.num_rows}")
    packed_input_ids = table.column("input_ids")[0].as_py()
    packed_loss_mask = [bool(value) for value in table.column("loss_mask")[0].as_py()]
    expected_packed_mask = assistant_mask[1:] + [False]
    if packed_input_ids != input_ids:
        raise RuntimeError("Packed input_ids do not roundtrip the tokenized contract record")
    if packed_loss_mask != expected_packed_mask:
        raise RuntimeError("Packed loss_mask does not match the generic packer's shifted assistant mask")

    evidence = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "tokenizer_path": str(model_path),
        "tokenizer_contract_fingerprint": tokenizer_contract_fingerprint(model_path),
        "supports": {
            "tools": True,
            "reasoning_content": True,
            "role_equivalent_use_loss": True,
            "packed_loss_mask_roundtrip": True,
            "explicit_use_thinking": False,
        },
        "artifacts": {
            "source": str(source_path.resolve()),
            "packed_parquet": str(output_path.resolve()),
            "packing_metadata": str(metadata_path.resolve()),
        },
    }
    temporary_path = args.evidence_path.with_suffix(args.evidence_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(args.evidence_path)


if __name__ == "__main__":
    main()
