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

"""Pack one audited JSONL shard for Qwen3.8-27B SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path


LOGGER = logging.getLogger(__name__)
CONTRACT_SCHEMA_VERSION = 1
TOKENIZER_CONTRACT_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "special_tokens_map.json",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the packing CLI shared by local and workflow jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True, help="One audited JSONL input shard.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output Parquet file.")
    parser.add_argument(
        "--metadata-path",
        type=Path,
        required=True,
        help="Output packing metadata JSON file.",
    )
    parser.add_argument(
        "--hf-model-path",
        required=True,
        help="Immutable Hugging Face tokenizer/model directory.",
    )
    parser.add_argument(
        "--contract-evidence",
        type=Path,
        help="Evidence JSON emitted by verify_data_contract.py for this exact tokenizer snapshot.",
    )
    parser.add_argument(
        "--contract-evidence-sha256",
        help="Expected lowercase SHA-256 of the exact contract evidence JSON bytes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--packed-sequence-size", type=int, default=4096)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--pad-seq-to-mult", type=int, default=16)
    parser.add_argument("--num-tokenizer-workers", type=int, default=60)
    return parser


def parse_args() -> argparse.Namespace:
    """Parse and validate packing options."""
    parser = build_parser()
    args = parser.parse_args()
    if args.packed_sequence_size < 1 or args.max_seq_length < 1:
        parser.error("sequence lengths must be positive")
    if args.pad_seq_to_mult < 1 or args.num_tokenizer_workers < 1:
        parser.error("--pad-seq-to-mult and --num-tokenizer-workers must be positive")
    if (args.contract_evidence is None) != (args.contract_evidence_sha256 is None):
        parser.error("--contract-evidence and --contract-evidence-sha256 must be provided together")
    if (
        args.contract_evidence_sha256 is not None
        and re.fullmatch(r"[0-9a-f]{64}", args.contract_evidence_sha256) is None
    ):
        parser.error("--contract-evidence-sha256 must be 64 lowercase hex characters")
    return args


def verify_file_sha256(path: Path, expected_sha256: str) -> None:
    """Reject a file whose exact bytes do not match the configured identity."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise ValueError(f"contract evidence does not exist: {path}") from error
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"contract evidence SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")


def tokenizer_contract_fingerprint(hf_model_path: Path) -> str:
    """Hash the tokenizer files that determine chat rendering and token IDs.

    Args:
        hf_model_path: Immutable local Hugging Face model directory.

    Returns:
        A SHA-256 digest over stable relative filenames and contents.

    Raises:
        FileNotFoundError: If the directory or all contract files are absent.
    """
    root = hf_model_path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Hugging Face model directory does not exist: {root}")
    digest = hashlib.sha256()
    matched = 0
    for name in TOKENIZER_CONTRACT_FILES:
        path = root / name
        if not path.is_file():
            continue
        matched += 1
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    if matched == 0:
        raise FileNotFoundError(
            f"No tokenizer contract files found under {root}; expected one of {TOKENIZER_CONTRACT_FILES}"
        )
    return digest.hexdigest()


def _load_contract_evidence(path: Path, *, hf_model_path: Path) -> dict[str, object]:
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"contract evidence does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"contract evidence is not valid JSON: {path}") from error
    if not isinstance(evidence, dict) or evidence.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ValueError(f"contract evidence must use schema_version={CONTRACT_SCHEMA_VERSION}")
    expected_path = str(hf_model_path.resolve())
    if evidence.get("tokenizer_path") != expected_path:
        raise ValueError(
            f"contract evidence tokenizer_path does not match --hf-model-path: {evidence.get('tokenizer_path')!r}"
        )
    expected_fingerprint = tokenizer_contract_fingerprint(hf_model_path)
    if evidence.get("tokenizer_contract_fingerprint") != expected_fingerprint:
        raise ValueError("contract evidence tokenizer fingerprint does not match the current tokenizer snapshot")
    supports = evidence.get("supports")
    if not isinstance(supports, dict):
        raise ValueError("contract evidence must contain a supports object")
    return supports


def validate_source_contract(
    input_path: Path,
    *,
    hf_model_path: Path,
    contract_evidence_path: Path | None,
) -> dict[str, int]:
    """Validate the exact CyberGym chat subset supported by the generic packer.

    Explicit ``use_loss`` is accepted only when it is equivalent to the
    generic assistant-role mask. Explicit ``use_thinking`` is rejected because
    Megatron-Bridge does not forward it into ``apply_chat_template``. Tools and
    ``reasoning_content`` require evidence from the real tokenizer smoke.

    Args:
        input_path: One source JSONL shard.
        hf_model_path: Immutable local tokenizer/model directory.
        contract_evidence_path: Evidence produced by verify_data_contract.py.

    Returns:
        Counts of records and extended fields validated.

    Raises:
        ValueError: If a record is malformed, requests unsupported semantics,
            or lacks matching tokenizer smoke evidence.
    """
    summary = {
        "records": 0,
        "tools": 0,
        "reasoning_content": 0,
        "explicit_use_loss": 0,
    }
    required_support: set[str] = set()
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: each record must be an object")
            if "use_thinking" in record:
                raise ValueError(
                    f"line {line_number}: explicit use_thinking is unsupported because the generic packer "
                    "does not forward it to the tokenizer"
                )
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"line {line_number}: messages must be a non-empty list")
            tools = record.get("tools")
            if tools is not None:
                if not isinstance(tools, list):
                    raise ValueError(f"line {line_number}: tools must be a list when present")
                if tools:
                    summary["tools"] += 1
                    required_support.add("tools")
            for message_index, message in enumerate(messages):
                location = f"line {line_number} message {message_index}"
                if not isinstance(message, dict):
                    raise ValueError(f"{location}: message must be an object")
                role = message.get("role")
                if not isinstance(role, str) or not role:
                    raise ValueError(f"{location}: role must be a non-empty string")
                if "use_thinking" in message:
                    raise ValueError(f"{location}: message-level use_thinking is unsupported by the generic packer")
                if "use_loss" in message:
                    use_loss = message["use_loss"]
                    if not isinstance(use_loss, bool):
                        raise ValueError(f"{location}: use_loss must be boolean")
                    if use_loss != (role == "assistant"):
                        raise ValueError(
                            f"{location}: use_loss must be role-equivalent (true only for assistant messages)"
                        )
                    summary["explicit_use_loss"] += 1
                    required_support.add("role_equivalent_use_loss")
                if "reasoning_content" in message:
                    reasoning_content = message["reasoning_content"]
                    if reasoning_content is not None and not isinstance(reasoning_content, str):
                        raise ValueError(f"{location}: reasoning_content must be a string or null")
                    summary["reasoning_content"] += 1
                    required_support.add("reasoning_content")
            summary["records"] += 1
    if summary["records"] == 0:
        raise ValueError(f"No JSON records found in {input_path}")
    if required_support:
        if contract_evidence_path is None:
            raise ValueError(
                "extended chat fields require contract evidence from verify_data_contract.py for the exact tokenizer"
            )
        supports = _load_contract_evidence(contract_evidence_path, hf_model_path=hf_model_path)
        required_support.add("packed_loss_mask_roundtrip")
        missing = sorted(name for name in required_support if supports.get(name) is not True)
        if missing:
            raise ValueError(f"contract evidence does not prove required support: {', '.join(missing)}")
    return summary


def main() -> None:
    """Pack the requested audited shard without assigning a validation role."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from megatron.bridge.data.datasets.packed_sequence import (
        prepare_packed_sequence_data,
    )
    from megatron.bridge.training.config import TokenizerConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

    args = parse_args()
    if not args.input_path.is_file():
        raise FileNotFoundError(f"Input shard does not exist: {args.input_path}")
    if args.contract_evidence is not None:
        assert args.contract_evidence_sha256 is not None
        verify_file_sha256(args.contract_evidence, args.contract_evidence_sha256)
    summary = validate_source_contract(
        args.input_path,
        hf_model_path=Path(args.hf_model_path),
        contract_evidence_path=args.contract_evidence,
    )
    LOGGER.info("Validated input contract: %s", summary)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer(
        TokenizerConfig(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=args.hf_model_path)
    )
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
