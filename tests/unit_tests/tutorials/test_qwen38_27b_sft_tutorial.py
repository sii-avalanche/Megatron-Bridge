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

"""Contract tests for the Qwen3.8-27B dense SFT tutorial."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TUTORIAL_ROOT = REPO_ROOT / "tutorials" / "qwen3.8-27b-sft"


def _load_train_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("qwen38_27b_sft_train", TUTORIAL_ROOT / "train.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pack_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("qwen38_27b_sft_pack", TUTORIAL_ROOT / "pack.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_to_dense_gbs8_research_recipe() -> None:
    """The supported CLI should default to the agreed dense GBS=8 experiment."""
    train = _load_train_module()

    args = train.build_parser().parse_args(
        [
            "--data-root",
            "/data/cybergym",
            "--exp-dir",
            "/experiments/qwen38",
            "--epochs",
            "1",
            "--wandb-project",
            "cybergym",
            "--wandb-exp-name",
            "qwen38-smoke",
            "--hf-model-path",
            "/models/qwen38-27b",
            "--megatron-checkpoint-path",
            "/checkpoints/qwen38-27b",
        ]
    )

    assert args.global_batch_size == 8
    assert args.micro_batch_size == 1
    assert args.tensor_model_parallel_size == 4
    assert args.pipeline_model_parallel_size == 4
    assert args.context_parallel_size == 1
    assert args.sequence_parallel is True
    assert not any(name.startswith(("expert_", "moe_")) for name in vars(args))


def test_launcher_uses_repository_supported_distributed_entrypoint() -> None:
    """The launcher should preserve uv's environment and avoid bare torchrun."""
    launcher = (TUTORIAL_ROOT / "launch_train.sh").read_text(encoding="utf-8")

    assert "uv run python -m torch.distributed.run" in launcher
    assert "\ntorchrun " not in launcher
    assert "--validate-world-size-only" in launcher


@pytest.mark.parametrize("world_size", [16, 32, 64])
def test_default_dense_topology_accepts_divisible_world_sizes(world_size: int) -> None:
    """The default TP4/PP4/CP1 topology should require 16-GPU multiples."""
    train = _load_train_module()
    args = train.build_parser().parse_args(
        [
            "--data-root",
            "/data/cybergym",
            "--exp-dir",
            "/experiments/qwen38",
            "--epochs",
            "1",
            "--wandb-project",
            "cybergym",
            "--wandb-exp-name",
            "qwen38-smoke",
            "--hf-model-path",
            "/models/qwen38-27b",
            "--megatron-checkpoint-path",
            "/checkpoints/qwen38-27b",
        ]
    )

    train.validate_world_size(args, world_size=world_size)


@pytest.mark.parametrize("world_size", [1, 8, 15, 24])
def test_default_dense_topology_rejects_small_or_indivisible_world_sizes(
    world_size: int,
) -> None:
    """World size should fail before torchrun when it cannot host model parallelism."""
    train = _load_train_module()
    args = train.build_parser().parse_args(
        [
            "--data-root",
            "/data/cybergym",
            "--exp-dir",
            "/experiments/qwen38",
            "--epochs",
            "1",
            "--wandb-project",
            "cybergym",
            "--wandb-exp-name",
            "qwen38-smoke",
            "--hf-model-path",
            "/models/qwen38-27b",
            "--megatron-checkpoint-path",
            "/checkpoints/qwen38-27b",
        ]
    )

    with pytest.raises(ValueError, match=r"TP\*PP\*CP=16"):
        train.validate_world_size(args, world_size=world_size)


def _write_contract_evidence(pack: ModuleType, path: Path, tokenizer_root: Path) -> None:
    evidence = {
        "schema_version": 1,
        "tokenizer_path": str(tokenizer_root.resolve()),
        "tokenizer_contract_fingerprint": pack.tokenizer_contract_fingerprint(tokenizer_root),
        "supports": {
            "tools": True,
            "reasoning_content": True,
            "role_equivalent_use_loss": True,
            "packed_loss_mask_roundtrip": True,
            "explicit_use_thinking": False,
        },
    }
    path.write_text(json.dumps(evidence), encoding="utf-8")


def test_contract_evidence_sha256_rejects_changed_bytes(tmp_path: Path) -> None:
    """The workflow-provided evidence digest must bind the exact JSON bytes."""
    pack = _load_pack_module()
    evidence_path = tmp_path / "contract.json"
    evidence_path.write_bytes(b'{"schema_version":1}')
    expected = hashlib.sha256(evidence_path.read_bytes()).hexdigest()

    pack.verify_file_sha256(evidence_path, expected)
    evidence_path.write_bytes(b'{"schema_version":1,"changed":true}')

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pack.verify_file_sha256(evidence_path, expected)


def test_packer_accepts_only_evidenced_extended_chat_contract(tmp_path: Path) -> None:
    """Tools/reasoning and role-equivalent use_loss require tokenizer smoke evidence."""
    pack = _load_pack_module()
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer_config.json").write_text('{"chat_template":"fixture"}', encoding="utf-8")
    evidence_path = tmp_path / "contract.json"
    _write_contract_evidence(pack, evidence_path, tokenizer_root)
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "inspect", "use_loss": False},
                    {
                        "role": "assistant",
                        "content": "done",
                        "reasoning_content": "reason",
                        "use_loss": True,
                    },
                ],
                "tools": [{"type": "function", "function": {"name": "inspect"}}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = pack.validate_source_contract(
        source_path,
        hf_model_path=tokenizer_root,
        contract_evidence_path=evidence_path,
    )

    assert summary == {
        "records": 1,
        "tools": 1,
        "reasoning_content": 1,
        "explicit_use_loss": 2,
    }


def test_packer_fails_closed_without_extended_contract_evidence(tmp_path: Path) -> None:
    """Reasoning content must not be silently discarded by an unverified template."""
    pack = _load_pack_module()
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "reason",
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contract evidence"):
        pack.validate_source_contract(source_path, hf_model_path=tokenizer_root, contract_evidence_path=None)


@pytest.mark.parametrize(
    "record,error",
    [
        (
            {"messages": [{"role": "assistant", "content": "answer", "use_loss": False}]},
            "role-equivalent",
        ),
        (
            {
                "messages": [{"role": "assistant", "content": "answer"}],
                "use_thinking": True,
            },
            "use_thinking",
        ),
    ],
)
def test_packer_rejects_unsupported_loss_or_thinking_semantics(
    tmp_path: Path,
    record: dict[str, object],
    error: str,
) -> None:
    """The generic packer must not silently reinterpret unsupported controls."""
    pack = _load_pack_module()
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    source_path = tmp_path / "source.jsonl"
    source_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=error):
        pack.validate_source_contract(source_path, hf_model_path=tokenizer_root, contract_evidence_path=None)


def test_conversion_wrapper_performs_strict_import_export_roundtrip() -> None:
    """Training should consume a preconverted checkpoint with a strict export gate."""
    conversion = (TUTORIAL_ROOT / "convert_checkpoint.sh").read_text(encoding="utf-8")

    assert "convert_checkpoints.py import" in conversion
    assert "convert_checkpoints.py export" in conversion
    assert "--not-strict" not in conversion
