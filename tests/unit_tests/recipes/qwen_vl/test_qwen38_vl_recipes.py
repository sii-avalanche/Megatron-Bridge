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

"""Unit tests for the Qwen3.8 dense SFT recipe."""

import importlib

import pytest
import torch


_qwen35_vl_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen35_vl")
_qwen38_vl_module = importlib.import_module("megatron.bridge.recipes.qwen_vl.qwen38_vl")


class _FakeModelCfg:
    def __init__(self) -> None:
        self.tensor_model_parallel_size = 1
        self.pipeline_model_parallel_size = 1
        self.pipeline_dtype = None
        self.virtual_pipeline_model_parallel_size = None
        self.context_parallel_size = 1
        self.expert_model_parallel_size = 1
        self.expert_tensor_parallel_size = 1
        self.sequence_parallel = False
        self.seq_length = 64
        self.freeze_language_model = False
        self.freeze_vision_model = False
        self.freeze_vision_projection = False

    def finalize(self) -> None:
        return None


class _FakeAutoBridge:
    hf_path: str | None = None

    @classmethod
    def from_hf_pretrained(cls, hf_path: str) -> "_FakeAutoBridge":
        cls.hf_path = hf_path
        return cls()

    def to_megatron_provider(self, load_weights: bool = False) -> _FakeModelCfg:
        assert load_weights is False
        return _FakeModelCfg()


def test_qwen38_vl_27b_sft_uses_dense_qwen35_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qwen3.8-27B should use the full VLM wrapper without MoE parallelism."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    cfg = _qwen38_vl_module.qwen38_vl_27b_sft_config()

    assert _FakeAutoBridge.hf_path == "Qwen/Qwen3.8-27B"
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.pipeline_dtype == torch.bfloat16
    assert cfg.model.expert_model_parallel_size == 1
    assert cfg.model.sequence_parallel is True
    assert cfg.model.freeze_language_model is False
    assert cfg.model.freeze_vision_model is True
    assert cfg.model.freeze_vision_projection is True
    assert cfg.model.mtp_num_layers == 1
    assert cfg.model.mtp_loss_scaling_factor == 0.1
    assert cfg.train.global_batch_size == 8
    assert cfg.train.micro_batch_size == 1
    assert cfg.peft is None


def test_qwen38_vl_27b_sft_accepts_local_checkpoint_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recipe must accept the immutable local HF checkpoint used by jobs."""
    monkeypatch.setattr(_qwen35_vl_module, "AutoBridge", _FakeAutoBridge)

    _qwen38_vl_module.qwen38_vl_27b_sft_config(hf_path="/models/qwen38-27b")

    assert _FakeAutoBridge.hf_path == "/models/qwen38-27b"
