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

"""Qwen3.8-VL recipes built on the compatible Qwen3.5 architecture bridge."""

from __future__ import annotations

from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.qwen_vl.qwen35_vl import _qwen35_vl_apply_common
from megatron.bridge.training.config import ConfigContainer


def qwen38_vl_27b_sft_config(hf_path: str = "Qwen/Qwen3.8-27B") -> ConfigContainer:
    """Return the dense Qwen3.8-27B full-SFT config used by CyberGym.

    Qwen3.8-27B publishes ``Qwen3_5ForConditionalGeneration`` and therefore
    uses the existing full Qwen3.5 VLM bridge.  CyberGym is text-only, so the
    visual tower and projection remain in the checkpoint but are frozen.

    The topology defaults to the already-supported 27B layout (TP=4, PP=4).
    The experiment-specific tutorial may add context parallelism for long
    packed sequences.  GBS=8 is fixed here to match this research track.

    Args:
        hf_path: Hugging Face model ID or immutable local model directory.
    """
    cfg = _sft_common_vlm()
    _qwen35_vl_apply_common(
        cfg,
        hf_path,
        tp=4,
        pp=4,
        max_lr=5e-6,
        min_lr=5e-7,
        gbs=8,
    )
    cfg.train.micro_batch_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True
    return cfg
