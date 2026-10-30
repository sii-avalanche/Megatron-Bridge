#!/usr/bin/env bash
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

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 HF_MODEL_PATH MEGATRON_CHECKPOINT_PATH ROUNDTRIP_HF_PATH" >&2
    exit 2
fi

hf_model_path=$1
megatron_checkpoint_path=$2
roundtrip_hf_path=$3

if [[ ! -d $hf_model_path ]]; then
    echo "HF model directory does not exist: $hf_model_path" >&2
    exit 2
fi
if [[ -e $megatron_checkpoint_path || -e $roundtrip_hf_path ]]; then
    echo "conversion destinations must not already exist" >&2
    exit 2
fi

uv run python examples/conversion/convert_checkpoints.py import \
    --hf-model "$hf_model_path" \
    --megatron-path "$megatron_checkpoint_path" \
    --torch-dtype bfloat16

uv run python examples/conversion/convert_checkpoints.py export \
    --hf-model "$hf_model_path" \
    --megatron-path "$megatron_checkpoint_path" \
    --hf-path "$roundtrip_hf_path"

test -f "$roundtrip_hf_path/config.json"
