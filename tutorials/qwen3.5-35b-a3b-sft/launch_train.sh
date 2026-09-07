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

if [[ $# -lt 4 || $1 != "--log-dir" || $3 != "--" ]]; then
    echo "usage: $0 --log-dir LOG_DIR -- TRAIN_ARGS..." >&2
    exit 2
fi

log_dir=$2
shift 3
mkdir -p "$log_dir"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
infra_repo_root=/opt/src/Megatron-Bridge

export NNODES="${NODE_COUNT:?NODE_COUNT is required}"
export WORLD_SIZE="$NNODES"
export GLOO_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME is required}"
export RANK="${NODE_RANK:?NODE_RANK is required}"
export MASTER_PORT=23467
export NPROC_PER_NODE="${PROC_PER_NODE:?PROC_PER_NODE is required}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$infra_repo_root"
source .venv/bin/activate

torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$script_dir/train.py" "$@" >> "$log_dir/RANK_${NODE_RANK}.log" 2>&1
