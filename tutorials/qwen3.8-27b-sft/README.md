# Qwen3.8-27B dense SFT

This tutorial is the minimal Megatron-Bridge entry point for the CyberGym
Qwen3.8-27B research track. It deliberately contains no MoE, expert-parallel,
DeepEP, or router options. Training consumes a preconverted Megatron checkpoint;
`--hf-model-path` supplies the matching tokenizer and HF configuration reference
but does not import HF weights during `train.py` startup.

The library recipe defaults to TP=4, PP=4, CP=1, micro batch size 1, and global
batch size 8. This requires at least 16 GPUs; larger world sizes must be
divisible by 16. The 4K sequence default is the first GPU smoke configuration.
Long-context topology and recomputation must be validated on the target GPU
image before increasing `--packed-sequence-size` or `--context-parallel-size`.

Current validation is limited to CPU-safe CLI contract tests, syntax checks,
and lint/format checks. Tokenizer semantics, checkpoint conversion, distributed
model construction, and training have not yet been validated. The gates below
must pass in the target GPU image before training is considered usable.

## Gate 1: tokenizer and packed loss-mask contract

The generic Megatron-Bridge packer masks by role. It does not implement
arbitrary per-message `use_loss`, and it does not forward explicit
`use_thinking` into the tokenizer. Therefore:

- explicit `use_loss` is accepted only when true for assistant messages and
  false for every other role;
- any explicit `use_thinking` is rejected;
- `tools` and `reasoning_content` require evidence from the exact immutable
  Qwen3.8 tokenizer snapshot.

Generate that evidence with the real tokenizer before packing production data:

```bash
uv run python tutorials/qwen3.8-27b-sft/verify_data_contract.py \
  --hf-model-path /models/Qwen3.8-27B \
  --work-dir /experiments/qwen38/tokenizer-contract \
  --evidence-path /experiments/qwen38/tokenizer-contract/evidence.json
```

This bounded smoke checks tool and reasoning rendering, assistant-role masking,
and the exact shifted loss mask after a Parquet pack/read roundtrip. A changed
tokenizer fingerprint invalidates the evidence. Record the exact evidence bytes
before starting parallel pack jobs:

```bash
sha256sum /experiments/qwen38/tokenizer-contract/evidence.json
```

Every pack invocation must receive that lowercase digest. The packer rejects a
missing paired argument or any evidence file whose current bytes do not match.

## Gate 2: checkpoint conversion and roundtrip export

Convert the immutable HF snapshot before training and require a strict export
roundtrip. Both destination paths must be new:

```bash
tutorials/qwen3.8-27b-sft/convert_checkpoint.sh \
  /models/Qwen3.8-27B \
  /checkpoints/Qwen3.8-27B-megatron \
  /experiments/qwen38/Qwen3.8-27B-roundtrip-hf
```

Both conversion commands must exit successfully and their logs must contain no
missing or unexpected vision, projector, language, or MTP tensors. A successful
roundtrip export is structural evidence, not numerical parity: compare a
bounded set of logits from the immutable HF checkpoint and the roundtripped HF
checkpoint before training.

## Prepare packed shards

Run one packing job per audited source shard:

```bash
uv run python tutorials/qwen3.8-27b-sft/pack.py \
  --input-path /data/audited/shard-00000.jsonl \
  --output-path /data/cybergym/packed/seq-4096/train/shard-00000.parquet \
  --metadata-path /data/cybergym/packed/seq-4096/train/shard-00000.metadata.json \
  --hf-model-path /models/Qwen3.8-27B \
  --contract-evidence /experiments/qwen38/tokenizer-contract/evidence.json \
  --contract-evidence-sha256 REPLACE_WITH_64_LOWERCASE_HEX
```

## Gate 3: launch and resume training

The launcher expects `NODE_COUNT`, `NODE_RANK`, `PROC_PER_NODE`, `MASTER_ADDR`,
and `NCCL_SOCKET_IFNAME` from the platform. `MBRIDGE_ROOT` may override the
image's `/opt/src/Megatron-Bridge` checkout.

```bash
tutorials/qwen3.8-27b-sft/launch_train.sh \
  --log-dir /experiments/qwen38/logs -- \
  --data-root /data/cybergym \
  --exp-dir /experiments/qwen38 \
  --epochs 1 \
  --wandb-project cybergym-qwen38 \
  --wandb-exp-name qwen38-27b-gbs8-smoke \
  --hf-model-path /models/Qwen3.8-27B \
  --megatron-checkpoint-path /checkpoints/Qwen3.8-27B-megatron
```

The first cluster training gate is a two-iteration smoke that loads the
preconverted Megatron checkpoint, constructs the dense provider, reads one
packed shard, completes forward/backward, saves a checkpoint, and resumes it
for one more iteration.
