#!/usr/bin/env bash
set -euo pipefail

model_name="${LECTURE_TEX_LLM_MODEL:-Qwen/Qwen3.8-27B-FP8}"
listen_host="${LECTURE_TEX_LLM_HOST:-127.0.0.1}"
listen_port="${LECTURE_TEX_LLM_PORT:-8000}"
gpu_fraction="${LECTURE_TEX_LLM_GPU_FRACTION:-0.65}"
max_model_len="${LECTURE_TEX_LLM_MAX_MODEL_LEN:-32768}"
max_sequences="${LECTURE_TEX_LLM_MAX_SEQUENCES:-4}"

exec python -m vllm.entrypoints.openai.api_server \
  --model "$model_name" \
  --host "$listen_host" \
  --port "$listen_port" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$gpu_fraction" \
  --max-model-len "$max_model_len" \
  --max-num-seqs "$max_sequences" \
  --tensor-parallel-size 1 \
  --limit-mm-per-prompt '{"image": 5}' \
  --default-chat-template-kwargs '{"enable_thinking": false}'
