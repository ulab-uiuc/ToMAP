export CUDA_VISIBLE_DEVICES=TODO ###
export LLM_DIR=TODO ###
port=1279 ###

vllm serve ${LLM_DIR}/Qwen2.5-7B-Instruct \
  --host "127.0.0.1" \
  --port ${port} \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --max-num-seqs 2048 \
  --max-num-batched-tokens 100000