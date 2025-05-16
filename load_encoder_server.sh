export CUDA_VISIBLE_DEVICES=4
# vllm >= 0.8.4
vllm serve BAAI/bge-m3 \
  --host "127.0.0.1" \
  --port 1450 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.1 \
  --max-num-seqs 2048 \
  --max-num-batched-tokens 100000

# lm_eval --model local-chat-completions \
#   --model_args model=${LLM_DIR}/Qwen2.5-7B-Instruct,base_url=http://127.0.0.1:8000/v1 \
#   --tasks lambada_openai

# lm_eval --model vllm \
#     --model_args pretrained=${LLM_DIR}/Qwen2.5-7B-Instruct,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8,base_url=http://127.0.0.1:1279/v1 \
#     --tasks lambada_openai \
#     --batch_size auto