export CUDA_VISIBLE_DEVICES=TODO ###
port=1450 ###

# vllm >= 0.8.4
vllm serve BAAI/bge-m3 \
  --host "127.0.0.1" \
  --port ${port} \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.1 \
  --max-num-seqs 2048 \
  --max-num-batched-tokens 100000
