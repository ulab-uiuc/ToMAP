export CUDA_VISIBLE_DEVICES=0
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
set -e

python data_gen/tom.py --input_dir /data/ph16/TinyZero/checkpoints/debate/Qwen2.5-3B-Instruct-v7-graph3-white_20250419_151605/intermediate_result.pkl --output_dir /data/ph16/TinyZero/datasets/tom_v7

# # File related hparams
# overall_base_dir=/data

# model_name=Qwen2.5-3B-Instruct ###
# BASE_MODEL=${overall_base_dir}/models/${model_name} ###

# python utils/eval_tom.py \
#     --model-path $BASE_MODEL \
#     --input-file /data/ph16/TinyZero/datasets/tom_v6/test.jsonl \
#     --port 1568 \
#     --run-name test