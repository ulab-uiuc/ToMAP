export CUDA_VISIBLE_DEVICES=6 ###
export WANDB_ENTITY=hanpx20 ###
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
set -e

# File related hparams
overall_base_dir=/data

model_name=Qwen2.5-3B-Instruct ###

BASE_MODEL=${overall_base_dir}/models/${model_name} ###
OUTPUT_DIR=${overall_base_dir}/ph16/TinyZero/checkpoints/debate/${model_name}-sft-biglr ###


python analysis/sft_train.py \
    --model_id ${BASE_MODEL} \
    --output_dir $OUTPUT_DIR \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --logging_steps 5 \
    --save_strategy epoch \
    --warmup_ratio 0.1 \
    --test