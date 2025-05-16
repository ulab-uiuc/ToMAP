export CUDA_VISIBLE_DEVICES=1,6
export LLM_DIR=/shared/nas2/shared/llms
export DATA_DIR=/shared/nas2/ph16/TinyZero/checkpoints/finetune
export WANDB_ENTITY=hanpx20

set -e

# Llama-3.1-8B-Instruct Qwen2.5-7B
model=Qwen2.5-1.5B-Instruct

python finetune.py \
    --model_id ${LLM_DIR}/${model} \
    --output_dir ${DATA_DIR}/${model} \
    --train_file /shared/nas2/ph16/TinyZero/mmlu_data/train.jsonl \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --num_train_epochs 15 \
    --batch_size 16 \
    --inst

# model=Qwen2.5-7B

# python src/finetune.py \
#     --model_id ${LLM_DIR}/${model} \
#     --output_dir ${DATA_DIR}/checkpoints/${model}_forward \
#     --train_file data/forward_train.jsonl \
#     --save_strategy epoch \
#     --learning_rate 1e-5 \
#     --num_train_epochs 15


# python src/finetune.py \
#     --model_id ${LLM_DIR}/${model} \
#     --output_dir ${DATA_DIR}/checkpoints/${model}_backward \
#     --train_file data/backward_train.jsonl \
#     --save_strategy epoch

