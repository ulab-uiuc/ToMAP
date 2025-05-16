#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=8
export WANDB_ENTITY=hanpx20 ###
set -e
# 创建输出目录
OUTPUT_DIR="./tom_model_output"
mkdir -p $OUTPUT_DIR

# 定义训练参数
EMBEDDING_MODEL="BAAI/bge-m3"
TRAIN_FILE="/data/ph16/TinyZero/datasets/tom_v7/train.jsonl"
TEST_FILE="/data/ph16/TinyZero/datasets/tom_v7/test.jsonl"
HIDDEN_DIMS="512 256"
BATCH_SIZE=64
EPOCHS=15
proportion=0.25
MAX_LENGTH=8192

warmup=0.1

LEARNING_RATES=("5e-4")
HIDDEN_DIMS_ARRAY=("1024 512 128")

for lr in "${LEARNING_RATES[@]}"; do
    for dims in "${HIDDEN_DIMS_ARRAY[@]}"; do
        echo "====== Training with LR=${lr}, HIDDEN_DIMS=${dims} ======"

        python train.py \
            --mode train \
            --train_file $TRAIN_FILE \
            --test_file $TEST_FILE \
            --embedding_model $EMBEDDING_MODEL \
            --hidden_dims $dims \
            --batch_size $BATCH_SIZE \
            --learning_rate $lr \
            --epochs $EPOCHS \
            --max_length $MAX_LENGTH \
            --output_dir $OUTPUT_DIR \
            --proportion $proportion \
            --run_name "tom_v7_lr${lr}_dims${dims// /_}" \
            --warmup_ratio $warmup \
            --weight_decay 0.1 \
            --dropout 0.1 \
            --wandb
    done
done


python utils/train_agreement_predictor.py \
    --mode evaluate \
    --test_file $TEST_FILE \
    --model_path tom_model/tom_v7_lr5e-4_dims1024_512_128/best_mlp_model.pth \
    --output_dir $OUTPUT_DIR \
    --run_name tom_v7_lr5e-4_dims1024_512_128 \
    --hidden_dims 1024 512 128