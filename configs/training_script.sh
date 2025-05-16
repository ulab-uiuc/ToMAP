export CUDA_VISIBLE_DEVICES=0,5
export N_GPUS=2
export BASE_MODEL=/shared/nas2/shared/llms/Qwen2.5-3B # 
export DATA_DIR=/shared/nas2/ph16/TinyZero/mmlu_data #
export OUTPUT_DIR=/shared/nas2/ph16/TinyZero/checkpoints #
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=mmlu-Qwen2.5-3B-grpo # 
export VLLM_ATTENTION_BACKEND=XFORMERS

# statistics for mmlu
# small
# The size of training set: 3565
# The size of testing set: 394
# all
# The size of training set: 99842
# The size of testing set: 1531


# # # Generate data for mmlu
python ./examples/data_preprocess/debate.py --local_dir /shared/nas2/ph16/TinyZero/debate_data

# # # Train the model, need to set environment variables
# bash ./scripts/train_tiny_zero_grpo.sh

# # Evaluate on mmlu
# python inference/inference.py \
#     --model-path ${BASE_MODEL} \
#     --input-file /shared/nas2/ph16/TinyZero/mmlu_data_qwen/test.parquet \
#     --run-name ${EXPERIMENT_NAME} \
#     --output-dir model_answers \
#     --limit 1

# python inference/eval_mmlu.py \
#     --run-name ${EXPERIMENT_NAME} \
#     --output-dir model_answers