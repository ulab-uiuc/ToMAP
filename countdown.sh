export CUDA_VISIBLE_DEVICES=5,7,8,9 ###
export WANDB_ENTITY=hanpx20 ###
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
set -e

model_name=Qwen2.5-3B-Instruct ###
BASE_MODEL=/data/models/${model_name} ###
task=countdown ###
OUTPUT_BASE_DIR=/data/ph16/TinyZero ###
EXPERIMENT_NAME=${task}/Qwen2.5-3B-Instruct-new ###


DATA_DIR=${OUTPUT_BASE_DIR}/datasets/${task}
OUTPUT_DIR=${OUTPUT_BASE_DIR}/checkpoints
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
ROLLOUT_TP_SIZE=$N_GPUS


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=16 \
    data.use_chat_template=True \
    data.prompt_key=prompt \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=256 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_hdfs_dir="" \
    trainer.default_local_dir=${OUTPUT_DIR}/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_training_steps=10 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.reward_funcs=['countdown','tag_count','format'] \
    trainer.reward_weights=[1,0.05,0.05] \
    trainer.total_epochs=15 2>&1 | tee verl_demo.log