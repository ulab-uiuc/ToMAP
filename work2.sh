export CUDA_VISIBLE_DEVICES=1,7,8,9 ###
export WANDB_ENTITY=hanpx20 ###
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
set -e

# File related hparams
overall_base_dir=/data

model_name=Qwen2.5-3B-Instruct ###
persuadee_model_name=Qwen2.5-7B-Instruct ###
BASE_MODEL=${overall_base_dir}/models/${model_name} ###
task=debate ###
OUTPUT_BASE_DIR=${overall_base_dir}/ph16/TinyZero ###
EXPERIMENT_NAME=${task}/${model_name}-v10-ToMAP_lite ###

# Training related hparams
sampling_bsz=128
gradient_bsz=64
forward_bsz=32
val_bsz=64
total_steps=200
save_interval=50
n_turns=3
warmup=0.2
max_width=3

DATA_DIR=${OUTPUT_BASE_DIR}/datasets/${task}
OUTPUT_DIR=${OUTPUT_BASE_DIR}/checkpoints
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
# Cost Estimation
# 3B inf bsz=32 -> ~32GB
# 3B vllm -> ~16GB
# 3B bsz=128,32,16 -> 12GB * 4


# python data_gen/debate.py --local_dir /data/ph16/TinyZero/datasets/debate

# keep_negative: True
# curve_exp: 1
# global_factor: 0.25
# white_box: False
# turns_warmup_steps_ratio


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$sampling_bsz \
    data.val_batch_size=$val_bsz \
    data.use_chat_template=True \
    data.max_prompt_length=3000 \
    data.max_response_length=1000 \
    data.max_arg_length=200 \
    data.prompt_key=pos \
    data.train_proportion=1 \
    data.val_proportion=0.2 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler=constant \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$warmup \
    actor_rollout_ref.actor.optim.total_training_steps=${total_steps} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$gradient_bsz \
    actor_rollout_ref.actor.ppo_micro_batch_size=$forward_bsz \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${sampling_bsz} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${sampling_bsz} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    critic.optim.lr=2e-6 \
    critic.optim.lr_scheduler=constant \
    critic.optim.lr_warmup_steps_ratio=$warmup \
    critic.optim.total_training_steps=${total_steps} \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_mini_batch_size=$gradient_bsz \
    critic.ppo_micro_batch_size=$forward_bsz \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_hdfs_dir="" \
    trainer.default_local_dir=${OUTPUT_DIR}/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_interval} \
    trainer.test_freq=${save_interval} \
    trainer.total_training_steps=${total_steps} \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.reward_funcs=['debate','tag_count','format','repetition_penalty','overlength_penalty'] \
    trainer.reward_weights=[1,0.1,0.1,0.1,0.1] \
    trainer.is_debate=True \
    trainer.port=1279 \
    trainer.max_turns=${n_turns} \
    trainer.persuadee_model=${overall_base_dir}/models/${persuadee_model_name} \

    trainer.global_factor=0 \
    trainer.turns_warmup_steps_ratio=0 \
    trainer.save_training_result=True \
    trainer.tom_style=black_skip \
    trainer.max_width=${max_width} \
    trainer.greedy_in_val=False \
    trainer.encoder_port=1450 \
    trainer.classifier_model_path=tom_model/tom_v7_lr5e-4_dims1024_512_128/best_mlp_model.pth \
    trainer.total_epochs=15 2>&1 | tee logs/debate.log
