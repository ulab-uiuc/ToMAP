export CUDA_VISIBLE_DEVICES=XXX ###
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
set -e
set -o pipefail


persuadee_model_names=("Qwen2.5-7B-Instruct") ###
corresponding_ports=(1279) ###
tasks=("debate") ###
OUTPUT_BASE_DIR=${overall_base_dir}/ph16/TinyZero ###
val_only=False ### set to True if there's only test set

sampling_bsz=0
gradient_bsz=0
forward_bsz=0
val_bsz=32
total_steps=0
save_interval=0
n_turns=3
warmup=0

N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')


for i in "${!persuadee_model_names[@]}"; do
    persuadee_model_name=${persuadee_model_names[$i]}
    port=${corresponding_ports[$i]}
    
    echo "Processing model: $persuadee_model_name with port: $port"
    
    BASE_MODEL=openai-community/gpt2 # This is not important for build_tree

    for task in "${tasks[@]}"; do
        echo "Processing task: $task with model: $persuadee_model_name"
        
        # set variables
        # most of them won't be used in build_tree, only the data path is crucial
        EXPERIMENT_NAME=${task}/build_tree/${persuadee_model_name}
        DATA_DIR=${OUTPUT_BASE_DIR}/datasets/${task}
        OUTPUT_DIR=${OUTPUT_BASE_DIR}/checkpoints



        if [ "$task" = "debate" ]; then
            val_proportion=0.2
        elif [ "$task" = "debate_anthropic" ]; then
            val_proportion=1
        elif [ "$task" = "debate_argsme" ]; then
            val_proportion=0.5
        fi

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
            data.val_proportion=${val_proportion} \
            actor_rollout_ref.model.path=$BASE_MODEL \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.optim.lr_scheduler=cosine \
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
            actor_rollout_ref.rollout.log_prob_micro_batch_size=$forward_bsz \
            actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS} \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
            actor_rollout_ref.rollout.n=1 \
            actor_rollout_ref.ref.log_prob_micro_batch_size=$forward_bsz \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            algorithm.kl_ctrl.kl_coef=0.001 \
            critic.optim.lr=2e-6 \
            critic.optim.lr_scheduler=cosine \
            critic.optim.lr_warmup_steps_ratio=$warmup \
            critic.optim.total_training_steps=${total_steps} \
            critic.model.path=$BASE_MODEL \
            critic.model.enable_gradient_checkpointing=True \
            critic.ppo_mini_batch_size=$gradient_bsz \
            critic.ppo_micro_batch_size=$forward_bsz \
            trainer.critic_warmup=0 \
            trainer.logger=['console'] \
            trainer.experiment_name=${EXPERIMENT_NAME} \
            trainer.default_hdfs_dir="" \
            trainer.default_local_dir=${OUTPUT_DIR}/${EXPERIMENT_NAME} \
            trainer.n_gpus_per_node=${N_GPUS} \
            trainer.nnodes=1 \
            trainer.save_freq=${save_interval} \
            trainer.test_freq=${save_interval} \
            trainer.total_training_steps=${total_steps} \
            trainer.init_tree_only=True \
            trainer.val_before_train=True \
            trainer.val_only=${val_only} \
            trainer.reward_funcs=['debate'] \
            trainer.reward_weights=[1] \
            trainer.is_debate=True \
            trainer.port=$port \
            trainer.max_turns=${n_turns} \
            trainer.persuadee_model=${overall_base_dir}/models/${persuadee_model_name} \
            trainer.external_persuadee=True \
            trainer.max_width=3 \
            trainer.tom_style=white \
            trainer.total_epochs=15
    done
done

