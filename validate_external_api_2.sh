# This script evaluates the model using multiple tasks.
export CUDA_VISIBLE_DEVICES=6 ###
export WANDB_ENTITY=hanpx20 ###
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
set -e
set -o pipefail

overall_base_dir=/data
OUTPUT_BASE_DIR=${overall_base_dir}/ph16/TinyZero ###
OUTPUT_DIR=${OUTPUT_BASE_DIR}/validate
N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')


tasks=("debate_argsme")
persuadee_model_names=("phi-4")
corresponding_ports=(2184)


settings=("google/gemma-3-27b-it" "deepseek-ai/deepseek-r1" "meta/llama-3.1-70b-instruct")

model_paths=(
    "${overall_base_dir}/models/Qwen2.5-0.5B-Instruct"
    "${overall_base_dir}/models/Qwen2.5-0.5B-Instruct"
    "${overall_base_dir}/models/Qwen2.5-0.5B-Instruct"
    "${overall_base_dir}/models/Qwen2.5-0.5B-Instruct"
)

n_turns=3
val_bsz=16

for task in "${tasks[@]}"; do
    echo "==============================================="
    echo "Starting validation for task: ${task}"
    echo "==============================================="
    
    if [ "$task" = "debate" ]; then
        val_proportion=0.2
    elif [ "$task" = "debate_anthropic" ]; then
        val_proportion=1
    elif [ "$task" = "debate_argsme" ]; then
        val_proportion=0.5
    fi

    # 循环遍历所有persuadee模型
    for i in "${!persuadee_model_names[@]}"; do
        persuadee_model_name=${persuadee_model_names[$i]}
        port=${corresponding_ports[$i]}
        persuadee_model_path=${overall_base_dir}/models/${persuadee_model_name}

        if [[ "${persuadee_model_name}" == *"gpt"* ]]; then
            external_persuadee=true
        else
            external_persuadee=false
        fi

        echo "Using persuadee model: ${persuadee_model_name} with port: ${port}"
        
        # 循环遍历所有实验设置
        for j in "${!settings[@]}"; do
            setting=${settings[$j]}

            # SKIP!!!!!!!!
            if [ "$task" = "debate" ] && [ "$setting" = "deepseek-ai/deepseek-r1" ]; then
                echo "Skipping debate task"
                continue
            fi
            if [ "$task" = "debate" ] && [ "$setting" = "google/gemma-3-27b-it" ]; then
                echo "Skipping debate task"
                continue
            fi
            
            export EXTERNAL_MODEL_NAME=${setting} # this will be used in the code (external_batch_inference)
            tested_model_path=${model_paths[$j]}
            
            tom_style=black_skip
            max_width=0
            
            external_persuader=true

            echo "Using setting: ${setting} with tom_style: ${tom_style}, max_width: ${max_width}"
            
            # 修改实验名称以包含当前设置和persuadee模型名称
            EXPERIMENT_NAME="${task}/${setting}/against_${persuadee_model_name}"

            DATA_DIR=${OUTPUT_BASE_DIR}/datasets/${task}
            if [ ! -f "$DATA_DIR/test.parquet" ]; then
                echo "Error: Data file '$DATA_DIR/test.parquet' not found. Aborting."
                exit 1
            fi

            for trial_num in {0..0}; do
                echo "-----------------------------------------------"
                echo "Starting trial ${trial_num} for task: ${task}, setting: ${setting}, model: ${persuadee_model_name}"
                echo "-----------------------------------------------"
                
                if [ "$trial_num" -eq 0 ]; then
                    seed=17
                elif [ "$trial_num" -eq 1 ]; then
                    seed=23
                else
                    seed=37
                fi
                
                echo "Using seed: ${seed} for trial ${trial_num}"
                
                TRIAL_DIR="trial${trial_num}"
                OUTPUT_PATH="${OUTPUT_DIR}/${EXPERIMENT_NAME}/${TRIAL_DIR}"
                
                python3 -m verl.trainer.main_ppo \
                    algorithm.adv_estimator=gae \
                    data.train_files=$DATA_DIR/train.parquet \
                    data.val_files=$DATA_DIR/test.parquet \
                    data.train_batch_size=8 \
                    data.val_batch_size=$val_bsz \
                    data.use_chat_template=True \
                    data.max_prompt_length=3000 \
                    data.max_response_length=1000 \
                    data.max_arg_length=200 \
                    data.prompt_key=pos \
                    data.train_proportion=1 \
                    data.val_proportion=$val_proportion \
                    actor_rollout_ref.model.path=$tested_model_path \
                    actor_rollout_ref.actor.optim.lr=2e-6 \
                    actor_rollout_ref.actor.optim.lr_scheduler=cosine \
                    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=8 \
                    actor_rollout_ref.actor.optim.total_training_steps=8 \
                    actor_rollout_ref.model.use_remove_padding=True \
                    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
                    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
                    actor_rollout_ref.actor.use_kl_loss=True \
                    actor_rollout_ref.actor.kl_loss_coef=0.001 \
                    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                    actor_rollout_ref.model.enable_gradient_checkpointing=True \
                    actor_rollout_ref.actor.fsdp_config.param_offload=False \
                    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
                    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
                    actor_rollout_ref.rollout.log_prob_micro_batch_size=512 \
                    actor_rollout_ref.rollout.tensor_model_parallel_size=${N_GPUS} \
                    actor_rollout_ref.rollout.gpu_memory_utilization=0.1 \
                    actor_rollout_ref.rollout.temperature=1 \
                    actor_rollout_ref.rollout.n=1 \
                    actor_rollout_ref.rollout.seed=${seed} \
                    actor_rollout_ref.ref.log_prob_micro_batch_size=512 \
                    actor_rollout_ref.ref.fsdp_config.param_offload=True \
                    algorithm.kl_ctrl.kl_coef=0.001 \
                    trainer.logger=['console'] \
                    trainer.experiment_name=${EXPERIMENT_NAME} \
                    trainer.default_hdfs_dir="" \
                    trainer.default_local_dir=${OUTPUT_PATH} \
                    trainer.n_gpus_per_node=${N_GPUS} \
                    trainer.nnodes=1 \
                    trainer.save_freq=1 \
                    trainer.test_freq=1 \
                    trainer.total_training_steps=0 \
                    trainer.val_before_train=True \
                    trainer.val_only=True \
                    trainer.reward_funcs=['debate'] \
                    trainer.reward_weights=[1] \
                    trainer.is_debate=True \
                    trainer.port=${port} \
                    trainer.max_turns=${n_turns} \
                    trainer.persuadee_model=${persuadee_model_path} \
                
                    trainer.greedy_in_val=False \
                    trainer.tom_style=${tom_style} \
                    trainer.max_width=${max_width} \
                    trainer.encoder_port=1450 \
                    trainer.external_persuader=${external_persuader} \
                    trainer.external_persuadee=${external_persuadee} \
                    trainer.classifier_model_path=tom_model/tom_v7_lr5e-4_dims1024_512_128/best_mlp_model.pth \
                    trainer.total_epochs=15 2>&1 | tee validate_external.log
            done
            python utils/summarize_trials.py ${OUTPUT_DIR}/${EXPERIMENT_NAME}
        done
    done
done