export CUDA_VISIBLE_DEVICES=XXX ###
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
set -e
set -o pipefail

N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

BASE_DIR=XXX ###
LLM_DIR=XXX  ### I typically use a local directory to store all LLM checkpoints


# settings for tasks
tasks=("debate" "debate_anthropic" "debate_argsme") ### 

# settings for persuadee, make sure the list length is the same
persuadee_model_names=("Qwen2.5-3B-Instruct" "Llama-3.1-8B-Instruct" "phi-4") ###
corresponding_ports=(1279 1568 2184) ###


# settings fot persuader, make sure the list length is the same
settings=("ToMAP" "QWen-3B" "QWen-3B-RL") ### Just nicknames, but if the model is ToMAP-based, it should contain ToMAP or ToMAP-lite
model_paths=(
    "xxx"
    "xxx"
    "xxx"
) ### If external_persuader=True, you can put a small model like Qwen2.5-0.5B-Instruct as placeholder here (it won't affect the actual evaluation).
encoder_port=1450 ### For ToMAP-related settings
external_persuader=false ###

n_turns=3
val_bsz=32

for task in "${tasks[@]}"; do
    echo "==============================================="
    echo "Starting validation for task: ${task}"
    echo "==============================================="
    
    DATA_DIR=${BASE_DIR}/datasets/${task}
    if [ ! -f "$DATA_DIR/test.parquet" ]; then
        echo "Error: Data file '$DATA_DIR/test.parquet' not found. Aborting."
        exit 1
    fi

    # This part controls the proportion of validation data used for each task, aiming to save time.
    if [ "$task" = "debate" ]; then
        val_proportion=0.2
    elif [ "$task" = "debate_anthropic" ]; then
        val_proportion=1
    elif [ "$task" = "debate_argsme" ]; then
        val_proportion=0.5
    fi


    for i in "${!persuadee_model_names[@]}"; do
        persuadee_model_name=${persuadee_model_names[$i]}
        port=${corresponding_ports[$i]}

        for j in "${!settings[@]}"; do
            setting=${settings[$j]}
            tested_model_path=${model_paths[$j]}
            
            # Setting TOM related parameters
            if [[ "${setting}" == *"ToMAP-lite"* ]]; then
                tom_style=black_skip
                max_width=3
            elif [[ "${setting}" == *"ToMAP"* ]]; then
                tom_style=black_external
                max_width=3
            else
                tom_style=black_skip
                max_width=0
            fi
            
            EXPERIMENT_NAME="${task}/${setting}/against_${persuadee_model_name}"
            for trial_num in {0..2}; do # We do 3 trials with different seeds. If you wanna save time, change it to "0..0".
                if [ "$trial_num" -eq 0 ]; then
                    seed=17
                elif [ "$trial_num" -eq 1 ]; then
                    seed=23
                else
                    seed=37
                fi
                
                TRIAL_DIR="trial${trial_num}"
                OUTPUT_PATH="${BASE_DIR}/validate/${EXPERIMENT_NAME}/${TRIAL_DIR}"
                
                python3 -m verl.trainer.main_ppo \
                    algorithm.adv_estimator=gae \
                    data.train_files=$DATA_DIR/train.parquet \
                    data.val_files=$DATA_DIR/test.parquet \
                    data.train_batch_size=8 \
                    data.val_batch_size=$val_bsz \
                    data.use_chat_template=True \
                    data.max_prompt_length=5000 \
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
                    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
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
                    trainer.persuadee_model=${LLM_DIR}/${persuadee_model_name} \
                    trainer.greedy_in_val=False \
                    trainer.tom_style=${tom_style} \
                    trainer.max_width=${max_width} \
                    trainer.encoder_port=${encoder_port} \
                    trainer.external_persuader=${external_persuader} \
                    trainer.external_persuadee=False \
                    trainer.classifier_model_path=tom_model/tom_v7_lr5e-4_dims1024_512_128/best_mlp_model.pth \
                    trainer.total_epochs=15
            done
            python utils/summarize_trials.py ${BASE_DIR}/validate/${EXPERIMENT_NAME}
        done
    done
done