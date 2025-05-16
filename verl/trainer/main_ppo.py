# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, multiply, countdown, mmlu, debate
from verl.utils.rewards import *
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import numpy as np
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM


class RewardManager():
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, reward_funcs, reward_weights, config=None, div='train') -> None:
        assert div in ['train', 'val']
        assert len(reward_funcs) == len(reward_weights)
        REWARD_FUNCS_REGISTRY = {
            "format": format_reward,
            "reasoning_steps": reasoning_steps_reward,
            "tag_count": tag_count_reward,
            "countdown": countdown_reward,
            "mcq": mcq_reward,
            "debate": make_debate_reward(config, raw = (div == 'val')),
            "fluency": fluency_reward,
            "repetition_penalty": make_repetition_penalty_reward(config),
            "overlength_penalty": overlength_penalty_reward,
            }
        self.div = div
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # if num_examine is a float then it represents the ratio
        self.reward_funcs = [{"func": REWARD_FUNCS_REGISTRY[func], "weight": weight, "name": func} for func, weight in zip(reward_funcs, reward_weights)]


    def __call__(self, data: DataProto, **kwargs):
        """We will expand this function gradually based on the available datasets"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        already_print_data_sources = {}
        prompt_sequences = []
        sequences = []
        response_lengthes = []
        try:
            for i in range(len(data)):
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch['prompts']

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                response_ids = data_item.batch['responses']
                valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]
                response_lengthes.append(valid_response_length)
                
                # decode
                prompt_sequence = valid_prompt_ids
                prompt_sequence_str = self.tokenizer.decode(prompt_sequence, skip_special_tokens=True)
                prompt_sequences.append(prompt_sequence_str)
                gen_sequence = valid_response_ids
                gen_sequence_str = self.tokenizer.decode(gen_sequence, skip_special_tokens=True)
                sequences.append(gen_sequence_str)

            data.non_tensor_batch.update({"completions": np.array(sequences, dtype=object)})
            data.non_tensor_batch.update({"response_lengthes": np.array(response_lengthes, dtype=object)})
        except:
            # This is especially designed for gpt-as-persuader scenario
            # since we only care about debate reward in this case, we can set completions and response_lengthes to any number
            data.batch["responses"] = torch.zeros_like(data.batch["attention_mask"], dtype=torch.int64)
            data.non_tensor_batch.update({"completions": np.array(["PLACEHOLDER"] * len(data), dtype=object)})
            data.non_tensor_batch.update({"response_lengthes": np.array([1] * len(data), dtype=object)})
        
        scores = [0] * len(data)
        itemized_rewards = {}
        for reward_func in self.reward_funcs:
            cur_score = reward_func["func"](**(data.non_tensor_batch), **kwargs)
            itemized_rewards[reward_func["name"]] = cur_score
            scores = [scores[i] + cur_score[i] * reward_func["weight"] for i in range(len(data))]
            
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        for i in range(len(data)):
            reward_tensor[i, data.non_tensor_batch["response_lengthes"][i] - 1] = scores[i]

        # print certain results
        if 0 < self.num_examine < 1:
            n_output = (int)(random.random() < self.num_examine)
        else:
            n_output = self.num_examine
        for i in range(n_output):
            index = random.randint(0, len(data) - 1)
            print('-'*25 + f" Example {self.div} " + '-'*25)
            try:
                print("#### The input is:\n", prompt_sequences[index])
            except:
                print("#### Cannot Find the input.")
            print("#### The completion is:\n", data.non_tensor_batch["completions"][index])
            print("#### The overall reward is:\n", scores[index])
            for key, value in itemized_rewards.items():
                print(f"#### The {key} reward is:\n", value[index])
        return reward_tensor, itemized_rewards


import ray
import hydra



class ExternalLLM:
    def __init__(self, config):
        model_name = config.persuadee_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = LLM(model_name, tokenizer=model_name, gpu_memory_utilization=0.8)
    
    def generate(self, prompts, **generate_kwargs):
        return self.model.generate(prompts, **generate_kwargs)


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if config.trainer.default_hdfs_dir == '':
        config.trainer.default_hdfs_dir = None
    if not ray.is_initialized():
        # this is for local ray cluster
        import os
        ray.init(_temp_dir=os.path.join(os.getcwd(), "tmp"), runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf

    if os.path.exists(config.trainer.default_local_dir) and not config.trainer.val_only:
        import datetime
        timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        config.trainer.default_local_dir += timestamp
    print("The local dir is: ", config.trainer.default_local_dir)
    dict_form_config = OmegaConf.to_container(config, resolve=True)
    pprint(dict_form_config)  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    
    if config.trainer.is_debate:
        assert config.data.use_chat_template, "Debate task requires chat template"

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes
    }
    # external_llm_pool = 'external_llm_pool'
    # resource_pool_spec = {
    #     global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    #     external_llm_pool: [1],
    # }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    assert config.trainer.tom_style in ['white', 'black_skip', 'black_self', 'black_external', 'black_random', 'black_max'], "Not supported"
    # if config.trainer.tom_style == 'black_skip':
    #     assert config.trainer.max_width == 0, "max_width should be 0 for black_skip"

    if config.trainer.is_debate and config.trainer.external_persuadee == False:
        from openai import OpenAI
        test_client = OpenAI(base_url=f"http://localhost:{config.trainer.port}/v1")
        assert test_client.models.list().data[0].id == config.trainer.persuadee_model, f"The model name |{test_client.models.list().data[0].id}| is not the same as the one in the config |{config.trainer.persuadee_model}|"
        del test_client
    if config.trainer.is_debate and config.trainer.tom_style == 'black_external':
        from openai import OpenAI
        test_client = OpenAI(base_url=f"http://localhost:{config.trainer.encoder_port}/v1")
        assert test_client.models.list().data[0].id == "BAAI/bge-m3", "We only support BAAI/bge-m3 as encoder"
        assert os.path.exists(config.trainer.classifier_model_path), "The classifier model path does not exist"
        del test_client
    # if config.trainer.is_debate and config.trainer.external_persuadee:
    #     assert config.trainer.persuadee_model == 'gpt-4o', "The gpt persuadee model should be gpt-4o"
        
        
    reward_fn = RewardManager(tokenizer, 1, config.trainer.reward_funcs, config.trainer.reward_weights, config=config, div='train')
    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer, 0.4, config.trainer.reward_funcs, config.trainer.reward_weights, config=config, div='val')
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    if config.trainer.init_tree_only:
        return
    os.makedirs(config.trainer.default_local_dir, exist_ok=True)
    json.dump(dict_form_config, open(os.path.join(config.trainer.default_local_dir, "config.json"), "w"), indent=4)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    main()
