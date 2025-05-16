# Adapted from: https://github.com/PeterGriffinJin/Search-R1

import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
# from search_r1.utils import set_seed
# from search_r1.utils.plot import (
#     save_trajectory_to_output,
#     parse_llm_output
# )
from verl import DataProto
from verl.utils.tracking import Tracking
from verl.env_feedback.argument_graph import format_turns, format_extra_info, extract_answer, judge_opinions, print_tree
from verl.llm_agent.batch_inference import external_batch_inference
from verl.env_feedback.debate_prompts import *
from verl.utils.dataset.rl_dataset import collate_fn, tokenizer_wrapper
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import copy


def _postprocess_responses(tokenizer, responses: Union[torch.Tensor, List[str]], config, get_lengths=False) -> torch.Tensor:
    if isinstance(responses, torch.Tensor):
        responses_str = tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )
    elif isinstance(responses, list):
        responses_str = responses
    else:
        raise ValueError(f"Invalid type for responses: {type(responses)}")

    thoughts_str = [extract_answer(resp, "thought", strict=True) for resp in responses_str]
    thoughts_str = [str.replace('\n', ' ').strip() for str in thoughts_str]
    responses_str = [extract_answer(resp, "argument", strict=True) for resp in responses_str]
    responses_str = [str.replace('\n', ' ').strip() for str in responses_str]
    
    thought_tokens = tokenizer(
        thoughts_str, 
        add_special_tokens=False, 
        return_tensors='pt', 
        padding="max_length", 
        max_length=config.data.max_response_length,
        truncation=True
    )
    responses_tokens = tokenizer(
            responses_str, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="max_length", 
            max_length=config.data.max_arg_length,
            truncation=True
        )
    responses_str = tokenizer.batch_decode(
        responses_tokens['input_ids'], 
        skip_special_tokens=True
    ) # this step is to truncate over length "arguments"
    if get_lengths:
        return responses_tokens, responses_str, thoughts_str, torch.sum(thought_tokens['attention_mask'], dim=1), torch.sum(responses_tokens['attention_mask'], dim=1)
    else:
        return responses_tokens, responses_str, thoughts_str


def multi_round_debate(data_batch, tokenizer, rollout, n_turns, config, client=None, tom_classifier=None):
    '''Set the intermediate variables'''
    assert config.actor_rollout_ref.rollout.n == 1, "GRPO (n>1) Might have some errors in the current implementation"
    is_train = not data_batch.meta_info.get("validate", False)
    tmp = config.actor_rollout_ref.rollout.temperature if is_train else 0

    all_gen_batches = None
    thought_lengths = torch.zeros(len(data_batch.non_tensor_batch['pos']), n_turns)
    argument_lengths = torch.zeros(len(data_batch.non_tensor_batch['pos']), n_turns)
    all_turns = np.full((len(data_batch.non_tensor_batch['pos']), n_turns * 2), None, dtype=object) 
    all_thoughts = np.full((len(data_batch.non_tensor_batch['pos']), n_turns * 2), None, dtype=object) 
    all_intermediate_opinions = np.full((len(data_batch.non_tensor_batch['pos']), n_turns + 1), None, dtype=object)
    all_intermediate_opinions[:,  0] = data_batch.non_tensor_batch.pop('trees')
    cur_tree_list = copy.deepcopy(all_intermediate_opinions[:,  0])  # cur_Tree might be different from the real tree if tom_style != white
    
    if config.trainer.tom_style == "black_external":
        '''requires an encoder'''
        from openai import OpenAI
        encoder_client = OpenAI(base_url=f"http://localhost:{config.trainer.encoder_port}/v1")
        
    '''loop over the turns'''
    for n_turn in range(n_turns):
        real_persuader_sys_prompts = [persuader_sys_prompt.replace("<root_statement>", pos_argument).replace("<extra_info>", format_extra_info(opinions, config, first = (n_turn == 0))) for opinions, pos_argument in zip(cur_tree_list, data_batch.non_tensor_batch['pos'])]
        real_persuadee_sys_prompts = [persuadee_sys_prompt.replace("<root_statement>", pos_argument) for pos_argument in data_batch.non_tensor_batch['pos']]
        
        '''prepare inputs for the persuader'''
        real_debater_prompts = [persuader_turn_prompt.replace("<turns>", format_turns(turns, "persuader", pos, neg)).replace("<extra_info>", format_extra_info(opinions, config, first = (n_turn == 0))) for turns, opinions, pos, neg in zip(all_turns, cur_tree_list, data_batch.non_tensor_batch['pos'], data_batch.non_tensor_batch['neg'])]
        real_prompts = [tokenizer.apply_chat_template([{"role": "system","content": sys_,},{"role": "user","content": usr_}],
                            tokenize=False,
                            add_generation_prompt=True
                        ) for sys_, usr_ in zip(real_persuader_sys_prompts, real_debater_prompts)]
        tokenized_input = collate_fn([tokenizer_wrapper(prompt, tokenizer, config=config) for prompt in real_prompts])
        data_batch.batch = data_batch.batch.update(tokenized_input)
        

        '''persuader inference'''
        if config.trainer.external_persuader == True:
            '''persuader is an external API'''
            persuader_chat_prompts = [[{"role": "system","content": sys_,},{"role": "user","content": usr_}] 
                                    for sys_, usr_ in zip(real_persuader_sys_prompts, real_debater_prompts)]
            gen_responses_persuader = external_batch_inference(
                client=client, # Will be ignored by batch_inference
                requests=persuader_chat_prompts,
                sampling_params={"temperature": tmp, "max_tokens": config.data.max_response_length},
                text_only=True,
                external_api=True,
            )
            responses_ids, responses_str, thoughts_str, t_length, a_length = _postprocess_responses(
                tokenizer, gen_responses_persuader, config, get_lengths=True
            )
            gen_batch = data_batch
        else:
            '''persuader is a locally loaded model'''
            tokenized_input = collate_fn([tokenizer_wrapper(prompt, tokenizer, config=config) for prompt in real_prompts])
            data_batch.batch = data_batch.batch.update(tokenized_input)
            gen_batch = rollout.generate_sequences(data_batch)
            responses_ids, responses_str, thoughts_str, t_length, a_length = _postprocess_responses(
                tokenizer, gen_batch.batch['responses'], config, get_lengths=True
            )

        all_turns[:, 2 * n_turn] = responses_str
        all_thoughts[:, 2 * n_turn] = thoughts_str
        thought_lengths[:, n_turn] = t_length
        argument_lengths[:, n_turn] = a_length
        
        '''prepare inputs for the persuadee'''
        real_debater_prompts_persuadee = [persuadee_turn_prompt.replace("<turns>", format_turns(turns, "persuadee", pos, neg)) for turns, pos, neg in zip(all_turns, data_batch.non_tensor_batch['pos'], data_batch.non_tensor_batch['neg'])]
        persuadee_chat_prompts = [[{"role": "system","content": sys_,},{"role": "user","content": usr_}] for sys_, usr_ in zip(real_persuadee_sys_prompts, real_debater_prompts_persuadee)]

        '''persuadee inference'''
        gen_responses_persuadee = external_batch_inference(
            client=client,
            requests=persuadee_chat_prompts, # Use chat format prompts
            sampling_params={"temperature": tmp, "max_tokens": config.data.max_response_length},
            text_only=True,
            external_api=config.trainer.external_persuadee,
        )
        responses_ids_persuadee, responses_str_persuadee, thoughts_str_persuadee = _postprocess_responses(
            tokenizer, gen_responses_persuadee, config, get_lengths=False # Keep get_lengths=False as per original
        )
        
        all_turns[:, 2 * n_turn + 1] = responses_str_persuadee
        all_thoughts[:, 2 * n_turn + 1] = thoughts_str_persuadee

        '''calc ground truth intermediate opinions'''
        root_only = config.trainer.tom_style != 'white' or (is_train and n_turn == n_turns - 1) # usually we only need the root node
        gt_tree_list = judge_opinions(tree_list=all_intermediate_opinions[:,  0].tolist(),
                                all_turns=all_turns.tolist(),
                                client=client,
                                max_width = 0 if root_only else config.trainer.max_width,
                                external_api=config.trainer.external_persuadee)
        
        '''calc predicted intermediate opinions'''
        if n_turn != n_turns - 1:
            if config.trainer.tom_style == "white" or config.trainer.tom_style == "black_skip":
                cur_tree_list = gt_tree_list # no need to predict since we already have the ground truth
            elif config.trainer.tom_style == "black_external":
                cur_tree_list = judge_opinions(tree_list=all_intermediate_opinions[:,  0].tolist(),
                                    all_turns=all_turns.tolist(),
                                    client=encoder_client,
                                    max_width = config.trainer.max_width,
                                    tokenizer = tokenizer,
                                    tom_classifier = tom_classifier,
                                    source="external")
            elif config.trainer.tom_style == "black_max":
                cur_tree_list = copy.deepcopy(all_intermediate_opinions[:,  0].tolist())
                for tree in cur_tree_list:
                    for node in tree.all_nodes():
                        node.data["persuadee_confidence"] = 1.0
                        node.data["persuader_confidence"] = 0.0
            elif config.trainer.tom_style == "black_random":
                confidence_options = [0.0, 0.25, 0.5, 0.75, 1.0]
                cur_tree_list = copy.deepcopy(all_intermediate_opinions[:,  0].tolist())
                for tree in cur_tree_list:
                    for node in tree.all_nodes():
                        node.data["persuadee_confidence"] = np.random.choice(confidence_options)
                        node.data["persuader_confidence"] = np.random.choice(confidence_options)
            else:
                raise ValueError(f"Unknown TOM style: {config.trainer.tom_style}")

        '''Update intermediate opinions'''
        all_intermediate_opinions[:, n_turn + 1] = gt_tree_list # since [0] means the original attitude
        gen_batch.non_tensor_batch = data_batch.non_tensor_batch
        gen_batch.non_tensor_batch["turns"] = all_turns[:, :2 * n_turn + 2] # ALL turns till this round
        gen_batch.non_tensor_batch["thoughts"] = all_thoughts[:, :2 * n_turn + 2] # ALL thoughts till this round
        gen_batch.non_tensor_batch["trees"] = all_intermediate_opinions[:, :n_turn + 2] # ALL trees till this round
        gen_batch.batch["thought_lengths"] = thought_lengths[:, n_turn]
        gen_batch.batch["argument_lengths"] = argument_lengths[:, n_turn]
        if is_train:
            if n_turn == 0:
                all_gen_batches = copy.deepcopy(gen_batch)
            else:
                all_gen_batches = DataProto.concat([all_gen_batches, gen_batch])
        elif n_turn == n_turns - 1:
            all_gen_batches = copy.deepcopy(gen_batch)

    '''After all turns, sync the final trees'''
    if is_train:
        all_gen_batches.non_tensor_batch["final_trees"] = np.full(len(data_batch.non_tensor_batch['pos']) * n_turns, None, dtype=object)
        all_gen_batches.non_tensor_batch["final_trees"][:] = gt_tree_list * n_turns
    else:
        all_gen_batches.non_tensor_batch["final_trees"] = np.full(len(data_batch.non_tensor_batch['pos']), None, dtype=object)
        all_gen_batches.non_tensor_batch["final_trees"][:] = gt_tree_list
    
    return all_gen_batches