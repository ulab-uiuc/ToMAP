"""Reward functions for GRPO training."""

import json
import math
import re
from typing import Dict, List
import re
import random
import ast
import operator
import torch
from nltk.corpus import words
from verl.env_feedback.argument_graph import extract_answer, get_score

# Extract Solution covered in <xxx>...</xxx>
def extract_solution(original_str, key_word):
    answer_pattern = fr'<{key_word}>(.*?)</{key_word}>'
    matches = [match.group(1).strip() for match in re.finditer(answer_pattern, original_str, re.DOTALL) if match.group(1).strip() != ""]
    if matches == []:
        return None
    return matches[-1]

'''
All rewards take the output dataproto as input. Different rewards require different fields, and the other fields in the DataProto will be in **kwargs.
If you want to use hparams to control the reward, Please refer to make_repetition_penalty_reward's design.
'''


# Format reward
def format_reward(completions, pattern=r"^<thought>.*?</thought>\s*<argument>.*?</argument>$", **kwargs):
    completion_contents = completions
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


# Overlength penalty for the argument part
def overlength_penalty_reward(completions, config, tokenizer, **kwargs):
    completion_contents = completions
    completion_contents = [_.strip() for _ in completion_contents]
    max_arg_length = config.data.max_arg_length
    results = []
    
    for content in completion_contents:
        argument_text = extract_answer(content, 'argument')
        if argument_text is None:
            results.append(0.0)
            continue
        tokens = tokenizer.encode(argument_text, add_special_tokens=False)
        token_count = len(tokens)
        if token_count <= max_arg_length:
            results.append(0.0)
        else:
            overrun_ratio = (token_count - max_arg_length) / max_arg_length
            penalty = min(overrun_ratio * 2, 1.0) * -1.0  # reaches maximum penalty of -1.0 when overlength for 50% or more
            results.append(penalty)
    
    return results

# Tag Reward
def tag_count_reward(completions, tags=["thought", "argument"], **kwargs) -> list[float]:
    def count_tags(text: str) -> float:
        max_score_per_tag = 1.0 / len(tags)
        score_per_element = max_score_per_tag / 2
        
        count = 0.0
        for tag in tags:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            
            if text.count(start_tag) == 1:
                count += score_per_element
            if text.count(end_tag) == 1:
                count += score_per_element
        
        return count

    contents = completions
    return [count_tags(c) for c in contents]


# Reward for controlling only English letters. Not used in the final version.
def fluency_reward(completions, **kwargs):
    rewards = []
    allowed_chars = r"a-zA-Z0-9_\s" + re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
    pattern = rf'^[{allowed_chars}]+$'
    
    for completion in completions:
        if re.fullmatch(pattern, completion, flags=re.ASCII):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# Repetition penalty between turns.
def make_repetition_penalty_reward(config):
    ngram_size = config.trainer.rep.ngram_size
    threshold = config.trainer.rep.threshold
    max_penalty = -1
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions: List[str], turns: List[List[str]], **kwargs) -> List[float]:
        contents = completions
        rewards = []
        for completion, turn_history in zip(contents, turns):
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue
            curr_ngrams = set()
            curr_total = 0
            for ng in zipngram(completion, ngram_size):
                curr_ngrams.add(ng)
                curr_total += 1
            
            curr_scaling = 1 - len(curr_ngrams) / curr_total if curr_total > 0 else 0

            history_ngrams = set()
            history_total = 0
            for prev_text in turn_history[:-2][::2]:
                for ng in zipngram(prev_text, ngram_size):
                    history_ngrams.add(ng)
                    history_total += 1
            if history_total > 0:
                overlap_ngrams = curr_ngrams.intersection(history_ngrams)
                history_scaling = len(overlap_ngrams) / len(curr_ngrams) if curr_ngrams else 0
            else:
                history_scaling = 0

            scaling = max(curr_scaling, history_scaling)
            reward = scaling * max_penalty
            reward = min(0.0, reward + threshold)
            rewards.append(reward)
            
        return rewards

    return repetition_penalty_reward



# Persuasion Reward
from verl.env_feedback.argument_graph import print_tree
def make_debate_reward(config, raw=False):
    def calc_diff(new, old):
        if new > old:
            if old > 0.999:
                reward = 1
            else:
                reward = (new - old) / (1.0 - old)
        else:
            if old < 0.001:
                return 0
            else:
                reward = (new - old) / old
        if not config.trainer.keep_negative:
            reward = max(0.0, reward)
        return reward ** config.trainer.curve_exp

    def debate_reward(turns, trees, final_trees, **kwargs):
        rewards = []
        for all_prev_trees, final_tree, this_turns in zip(trees, final_trees, turns):
            initial_tree = all_prev_trees[0]
            last_tree = all_prev_trees[-2]
            current_tree = all_prev_trees[-1]
            
            initial_score =  get_score(initial_tree.get_node(initial_tree.root))
            last_score = get_score(last_tree.get_node(last_tree.root))
            current_score = get_score(current_tree.get_node(current_tree.root))
            final_score = get_score(final_tree.get_node(final_tree.root))
            if raw:
                assert abs(final_score - current_score) < 1e-6
                rewards.append(final_score - initial_score)
            else:
                local_reward = calc_diff(current_score, last_score)
                global_reward = calc_diff(final_score, initial_score)
                rewards.append(local_reward * (1 - config.trainer.global_factor) + global_reward * config.trainer.global_factor)
        return rewards
    
    return debate_reward


attitude_to_score = {
    "Agree": 4,
    "Partly Agree": 3,
    "Neutral": 2,
    "Partly Disagree": 1,
    "Disagree": 0
}

def tom_reward(completions, agreement, **kwargs):
    rewards = []
    for completion, inv_agreement in zip(completions, agreement):
        attitude = extract_answer(completion, "answer").strip()
        try:
            score = attitude_to_score[attitude]
            gt = (int)(inv_agreement * 4 + 0.5)
            if gt == score:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards
