import itertools
import math
import dataclasses
from dataclasses import dataclass
from typing import List, Union, Tuple
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import os
import json
from treelib import Tree
from openai import OpenAI
import requests
from verl.env_feedback.debate_prompts import *
from verl.llm_agent.batch_inference import external_batch_inference
from verl.utils.dataset.rl_dataset import collate_fn, tokenizer_wrapper
from verl.protocol import DataProto
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import torch




def extract_answer(original_str, key_word, n_answers=1, strict=True):
    answer_pattern = fr'<{key_word}>(.*?)</{key_word}>'
    matches = [match.group(1).strip() for match in re.finditer(answer_pattern, original_str, re.DOTALL) if match.group(1).strip() != ""]
    if matches == []:
        answer_pattern = fr'<{key_word}>(.*?)$'
        matches = [match.group(1).strip() for match in re.finditer(answer_pattern, original_str, re.DOTALL) if match.group(1).strip() != ""]
    
    if matches == []:
        if strict:
            list_to_return =  [""]
        else:
            list_to_return = [original_str]
    else:
        list_to_return = matches[-n_answers:]

    if n_answers == 1:
        return list_to_return[0]
    else:
        return list_to_return
    

def format_turns(turns, role, pos=None, neg=None):
    assert role in ["persuader", "persuadee", "guesser"]
    turns = [turn for turn in turns if turn != None]
    if role != "guesser":
        turns = ["Hi, I am Alice. How are you today? ", f"Nice to meet you. Let's begin the discussion."] + turns
    turns = [turn if turn != "" else "I have no argument to make in this round." for turn in turns]
    if turns == []:
        return "# There are no previous turns in the conversation. Do NOT make up any previous turns."
    
    single = 1 if len(turns) == 1 else 0
    persuader_side = 'Alice' if role != "persuader" else 'Alice (You)'
    persuadee_side = 'Bob' if role != "persuadee" else 'Bob (You)'
    return "# Following are the previous turns of the conversation.\n" + '\n'.join([f'{persuader_side if idx % 2 == single else persuadee_side} (turn {idx + 1}): "{turn}"' for idx, turn in enumerate(turns)])



def get_score(node):
    assert node.data["persuadee_confidence"] >= 0 and node.data["persuader_confidence"] >= 0
    conf_A = node.data["persuadee_confidence"]
    conf_B = node.data["persuader_confidence"]
    return 0.5 + (conf_B - conf_A) / 2


def format_extra_info(opinions, config, first=False):
    if config.trainer.max_width == 0:
        return ""
    info = "# Here are some claims your OPPONENT might hold (so DO NOT accept these claims!). You may refute them when you need to, but make your each argument single-focused and concise:\n"
    for idx, node in enumerate(opinions.all_nodes()[1:config.trainer.max_width+1]): # remove the root node
        if config.trainer.tom_style == "white":
            score = 8 - (int)(get_score(node) * 8 + 0.5)
            info += f'''"{node.data['persuadee_claim']}" (Bob's agreement on this claim is {score}/8)\n'''
        elif not first and config.trainer.tom_style != "black_skip":
            score = 8 - (int)(get_score(node) * 8 + 0.5)
            info += f'''"{node.data['persuadee_claim']}" (Bob's agreement on this claim is {score}/8)\n'''
        else:
            info += f'''"{node.data['persuadee_claim']}"\n'''
    info = info[:-1]
    return info



def format_prompt(prompt, sys_prompt="", raw=False, shots=[]):
    if sys_prompt:
        messages = [{"role": "system", "content": sys_prompt}]
    else:
        messages = []
    for shot in shots:
        messages.append({"role": "user", "content": shot[0]})
        messages.append({"role": "assistant", "content": shot[1]})
    messages.append({"role": "user", "content": prompt})
    return messages


def build_tree_and_init_opinion(statements: List[dict], client, config, source_path, split):
    # this function hasn't been tested !
    max_depth=config.trainer.max_depth
    max_width=config.trainer.max_width
    persuadee_model=config.trainer.persuadee_model
    model_name = persuadee_model.split('/')[-1]
    external_api = config.trainer.external_persuadee
    
    # Part 1: opinion tree (ALREADY PREPROCESSED, if you wanna do from scratch, remember to set the "persuadee" to be the actual persuader)
    if os.path.exists(os.path.join(source_path, f"{split}_argument_tree.pkl")):
        print("########## Opinion: Loading from cache")
        tree_list = pickle.load(open(os.path.join(source_path, f"{split}_argument_tree.pkl"), "rb"))
    else:
        print("########## Opinion: Generating trees")
        def get_tree(idx):
            pos = statements[idx]["pos"]
            neg = statements[idx]["neg"]
            processor = PersuadeeOpinionGraph(
                                            persuadee_claim=neg, 
                                            persuader_claim=pos, 
                                            client=client,
                                            max_depth=max_depth, 
                                            max_width=max_width)
            return processor.tree
        with ThreadPoolExecutor(max_workers=min(len(statements), 256)) as executor:
            tree_list = list(tqdm(
                executor.map(get_tree, range(len(statements))),
                total=len(statements),
                desc="Processing statements"
            ))

        os.makedirs(source_path, exist_ok=True)
        with open(os.path.join(source_path, f"{split}_argument_tree.pkl"), "wb") as f:
            pickle.dump(tree_list, f)
        
    # Part 2: model initial opinions (model-specific)
    if os.path.exists(os.path.join(source_path, model_name, f"{split}.pkl")):
        print("########## Opinion: Loading from cache")
        tree_list = pickle.load(open(os.path.join(source_path, model_name, f"{split}.pkl"), "rb"))
    else:
        print("########## Opinion: Generating trees")
        tree_list = judge_opinions(tree_list, [[] for _ in range(len(statements))], client, max_width=max_width, external_api=False)
        sum_scores = 0
        Stat = {"raw_results": []}
        for tree in tree_list:
            root = tree.get_node(tree.root)
            Stat["raw_results"].append({'pos': root.data["persuader_claim"], 'neg': root.data["persuadee_claim"], 'debate': get_score(root), 'turns': []})
            sum_scores += get_score(root)
        Stat["score"] = {"val/itemized_rewards/debate" : sum_scores / len(tree_list)}
        os.makedirs(os.path.join(source_path, model_name), exist_ok=True)
        pickle.dump(tree_list, open(os.path.join(source_path, model_name, f"{split}.pkl"), "wb"))
        json.dump(Stat, open(os.path.join(source_path, model_name, f"{split}_initial_attitude.json"), "w"), indent=4)

    return tree_list



def get_hidden_states(tokenizer, model, input_texts):
    '''
    Get the internal states
    return: [bsz, hidden_dim]
    '''
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        encoding = tokenizer(input_texts, return_tensors="pt", padding=True, return_attention_mask=True, padding_side="right")
        attention_mask = encoding.attention_mask.to(device)
        valid_lengths = (attention_mask.sum(dim=1).long() - 1).cpu() # the last position of valid token
        input_ids = encoding.input_ids.to(device)
        
        states = model(input_ids, output_hidden_states=True).hidden_states[-1]
        extracted_outputs = []
        for i in range(states.shape[0]):
            extracted_outputs.append(states[i, valid_lengths[i], :])
    return torch.stack(extracted_outputs)



def calc_argument_vectors(model_path, tree_list_path, result_path, model=None):
    if model == None:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    trees = pickle.load(open(tree_list_path, "rb"))[:20] # debug
    max_len = max([tree.size() for tree in trees])
    states = torch.zeros((len(trees), max_len, model.config.hidden_size))
    for idx, tree in tqdm(enumerate(trees), total = len(trees), desc='Processing Argument Vectors'):
        arguments = []
        for node in tree.all_nodes():
            arguments.append(node.data["persuadee_claim"])
        states[idx, :len(arguments), :] = get_hidden_states(tokenizer, model, arguments)
    torch.save(states, result_path)



def predict_graph_nodes(tree, vectors):
    pass


def judge_opinions(tree_list: List, all_turns: List[List[str]], client, debug=False, max_width=1000, tokenizer = None, tom_classifier = None, source = 'gt', external_api=False):
    tree_list = copy.deepcopy(tree_list)

    def format_prompt(sys_prompt, prompt):
        return [{"content": sys_prompt, "role": "system"},
                {"content": prompt, "role": "user"}]

    if source == 'gt':
        confidence_prompts = []
        for tree, turns in zip(tree_list, all_turns):
            cur_sys_prompt = persuadee_sys_prompt.replace("<root_statement>", tree.get_node(tree.root).data["persuader_claim"])
            turn_prompt = format_turns(turns, "persuadee", pos=tree.get_node(tree.root).data["persuader_claim"], neg=tree.get_node(tree.root).data["persuadee_claim"])
            for node in tree.all_nodes()[:max_width+1]:
                confidence_prompts.append(format_prompt(cur_sys_prompt, persuadee_confidence_prompt.replace("<turns>", turn_prompt).replace("<statement>", node.data["persuadee_claim"]).replace("<statement2>", node.data["persuader_claim"])))
                confidence_prompts.append(format_prompt(cur_sys_prompt, persuadee_confidence_prompt.replace("<turns>", turn_prompt).replace("<statement>", node.data["persuader_claim"]).replace("<statement2>", node.data["persuadee_claim"])))
        # calculate and post-process results
        
        if debug:
            for prompt in confidence_prompts:
                print(prompt)
                print('-'*50)
        
        results = external_batch_inference(client=client,
                                            requests=confidence_prompts,
                                            sampling_params={"temperature": 0, "max_tokens": 512},
                                            text_only=True, 
                                            progress=len(tree_list) > 1000,
                                            external_api=external_api)
    elif source == "external":
        assert tom_classifier != None, "tom_classifier should not be None"
        formated_turns = [format_turns(turns, "guesser") for turns in all_turns]
        tmp_turn_vectors = client.embeddings.create(model=client.models.list().data[0].id, input=formated_turns)
        tmp_turn_vectors = torch.tensor([x.embedding for x in tmp_turn_vectors.data]) # B
        
        statements = []
        turn_vectors = []
        for idx, tree in enumerate(tree_list):
            for node in tree.all_nodes()[:max_width+1]:
                statements.append(node.data["persuadee_claim"])
                statements.append(node.data["persuader_claim"])
                turn_vectors.append(tmp_turn_vectors[idx])
                turn_vectors.append(tmp_turn_vectors[idx])
                
        stat_vectors = client.embeddings.create(model=client.models.list().data[0].id, input=statements)
        stat_vectors = torch.tensor([x.embedding for x in stat_vectors.data])
        turn_vectors = torch.stack(turn_vectors)
        combined_embeddings = torch.cat([stat_vectors, turn_vectors], dim=1)

        classifier_output = tom_classifier(combined_embeddings)
        results = torch.argmax(classifier_output, dim=1)
    else:
        raise ValueError(f"Unknown source: {source}")
        
        
    def get_score(result):
        if debug:
            print(result)
        if source == 'gt':
            # Avoid prefix problem
            if "Partly Agree" in result:
                return attitude_weights[1]  # 3
            elif "Partly Disagree" in result:
                return attitude_weights[3]  # 1
            elif "Agree" in result:
                return attitude_weights[0]  # 4
            elif "Disagree" in result:
                return attitude_weights[4]  # 0
            elif "Neutral" in result:
                return attitude_weights[2]  # 2
            else:
                return 0
        elif source == "external":
            x = (int)(result)
            assert x in attitude_weights
            return x
        else:
            raise ValueError(f"Unknown source: {source}")
        
        
    confidences = [get_score(result) for result in results]
    confidences = confidences[::-1]
    new_tree_list = []
    
    for idx, (tree, turns) in enumerate(zip(tree_list, all_turns)):
        for node in tree.all_nodes()[:max_width+1]:
            persuadee_claim_agreement = confidences.pop()
            persuader_claim_agreement = confidences.pop()
            if debug:
                print(persuadee_claim_agreement, persuader_claim_agreement)
            node.data["persuadee_confidence"] = persuadee_claim_agreement / 4
            node.data["persuader_confidence"] = persuader_claim_agreement / 4
        for node in tree.all_nodes()[max_width+1:]:
            node.data["persuadee_confidence"] = -1
            node.data["persuader_confidence"] = -1
        new_tree_list.append(tree)
        
    return new_tree_list



def print_tree(tree):
    for node in tree.all_nodes():
        print('-'*20)
        print("Persuadee Claim: " + node.data["persuadee_claim"])
        # print(f"Counter-claim: {node.data['persuader_claim']}")
        print(f"Persuadee side Confidence: {node.data['persuadee_confidence']:.4f}")
        print(f"Persuader Claim: {node.data['persuader_claim']}")
        print(f"Persuader side Confidence: {node.data['persuader_confidence']:.4f}")
        # if node.data["depth"] != 0:
        #     print(f"Coherence: {node.data['coherence']:.4f}")
        print('-'*20)

def serialize_tree(tree):
    data = []
    for node in tree.all_nodes():
        data.append({'claim': node.data["persuadee_claim"], "persuadee_confidence": node.data["persuadee_confidence"]})
    return data

class PersuadeeOpinionGraph:
    def __init__(self, persuadee_claim, persuader_claim, client, max_depth = 0, max_width = 1):
        self.max_depth = max_depth
        self.max_width = max_width
        self.edge_types = ["abductive"]
        self.client = client
        self.persuadee_claim = persuadee_claim
        self.persuader_claim = self.prompt_tilde(persuadee_claim) if persuader_claim == None else persuader_claim
        self.turns = []
        self.tree = self.create_maieutic_graph()
        # self.initial_tree = copy.deepcopy(self.tree)
    
    def create_maieutic_graph(self):
        # Only creates the graph structure, doesn't calculate opinion
        G = Tree()
        G.create_node(self.persuadee_claim, "Root", data={
            "persuadee_claim": self.persuadee_claim,
            "persuader_claim": self.persuader_claim,
            "type": "statement",
            "depth": 0
        }) # Root node
        
        for depth in range(1, self.max_depth + 1):
            parents_to_generate_from = list(G.leaves())
            for parent_node in parents_to_generate_from:
                if parent_node.data["type"] == "statement":
                    # abductive reasoning (find statement)
                    if "abductive" in self.edge_types:
                        new_statement_list = extract_answer(self.get_statement_abductive(parent_node.data["persuadee_claim"]), "reason", n_answers=self.max_width)
                        new_statement_tilde_list = [self.prompt_tilde(statement) for statement in new_statement_list]
                        for idx, (statement, statement_tilde) in enumerate(zip(new_statement_list, new_statement_tilde_list)):
                            node_identifier = f"{parent_node.identifier}-abd-{idx}"
                            G.create_node(statement, node_identifier, parent=parent_node.identifier, data={
                                "persuadee_claim": statement,
                                "persuader_claim": statement_tilde,
                                "type": "statement",
                                "depth": depth
                            })
        return G
    
    def get_statement_abductive(self, Q, n=1) -> List[str]:
        sys_prompt = persuadee_sys_prompt
        user_prompt = judge_abductive_prompt.replace("<statement>", Q).replace("<width>", str(self.max_width))
        prompt_str = format_prompt(user_prompt, sys_prompt=sys_prompt)
        response = external_batch_inference(self.client, [prompt_str], generation_config)[0]
        return response

    def prompt_tilde(self, E: str):
        # prompt_str = format_prompt(E, sys_prompt=negation_prompt_json["sys_prompt"], shots=negation_prompt_json["shots"])
        prompt_str = format_prompt(E, sys_prompt=negation_prompt)
        response = external_batch_inference(self.client, [prompt_str], negation_config)[0]
        return extract_answer(response, "answer")