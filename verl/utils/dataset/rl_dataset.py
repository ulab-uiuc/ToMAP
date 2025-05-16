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

from omegaconf import ListConfig
import os
from typing import List, Union
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.env_feedback.debate_prompts import *
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
import json

def tokenizer_wrapper(prompt, tokenizer, config = None, batched=False, max_len = None):
    if batched == False:
        assert isinstance(prompt, str), "Prompt should be a string"
    else:
        assert isinstance(prompt, list), "Prompt should be a list of strings"
    input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt,
                                                                        tokenizer=tokenizer,
                                                                        max_length = max_len if max_len != None else config.data.max_prompt_length,
                                                                        pad_token_id=tokenizer.pad_token_id,
                                                                        left_pad=True,
                                                                        truncation="error")

    position_ids = compute_position_id_with_mask(attention_mask)
    if batched == False:
        row_dict = {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0]
        }
    else:
        row_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
    return row_dict


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        '''
        The reason for setting an ampty np.array first is to avoid automatic type detection
        for a list of certain objects like "Tree", automatic type detection will cause error
        '''
        non_tensors[key] = np.empty(len(val), dtype=object)
        non_tensors[key][:] = val

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='question',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 use_chat_template=None,
                 return_raw_chat=False,
                 truncation='error',
                 proportion=1,
                 config=None):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]
        self.config = config
        self.parquet_files = parquet_files
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.proportion = proportion
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.use_chat_template = use_chat_template
        self.truncation = truncation

        self._download()
        self._read_files()
        

    def _download(self):
        from verl.utils.fs import copy_local_path_from_hdfs
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key

        # nvm if prompt is too long
        # print(doc[prompt_key])
        # self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
        #     tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
        #                                                      axis=1)]

        # truncate dataset based on proportion
        if self.proportion < 1:
            keep_elements = max(1, int(len(self.dataframe) * self.proportion))
            self.dataframe = self.dataframe[:keep_elements]
        
        print(f'filter dataset len: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)
        if self.use_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                                chat,
                                tokenize=False,
                                add_generation_prompt=True
                            )
        else:
            assert isinstance(chat, str), "Without chat template, the prompt should be a string"
            prompt = chat
            
        row_dict.update(tokenizer_wrapper(prompt, self.tokenizer, config=self.config))

        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict


class DebateRLHFDataset(RLHFDataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self, client, **kwargs):
        print("WARNING: In debate dataset, the data.prompt_key parameter is not working")
        self.client = OpenAI(base_url=f"http://localhost:{kwargs['config'].trainer.port}/v1")
        super().__init__(**kwargs)
        self.load_tree_and_opinion(kwargs['config'], client)

    def load_tree_and_opinion(self, config, client):
        from verl.env_feedback.argument_graph import build_tree_and_init_opinion
        self.trees = build_tree_and_init_opinion(statements=self.dataframe.to_dict(orient="records"), 
                                    client=self.client,
                                    config=config,
                                    source_path='/'.join(self.parquet_files[0].split('/')[:-1]), 
                                    split='train' if 'train' in self.parquet_files[0] else 'test')[:len(self.dataframe)]
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        In debate, we only return the raw dict. tokenization process is done in the generation loop
        """
        row_dict = self.dataframe.iloc[item].to_dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        chat = persuader_sys_prompt.replace("<root_statement>", row_dict["pos"])
        row_dict.update(tokenizer_wrapper(chat, self.tokenizer, config=self.config))
        
        row_dict['trees'] = self.trees[item]
        # row_dict['client'] = self.client
        return row_dict
