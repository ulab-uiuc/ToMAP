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
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets
import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.env_feedback.argument_graph import PersuadeeOpinionGraph
from verl.env_feedback.debate_prompts import *
import argparse
from transformers import AutoTokenizer
from openai import OpenAI

def parquet_to_json(parquet_path, json_path, orient='records', lines=True, encoding='utf-8'):
    df = pd.read_parquet(parquet_path)
    json_str = df.to_json(orient=orient, lines=lines, force_ascii=False)
    with open(json_path, 'w', encoding=encoding) as f:
        f.write(json_str)


if __name__ == '__main__':
    '''
    pos is the statement for the persuader
    neg is the statement for the opponent (which has a HIGHER agreement rate by the judge initially)
    The debate graph represents the opinion of the judge, so its root statement is 'neg'
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    args = parser.parse_args()

    data_source = "debate"

    dataset = datasets.load_dataset('json', data_files = {"train": args.base_dir + "/train.json", "test": args.base_dir + "/test.json"})
    
    def make_map_fn(split):

        def process_fn(example, idx):
            data = {
                "data_source": data_source,
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = dataset['train'].map(function=make_map_fn('train'), with_indices=True, num_proc=24)
    test_dataset = dataset['test'].map(function=make_map_fn('test'), with_indices=True, num_proc=24)
    print(f"The size of training set: {len(train_dataset)}")
    print(f"The size of testing set: {len(test_dataset)}")
    
    base_dir = args.base_dir
    train_dataset.to_parquet(os.path.join(base_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(base_dir, 'test.parquet'))
