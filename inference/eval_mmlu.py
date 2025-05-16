import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import SamplingParams, LLM
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import pandas as pd
from verl.utils.reward_score import mmlu


def load_answer_file(path):
    if 'jsonl' in path:
        with open(path, "r") as f:
            return [json.loads(l) for l in f]
    elif 'parquet' in path:
        df = pd.read_parquet(path)
        return [{"text": q["prompt"][0]["content"], "subject": q["subject"], "answer": q["extra_info"]["answer"]} for q in df.to_dict(orient='records')]
    
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="model_answers", help="The output answer directory.")

    
    args = parser.parse_args()
    answer_file = os.path.join(args.output_dir, args.run_name, "answers.jsonl")
    print(f"Reading from to {answer_file}")


    cor = 0
    cor_form = 0
    answers = load_answer_file(answer_file)
    for line in answers:
            for ans in line["model_answer"]:
                if mmlu.compute_score(ans, line["answer"]) == 1.0:
                    cor += 1
                    break
            for ans in line["model_answer"]:
                if mmlu.compute_score(ans, line["answer"]) > 0:
                    cor_form += 1
                    break
    print(f"Correct answers: {cor}/{len(answers)}={cor/len(answers):.3f}")
    print(f"Correct form answers: {cor_form}/{len(answers)}={cor_form/len(answers):.3f}")
    with open(os.path.join(args.output_dir, args.run_name, "acc.txt"), "w") as f:
        f.write(f"Correct answers: {cor}/{len(answers)}={cor/len(answers):.3f}\n")
        f.write(f"Correct form answers: {cor_form}/{len(answers)}={cor_form/len(answers):.3f}\n")