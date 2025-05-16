import argparse
import os
import csv
import json
from statistics import mean
from datasets import load_dataset
from verl.utils.rewards import *
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer


REWARD_FUNCS_REGISTRY = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "reasoning_steps": reasoning_steps_reward,
    "length": len_reward,
    "code": code_reward,
    "tag_count": tag_count_reward,
    "countdown": countdown_reward
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--reward_funcs', nargs='+', type=str, required=True)
    parser.add_argument('--reward_weights', nargs='+', type=float, default=[])
    parser.add_argument('--system_prompt', type=str, default="You are a helpful assistant")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_format', type=str, default="json")
    parser.add_argument("--data_proportion", type=float, default=1.0)
    parser.add_argument('--n_gpu', type=int, default=1)
    args = parser.parse_args()

    if args.reward_weights == []:
        args.reward_weights = [1.0] * len(args.reward_funcs)
    if len(args.reward_funcs) != len(args.reward_weights):
        raise ValueError("reward_funcs 和 reward_weights must have the same length")

    # Load the model
    model = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        dtype='bfloat16',
        tensor_parallel_size=args.n_gpu,
        disable_custom_all_reduce=True,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        max_model_len=1024
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    sampling_params = SamplingParams(temperature=0, max_tokens=1024, n=1)


    # Load the dataset
    assert os.path.exists(os.path.join(args.dataset_path, f"test.{args.dataset_format}")), f"Test dataset file not found at {args.dataset_path}"
    data_file = os.path.join(args.dataset_path, f"test.{args.dataset_format}")
    dataset = load_dataset(args.dataset_format, data_files={"test": data_file})["test"]

    if args.data_proportion < 1.0:
        total_samples = len(dataset)
        truncated_samples = int(total_samples * args.data_proportion)
        dataset = dataset.select(range(truncated_samples))

    
    # initialize the files
    reward_scores_all = {name: [] for name in args.reward_funcs}
    weighted_scores_all = []

    os.makedirs(args.output_dir, exist_ok=True)
    json_file = os.path.join(args.output_dir, "results.jsonl")
    avg_file = os.path.join(args.output_dir, "average_scores.txt")

    conversations = dataset #####
    batch_prompts = [tokenizer.apply_chat_template(conv["prompt"], tokenize=False, add_generation_prompt=True) for conv in conversations]
    completions = model.generate(batch_prompts, sampling_params)
    
    all_convs = []
    rows = []
    for example, conv, completion in zip(dataset, conversations, completions):
        completion_text = completion.outputs[0].text.strip()
        conv["completions"] = [{"content": completion_text}]
        row = {"question": example["question"], "completion": completion_text, "weighted_reward": 0.0}
        rows.append(row)
        all_convs.append(conv)

    batch_kwargs = {}
    if all_convs:
        keys = all_convs[0].keys()
        for key in keys:
            batch_kwargs[key] = [conv[key] for conv in all_convs]

    for name, weight in zip(args.reward_funcs, args.reward_weights):
        reward_func = REWARD_FUNCS_REGISTRY[name]
        scores = reward_func(**batch_kwargs)
        for row, score in zip(rows, scores):
            row[name] = score
            row["weighted_reward"] += score * weight
            reward_scores_all[name].append(score)
            weighted_scores_all.append(row["weighted_reward"])
            
    with open(json_file, "w", encoding="utf-8") as f_json:
        for row in rows:
            f_json.write(json.dumps(row) + "\n")


    with open(avg_file, "w", encoding="utf-8") as f_txt:
        f_txt.write("Overall statistics:\n")
        print("Overall statistics:\n")
        for name in args.reward_funcs:
            avg_score = mean(reward_scores_all[name]) if reward_scores_all[name] else 0.0
            f_txt.write(f"{name}: {avg_score:.4f}\n")
            print(f"{name}: {avg_score:.4f}")
        avg_weighted = mean(weighted_scores_all) if weighted_scores_all else 0.0
        f_txt.write(f"\nFinal weighted score: {avg_weighted:.4f}\n")
        print(f"Final weighted score: {avg_weighted:.4f}")

    print(f"Results saved to {args.output_dir}")




if __name__ == "__main__":
    main()
