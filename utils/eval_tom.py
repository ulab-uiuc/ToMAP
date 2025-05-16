import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import SamplingParams, LLM
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from verl.env_feedback.debate_prompts import *
from verl.env_feedback.argument_graph import  extract_answer
from verl.utils.rewards import tom_reward
from verl.llm_agent.batch_inference import external_batch_inference
from openai import OpenAI
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score

HELPFUL_PROMPT = guesser_sys_prompt
attitude_to_score = {
    "Agree": 4,
    "Partly Agree": 3,
    "Neutral": 2,
    "Partly Disagree": 1,
    "Disagree": 0
}


def load_jsonl(file_path):
    """
    Load a JSONL file and return a list of dictionaries.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def eval(answer_file):
    """
    使用tom_reward函数评估模型的回答
    """
    answers = load_jsonl(answer_file)
    
    # 提取模型答案和真实协议度
    completions = []
    agreements = []
    model_actual_answers = []
    for ans in answers:
        completions.append(ans["model_answer"])
        agreements.append(ans["agreement"])
        model_actual_answers.append(ans.get("model_actual_answer", extract_answer(ans["model_answer"], "answer")))
    
    # 使用tom_reward函数计算奖励
    rewards = tom_reward(completions, agreements)
    
    # 将文本标签转换为数值以便计算 MSE
    true_vals = []
    pred_vals = []
    for idx, (agr, ans) in enumerate(zip(agreements, model_actual_answers)):
        true_val = (int)(agr * 4 + 0.1)
        pred_val = attitude_to_score.get(ans, 2)  # 默认为 Neutral (2)
        true_vals.append(true_val)
        pred_vals.append(pred_val)
    
    # 计算指标
    total = len(answers)
    correct = sum(1 for r in rewards if r == 1.0)  # 只有完全匹配才算正确
    accuracy = correct / total if total > 0 else 0
    mse_score = mean_squared_error(true_vals, pred_vals)
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_vals, pred_vals, labels=[0, 1, 2, 3, 4])
    cm_list = cm.tolist()
    
    # 计算分类报告
    label_names = ["Disagree", "Partly Disagree", "Neutral", "Partly Agree", "Agree"]
    report = classification_report(true_vals, pred_vals, labels=[0, 1, 2, 3, 4], 
                                  target_names=label_names, output_dict=True)
    
    # 计算真实标签和预测标签的分布
    true_dist = {i: true_vals.count(i) for i in range(5)}
    pred_dist = {i: pred_vals.count(i) for i in range(5)}
    
    # 准备详细结果
    results = []
    for idx, ans in enumerate(answers):
        result = {
            "id": ans["id"],
            "statement": ans["statement"],
            "ground_truth": ans["agreement"],
            "model_answer": ans["model_answer"],
            "model_actual_answer": model_actual_answers[idx],
            "true_val": true_vals[idx],
            "pred_val": pred_vals[idx],
            "model_score": rewards[idx],
            "is_match": (rewards[idx] == 1.0)
        }
        results.append(result)
    
    # 准备指标汇总
    metrics = {
        "test_accuracy": float(accuracy),
        "test_mse": float(mse_score),
        "confusion_matrix": cm_list,
        "classification_report": report,
        "true_label_distribution": true_dist,
        "predicted_label_distribution": pred_dist,
        "total_samples": total,
        "correct_samples": correct
    }
    
    # 输出总结到终端
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"MSE: {mse_score:.4f}")
    
    # 保存结果到文件
    result_file = answer_file.replace(".jsonl", "_eval_results.jsonl")
    with open(result_file, "w") as f:
        f.write(json.dumps({
            "metrics": metrics,
            "results": results
        }, ensure_ascii=True, indent=4) + "\n")


def run_eval(
    model_path,
    prompts,
    answer_file,
    max_new_token,
    num_gpus_per_model,
    port
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = None
    dialogs = []
    answers = []
    openai_api = (port == 1)
    
    for idx, question in tqdm(enumerate(prompts)):
        if openai_api:
            dialogs.append(question['prompt'])
        else:
            prompt = tokenizer.apply_chat_template(question['prompt'], tokenize=False, add_generation_prompt=True)
            dialogs.append(prompt)


    if openai_api:
        os.environ["EXTERNAL_MODEL_NAME"] = model_path
        vllm_outputs = external_batch_inference(OpenAI(), dialogs, {"temperature": 0.7, "max_tokens": max_new_token}, external_api=True,  progress=True)
    else:
        model = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype='bfloat16',
            tensor_parallel_size=num_gpus_per_model,
            disable_custom_all_reduce=True,
            enforce_eager=True,
            gpu_memory_utilization=0.65,
            max_model_len=2048
        )
        sampling_params = SamplingParams(temperature=0.7, max_tokens=max_new_token, n=1)
        vllm_outputs = model.generate(dialogs, sampling_params)
        vllm_outputs = [[attempt.text.strip() for attempt in decoded_answer.outputs] for decoded_answer in vllm_outputs]
    
    
    # Dump answers
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(os.path.expanduser(answer_file), "w") as fout:
        for idx, model_answers in enumerate(vllm_outputs):
            prompt = prompts[idx]
            model_answer = model_answers[0] if isinstance(model_answers, list) and len(model_answers) > 0 else model_answers
            # 确保model_answer是字符串
            if not isinstance(model_answer, str):
                model_answer = str(model_answer)
            
            ans_json = {
                "id": idx,
                "statement": prompt["statement"],
                "agreement": prompt["agreement"],
                "model_answer": model_answer,
                "model_actual_answer": extract_answer(model_answer, "answer"),
                "turns": prompt["turns"],
                "side": prompt["side"],
            }
            fout.write(json.dumps(ans_json, ensure_ascii=True) + "\n")
    
    # 评估所有答案
    eval(answer_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="The path to the weights. This can be a local folder or a Hugging Face repo ID.")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default="model_answers", help="The output answer directory.")
    parser.add_argument("--max-new-token", type=int, default=512, help="The maximum number of new generated tokens.")
    parser.add_argument("--num-gpus-per-model", type=int, default=1, help="The number of GPUs per model.")
    parser.add_argument("--limit", type=float, default=1.0, help="The portion of data used.")
    parser.add_argument("--port", type=int, default=0)
    
    args = parser.parse_args()
    args.run_name = args.run_name.split("/")[-1]
    prompts = load_jsonl(args.input_file)
    answer_file = os.path.join(args.output_dir, args.model_path.split("/")[-1], f"{args.run_name}.jsonl")
    print(f"Output to {answer_file}")
    assert 0 < args.limit <= 1
    prompts = prompts[:int(len(prompts) * args.limit)]

    run_eval(
        model_path=args.model_path,
        prompts=prompts,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_gpus_per_model=args.num_gpus_per_model,
        port = args.port
    )