import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import SamplingParams, LLM
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import pandas as pd

HELPFUL_PROMPT = '''You are a helpful assistant.'''

def load_question_file(path):
    if 'jsonl' in path:
        with open(path, "r") as f:
            return [json.loads(l) for l in f]
    elif 'parquet' in path:
        df = pd.read_parquet(path)
        return [{"text": q["prompt"][0]["content"], "subject": q["subject"], "answer": q["extra_info"]["answer"]} for q in df.to_dict(orient='records')]
    
def format_prompt(tokenizer, prompt, sys_prompt = "", raw=False, shots=[]):
    if (not raw) and tokenizer.chat_template:
        if sys_prompt:
            messages = [{"role": "system", "content": sys_prompt}]
        else:
            messages = []
        for shot in shots:
            messages.append({"role": "user", "content": shot[0]})
            messages.append({"role": "assistant", "content": shot[1]})
        messages.append({"role": "user", "content": prompt})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        if sys_prompt != "":
            return sys_prompt + '\n\n' + prompt + '\n'
        else:
            return prompt + '\n'
    

def run_eval(
    model_path,
    prompts,
    answer_file,
    max_new_token,
    num_gpus_per_model,
    openai_api=False,
    no_cot=False
):
    print(model_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        tokenizer = None
    dialogs = []
    answers = []
    
    for idx, question in tqdm(enumerate(prompts)):
        qs = question["text"]
        if no_cot:
            qs = qs.replace("Please express your thought step by step. ", "")
        if openai_api:
            prompt = qs
        else:
            prompt = format_prompt(tokenizer, qs, sys_prompt=HELPFUL_PROMPT)
        dialogs.append(prompt)
        if "answer" in question:
            answers.append(question["answer"])


    if openai_api:
        vllm_outputs = []
        for dialog in tqdm(dialogs):
            model_ans = call_GPT(dialog, sys_prompt=HELPFUL_PROMPT, model=model_path)
            vllm_outputs.append([model_ans]) 
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
            ans_json = {
                "id": idx,
                "prompt": prompt["text"],
                "answer": answers[idx] if answers != [] else None,
                "model_answer": model_answers,
                "subject": prompt["subject"] if "subject" in prompt else None
            }
            fout.write(json.dumps(ans_json, ensure_ascii=True) + "\n")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="The path to the weights. This can be a local folder or a Hugging Face repo ID.")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="model_answers", help="The output answer directory.")
    parser.add_argument("--max-new-token", type=int, default=2048, help="The maximum number of new generated tokens.")
    parser.add_argument("--num-gpus-per-model", type=int, default=1, help="The number of GPUs per model.")
    parser.add_argument("--limit", type=float, default=1.0, help="The portion of data used.")
    parser.add_argument("--no_cot", type=int, default=0, choices=[0, 1])
    
    args = parser.parse_args()
    prompts = load_question_file(args.input_file)
    os.makedirs(os.path.join(args.output_dir, args.run_name), exist_ok=True)
    answer_file = os.path.join(args.output_dir, args.run_name, "answers.jsonl")
    print(f"Output to {answer_file}")

    assert 0 < args.limit <= 1
    prompts = prompts[:int(len(prompts) * args.limit)]

    run_eval(
        model_path=args.model_path,
        prompts=prompts,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_gpus_per_model=args.num_gpus_per_model,
        openai_api= "gpt" in args.model_path,
        no_cot = args.no_cot == 1
    )