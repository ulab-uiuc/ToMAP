from openai import OpenAI
from verl.env_feedback.debate_prompts import gpt_debate_prompt
from verl.env_feedback.argument_graph import extract_answer
from tqdm import tqdm
import json
import pandas as pd
import argparse

def gen_counter_example_with_gpt(source_statements):
    results = []
    client = OpenAI()
    for statement in tqdm(source_statements):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": gpt_debate_prompt},
                    {"role": "user", "content": statement}
                ],
                temperature=0.5
            )
            ans = response.choices[0].message.content.strip()
            result = {
                'pos': ans.split('\n')[0].strip(),
                'neg': ans.split('\n')[1].strip()
            }
            results.append(result)
            
        except Exception as e:
            continue
            
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--test_ratio', type=float, default=1.0, help='Ratio of test set (0-1)')
    
    args = parser.parse_args()
    
    original_statements = json.load(open(args.base_dir + "/statements.json", "r"))
    results = gen_counter_example_with_gpt(original_statements)

    total_size = len(results)
    test_size = int(total_size * args.test_ratio)
    
    test_results = results[:test_size]
    train_results = results[test_size:]
    
    with open(args.base_dir + "/test.jsonl", "w") as f:
        json.dump(test_results, f, indent=4, ensure_ascii=False)
    
    if len(train_results) > 0:
        with open(args.base_dir + "/train.jsonl", "w") as f:
            json.dump(train_results, f, indent=4, ensure_ascii=False)