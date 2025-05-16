# Test on LLM
import argparse
import os
import json
import re
from tqdm import tqdm
import ast

from openai import OpenAI

sys_msg = '''You are a helpful assistsnt with expertise in debate. Given a statement, please judge if it's a suitable title for debate. If not, rewrite the statement while keeping the main idea. If the statement is too brief, you may paraphrase it to be clearer, but keep it in one or two sentences.
A good debate topic should be:
1. Controversial and Debatable: It should spark genuine disagreement with well-supported arguments on both sides.
2. Clear and Specific: The topic must be narrowly defined so that debaters know exactly what’s being argued. Do not use metaphor or analogy unless you must do so.
3. Objective: The topic should not contain subjective descriptions like "I think", "I agree".
4. The topic should be a descriptive sentence (not a question).
5. The topic should not be too academic or abstract. It should be easily understood by the general public.
6. The topic shouldn't contain specific names.
Put your final statement in <statement> </statement> tags.
After that, write another statement for the OPPONENT in the debate, put it in <opponent> </opponent> tags.
'''

def extract_answer(original_str, key_word):
    answer_pattern = fr'<{key_word}>(.*?)</{key_word}>'
    matches = [match.group(1).strip() for match in re.finditer(answer_pattern, original_str, re.DOTALL) if match.group(1).strip() != ""]
    return matches[-1] if matches else None


def call_GPT(user_msg, system_msg="You are a helpful assistant.", model="gpt-4o"):
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    test_data = json.load(open("/home/ph16/TinyZero/cmv/all/train_period_data.json", "r"))
    output_data = []
    for data in tqdm(test_data):
        data = data.replace("CMV: ", "")
        response = call_GPT(data, sys_msg)
        formatted_data = {"pos": extract_answer(response, "statement"), "neg": extract_answer(response, "opponent")}
        if all(formatted_data.values()):
            output_data.append(formatted_data)
    json.dump(output_data, open("/home/ph16/TinyZero/cmv/final/train.json", "w"), indent=4)
        
    