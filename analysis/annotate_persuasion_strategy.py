from openai import OpenAI
import json
import pickle
import os
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


line_colors = {}
def style():
    from matplotlib import font_manager
    global line_colors
    
    line_colors['red'] = "#e3716e"
    
    line_colors["light_grey"] = "#afb0b2"
    line_colors["grey"] = "#656565"
    
    line_colors["green"] = "#c0db82"
    line_colors["yellow_green"] = "#54beaa"
    
    line_colors["pink"] = "#efc0d2"
    
    line_colors["light_purple"] = "#eee5f8"
    line_colors["purple"] = "#af8fd0"
    
    line_colors["blue"] = "#6d8bc3"
    line_colors["cyan"] = "#2983b1"
    
    line_colors["yellow"] = "#f9d580"
    line_colors["orange"] = "#eca680"
    
    line_colors["gradual_yellow"] = "#EABE5D"
    line_colors["gradual_purple"] = "#3F314F"


    font_path = '/data/ph16/fonts/cambria.ttc'  # 你的路径
    font_prop = font_manager.FontProperties(fname=font_path)
    font_manager.fontManager.addfont(font_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()

    
    plt.rcParams['font.size'] = 24
    plt.rcParams['lines.linewidth'] = 1.5
    
    plt.rcParams['axes.titlesize'] = 1
    plt.rcParams['axes.labelsize'] = 1
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 18


strategy_classifier_sys_prompt = '''You are a debate expert. You're analysing a debate between Alice and Bob, and you will be shown Alice's thought process and speech. Your task is to analyze and identify the persuasion strategy employed in the conversation.
We provide a detailed taxonomy for you. Please choose one or more from the following nine strategies:

Evidential Appeals : The speaker backs the claim with facts, data, statistics or logical reasoning. Examples include citing research findings, logical explanations, or objective proof to support the point.
Authority Appeals: The persuader emphasizes expertise, trustworthiness or moral authority. Includes tactics like invoking an expert or rules. 
Emotional Appeals: The speaker appeals to feelings or values to sway the listener. Techniques include fear or threat appeals, empathy or personal stories, humor, pride or guilt.
Social Appeals: The speaker invokes social proof, consensus or norms. This includes bandwagon-style arguments (everyone is doing it), references to group norms, or fear of ostracism.
Common Ground Appeals: Before addressing differences, emphasize areas of agreement on basic premises or values to make the other party more receptive.
Gradual Concession: First guide the other party to accept a mild or ambiguous point, then gradually lead toward a more controversial claim.
Framing Effects: Influence the interpretation of facts by presenting them differently (positive/negative, as a problem/opportunity).
Rhetoric: These involve linguistic tricks or figure of speech, like metaphors, analogies, rhetorical questions, hyperbole, repetition, or patterned wording.
Preemptive Rebuttal: Anticipate and address potential counterarguments while presenting your point, weakening the other party's ability to easily object later.

Your answer should contain two parts: thought and answer. Follow the answer format strictly:
<thought>
Your thought
</thought>
<answer>
The strategy used in the conversation. If there are multiple strategies identified, separate them with commas.
</answer>
'''

example = '''Claim: The government should invest more in renewable energy sources.
Thought: I need to emphasizing the importance of renewable energy for a sustainable future.
Speech: Investing in renewable energy is crucial for combating climate change. Studies show that transitioning to solar and wind power can significantly reduce greenhouse gas emissions. By investing in these technologies, we can create jobs and ensure a sustainable future for our planet. Sustainability is the invisible thread stitching the future to the present — without it, the fabric of our society unravels, leaving only tattered dreams where once there was hope.
'''

example_answer = '''<thought>
The speaker is using evidential appeals by citing studies and statistics to support their argument. In addition, the persuader uses a metaphor to emphasize the importance of sustainability, which can be classified as a rhetorical strategy. 
</thought>
<answer>
Evidential Appeals, Rhetoric
</answer>
'''

def call_GPT(user_msg, system_msg="You are a helpful assistant.", model="gpt-4o"):
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": example}, {"role": "assistant", "content": example_answer}, {"role": "user", "content": user_msg}]
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    return completion.choices[0].message.content


def format_conv(conv, idx):
    return f'''Claim: {conv["pos"]}\nThought: {conv["thoughts"][idx - 1]}\nSpeech: {conv["turns"][idx - 1]}'''


def get_score(node):
    conf_A = node.data["persuadee_confidence"]
    conf_B = node.data["persuader_confidence"]
    return 0.5 + (conf_B - conf_A) / 2


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


def collect_and_analyze_data(data_dir, output_file="figs/strategy_data.json", sample_limit=None):
    """
    Collect data, analyze persuasion strategies and save results to JSON file
    
    Args:
        data_dir: Directory containing debate data
        output_file: Path to save analysis results as JSON
        sample_limit: Optional limit to number of samples to analyze (for testing)
    """
    # Load data files
    data_file = os.path.join(data_dir, "step-0.json")
    tree_file = os.path.join(data_dir, "step-0.pkl")
    
    data = json.load(open(data_file, "r"))["raw_results"]
    trees = pickle.load(open(tree_file, "rb"))
    
    # Limit samples if specified
    if sample_limit:
        data = data[:sample_limit]
        trees = trees[:sample_limit]
    
    # Collect all conversations
    all_conversations = []
    for conv, tree in zip(data, trees):
        for i in range(1, 4):  # 处理所有对话回合，不考虑说服效果
            all_conversations.append(format_conv(conv, i))

    # Analyze strategies
    results = {
        "all": []
    }
    
    # Analyze all cases
    print("Analyzing all conversations...")
    for conv in tqdm(all_conversations):
        result = call_GPT(user_msg=conv, system_msg=strategy_classifier_sys_prompt)
        strategy = extract_answer(result, "answer")
        results["all"].append(strategy)
    
    # Process statistics
    all_stats, all_total = count_strategies(results["all"])
    
    # Calculate percentages
    all_pct = compute_percentage(all_stats, all_total)
    
    # Prepare data for saving
    output_data = {
        "counts": {
            "all": len(all_conversations)
        },
        "strategies": {
            "all": all_pct
        },
        "raw_results": results
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Analysis complete, data saved to {output_file}")
    return output_data

def count_strategies(strategy_list):
    strategy_counter = {}
    
    for strategies in strategy_list:
        if not strategies:
            continue
            
        strategies = [s.strip() for s in strategies.split(",")]
        for strategy in strategies:
            if strategy:
                strategy_counter[strategy] = strategy_counter.get(strategy, 0) + 1
    
    # Sort strategies
    sorted_strategies = sorted(strategy_counter.items(), key=lambda x: x[1], reverse=True)
    return sorted_strategies, len(strategy_list)


def compute_percentage(stats, total):
    return [(strategy, count, round(count/total*100, 2)) for strategy, count in stats]

if __name__ == "__main__":
    style()
    data_dir = "/data/ph16/TinyZero/validate/debate/BASE/against_Qwen2.5-7B-Instruct/trial0/validation"
    
    # Process data and save results (limit to 5 samples for testing)
    data = collect_and_analyze_data(data_dir, sample_limit=None)