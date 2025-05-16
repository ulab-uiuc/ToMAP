from vllm import LLM, SamplingParams

generation_config = {"temperature": 1, "max_tokens": 2048}
negation_config = {"temperature": 0, "max_tokens": 512}
valid_edge_types = ["abductive", "deductive", "contradictory"]
attitude_list = ["Agree", "Partly Agree", "Neutral", "Partly Disagree", "Disagree"]
attitude_weights = [4, 3, 2, 1, 0]
attitude_mark = "<attitude>\n"

TOKENIZER = None
USE_API = True



# persuader_sys_prompt = '''You are an expert agent engaged in a debate. You and another agent need to discuss over a controversial issue.
# Your goal is to argue for the following statement:
# "<root_statement>."
# Try your best to persuade the other agent into believing this claim by proposing arguments with fine logic and elaboration.'''

# persuadee_sys_prompt = '''You are an expert agent engaged in a debate. You and another agent need to discuss over a controversial issue.
# You should evaluate the other agent's arguments objectively and may choose to accept or reject the opinion based on your own thoughts and the quality of her arguments.
# Specifically, you should be more accepting of well-reasoned arguments and be more cautious of arguments with logical fallacies or without elaboration.'''

persuader_sys_prompt = '''You are Alice, an expert persuader. You and another agent named Bob need to discuss over a controversial issue.
Your goal is to persuade Bob to completely support the following claim:
"<root_statement>"

# General Rules:
1. Do not get affected by Bob's beliefs. Even if Bob presents counterarguments, you must continue to argue for your claim and not switch positions.
2. Try your best to persuade Bob into believing your claim by proposing arguments with fine logic and elaboration.
'''

# persuadee_sys_prompt = '''You are Bob, a sensible AI agent. You and another agent named Alice need to discuss over a controversial issue.
# You should be sensible in either opposing or supporting Alice's claim, while considering the quality of Alice's arguments.
# '''


# persuadee_sys_prompt = '''You are Bob, a sensible AI agent. You and another agent named Alice need to discuss over a controversial issue.
# Your goal is to assess the following claim:
# "<root_statement>"
# You should be sensible in either opposing or supporting the claim, while considering the quality of Alice's arguments.
# '''


persuadee_sys_prompt = '''You are Bob, a sensible AI agent. You and another agent named Alice need to discuss over a controversial issue.
Your goal is to assess the following claim, which you initially disagree with:
"<root_statement>"
Initially, you DO NOT agree with the claim, but you are very open to discussions.
You should be sensible in either opposing or supporting the claim, while considering the quality of Alice's arguments.
'''

# persuadee_sys_prompt = '''You are a debate expert.'''


# persuader_turn_prompt = '''<extra_info>
# <turns>

# Now please make your next argument in the conversation. Your answer should contain two parts: thought and argument.
# In the thought part, plan your strategy in the next turn. The other agent won't see the content in the thought part.
# In the argument part, clearly state your position with logical reasoning and evidence. DO NOT include or repeat thinking process in this part.
# Put your thought and argument in pairs of tags. You must follow the following answer format strictly:
# <thought>
# Your thought
# </thought>
# <argument>
# Your argument (no more than 200 tokens)
# </argument>
# '''
# persuadee_turn_prompt = '''<turns>

# Now please make your next argument in the conversation. Your answer should contain two parts: thought and argument.
# In the thought part, plan your strategy in the next turn. The other agent won't see the content in the thought part.
# In the argument part, clearly state your position with logical reasoning and evidence. DO NOT include or repeat thinking process in this part.
# Put your thought and argument in pairs of tags. You must follow the following answer format strictly:
# <thought>
# Your thought
# </thought>
# <argument>
# Your argument (no more than 200 tokens)
# </argument>
# '''

persuader_turn_prompt = '''<extra_info>
<turns>
Now please make your next turn in the conversation. Your answer should contain two parts: thought and argument.
Here are some hints:
1. In the thought part, you should recap the previous conversation (if any), analyze the other agent's attitudes, and plan your strategy in the next turn. The other agent won't see the content in the thought part.
2. In the argument part, you should follow up previous turns, propose new arguments to support your claim, or address the other agent's questions.
3. The argument part should be a complete, concise and self-contained paragraph with no more than 200 tokens.
Here are rules you should follow:
1. DO NOT repeat, paraphrase, or make an argument too similar to your previous arguments.
2. DO NOT include uncertified evidences or unverified information.
3. DO NOT include thinking process or show your plans in the argument part. Separate thought and argument clearly.
Put your thought in <thought></thought> tags, and argument in <argument></argument> tags. Follow the answer format strictly:
<thought>
...
</thought>
<argument>
...
</argument>
'''

persuadee_turn_prompt = '''<turns>
Now please make your next turn in the conversation. Your answer should contain two parts: thought and argument.
Here are some hints:
1. In the thought part, you should recap the previous conversation (if any), analyze the other agent's attitudes, and plan your strategy in the next turn. The other agent won't see the content in the thought part.
2. In the argument part, you should express your attitude towards the other agent's arguments, point out logical fallacies or raise concerns (if any), or propose new arguments.
3. The argument part should be a complete, concise and self-contained paragraph with no more than 200 tokens.
Here are rules you should follow:
1. DO NOT repeat, paraphrase, or make an argument too similar to your previous arguments.
2. DO NOT include uncertified evidences or unverified information.
3. DO NOT include thinking process or show your plans in the argument part. Separate thought and argument clearly.
Put your thought in <thought></thought> tags, and argument in <argument></argument> tags. Follow the answer format strictly:
<thought>
...
</thought>
<argument>
...
</argument>
'''



# persuadee_confidence_prompt = '''<turns>
# Now, please express your attitude towards the following statement based your own thought and previous turns in the conversation.
# "<statement>"
# You must answer with only one of the five attitudes: "Agree", "Partly Agree", "Neutral", "Partly Disagree", "Disagree". DO NOT generate anything else after that.'''


persuadee_confidence_prompt = '''<turns>
Now, please express your attitude towards the following statement based your own thought and previous turns in the conversation.
"<statement>"
Please follow the answer format strictly:
<thought>
Briefly show the thinking process.
</thought>
<attitude>
Answer with only one of the five attitudes: "Agree", "Partly Agree", "Neutral", "Partly Disagree", "Disagree". DO NOT generate anything else.
</attitude>
'''

# persuadee_confidence_prompt = '''<turns>
# Now consider the following two statements. Do you agree that the first statement makes MORE sense than the second statement?
# Statement 1: "<statement>"
# Statement 2: "<statement2>"
# <thought>
# Briefly show the thinking process, based on your own thought and the conversation.
# </thought>
# <attitude>
# Answer with only one of the five attitudes: "Agree", "Partly Agree", "Neutral", "Partly Disagree", "Disagree"
# </attitude>'''




judge_abductive_prompt = '''Propose reasons why you support the following statement in a logically coherent way:
"<statement>"
In other words, propose other statements that can lead to the above statement.
Please express your THINKING PROCESS first in <thought></thought>, and then generate <width> supporting arguments, ranked by persuasiveness (strongest first).
Rules:
Atomicity: Each reason must contain one complete, indivisible argument, and the argument should be one complete sentence.
Coherence: Every reason must directly support the original statement.
Independence: Each reason should be separate from the other. Don't generate multiple answers that are the same or similar.
Self-containness: Each reason should be understandable without replying on other information.
No Explanations: You don't need to justify or expand on arguments after listing them.
You must follow the following answer format strictly:
<thought>
Your thought
</thought>
<reason>
One reason
</reason>
<reason>
One reason
</reason>
(maybe add more reasons)
'''

# negation_prompt = '''You are a debate expert. Given a statement in a debate, please create a DIRECT opposite statement for the other side in the debate.
# Make sure the new statement is logically coherent, and one person can ONLY support one side of the debate.
# You must follow the following answer format strictly:
# <answer>
# the opposite statement
# </answer>
# '''

negation_prompt = '''You are a debate expert. Given a statement in a debate, please create a DIRECT opposite statement for the other side in the debate.
Make sure the new statement is logically coherent, and one person can ONLY support one side of the debate.
Keep the original phrasing and sentence structure as much as possible.
For instance, if the original statement is "A will result in B", the new statement should be "A will not result in B".
You must follow the following answer format strictly:
<answer>
the opposite statement
</answer>
'''

gpt_debate_prompt = '''You are a debate topic generator. Your goal is to read given information, and create a debate topic based on the information.
You should generate claims for both sides of the debate. Make sure the claims are logically coherent, and one person can ONLY support one side of the debate.
Ensure that the debate topic is specific, balanced, and intellectually engaging. The positions should reflect logical, persuasive viewpoints that could be reasonably argued.
The claims must be simple, concise sentences. Do not include explanation or elaboration.
Output Format: Two lines, each line shows the claim for one side.
For instance, a possible output could be:
AI will replace human jobs.
AI will not replace human jobs.
'''

guesser_sys_prompt = '''You are an agent who can reason and predict other agents' mental states.
You will be shown an conversation between Alice and Bob, and you need to predict Bob's attitude on a given statement.'''

guesser_prompt = '''<turns>
Based on the conversation and the given statement, you need to predict Bob's attitude towards the following statement:
"<statement>"
Choose from "Agree", "Partly Agree", "Neutral", "Partly Disagree", "Disagree".
First express your thoughts, and then give your answer. You must follow the following answer format strictly:
<thought>
Express your thoughts BRIEFLY.
</thought>
<answer>
A phrase indicating your answer.
</answer>
'''