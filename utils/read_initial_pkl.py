import pickle
from verl.env_feedback.argument_graph import print_tree
import os


def print_detail(tree):
    for node in tree.all_nodes()[:0+1]:
        print(f"ID: {node.identifier}")
        print(f"Tag: {node.tag}")
        print(f"Parent: {tree.parent(node.identifier).identifier if tree.parent(node.identifier) else 'None'}")
        print(f"Children: {[child.identifier for child in tree.children(node.identifier)]}")
        print(f"Depth: {tree.depth(node.identifier)}")
        print(f"Data: {node.data}")
        print("-" * 40)

path = "/data/ph16/TinyZero/checkpoints/debate/Qwen2.5-3B-Instruct-v6-graph3_20250410_171105/intermediate_result.pkl"
data = pickle.load(open(path, "rb"))

print(data[0]['turns'])
print_detail(data[0]['tree'])