import pickle
from verl.env_feedback.argument_graph import print_tree
import os


path = "/data/ph16/TinyZero_test_bt/datasets/debate_phi-4/test_argument_tree.pkl"
data = pickle.load(open(path, "rb"))

print(data[2])

print('-'*50)
path = "/data/ph16/TinyZero/datasets/debate/test_argument_tree.pkl"
data = pickle.load(open(path, "rb"))

print(data[2])
# print(f"原始数据中有 {len(data)} 棵树")

# # 修改所有树的所有节点的confidence值
# def modify_node_confidence(tree):
#     """修改节点的persuadee_confidence和persuader_confidence为-1"""
#     for node in tree.all_nodes():
#         node.data["persuadee_confidence"] = -1
#         node.data["persuader_confidence"] = -1
# # 处理所有树
# modified_count = 0
# for i, tree in enumerate(data):
#     modify_node_confidence(tree)
#     modified_count += 1
    
#     # 每100棵树打印一次进度
#     if (i + 1) % 1000 == 0:
#         print(f"已处理 {i + 1}/{len(data)} 棵树")

# print(f"已修改 {modified_count} 棵树的confidence值")

# # 保存修改后的数据回原文件
# with open(path, "wb") as f:
#     pickle.dump(data, f)

# print(f"修改后的数据已保存回 {path}")

# # 验证一棵树的修改结果
# print("\n修改后的第一棵树示例:")
# print_tree(data[0])
