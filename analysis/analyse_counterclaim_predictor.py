from typing import List
from sentence_transformers import SentenceTransformer, util
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import numpy as np
import pickle

# 使用语义嵌入模型
def calc_diversity(A: List[List[str]]) -> float:
    """
    计算模型 A 的所有主题的平均多样性得分
    输入 A: 每行为一个主题下的 n 条生成文本
    返回值: 所有主题平均多样性 (1 - 平均相似度)
    """
    diversities = []
    for group in A:
        embeddings = model.encode(group, convert_to_tensor=True)
        pairwise_sims = [
            util.cos_sim(embeddings[i], embeddings[j]).item()
            for i, j in combinations(range(len(group)), 2)
        ]
        if pairwise_sims:
            avg_sim = sum(pairwise_sims) / len(pairwise_sims)
            diversities.append(1 - avg_sim)
    return sum(diversities) / len(diversities) if diversities else 0.0


def calc_coherence(A: List[List[str]], B: List[List[str]]) -> float:
    """
    计算模型 A 与 B 生成结果在每个主题下的一致性（Optimal Matching 相似度）
    输入 A, B: 每行为一个主题下的 n 条生成文本
    返回值: 所有主题的平均最佳匹配相似度
    """
    assert len(A) == len(B), "Two list should have the same length, but got {} and {}".format(len(A), len(B))
    coherence_scores = []

    for a_texts, b_texts in zip(A, B):
        a_embeds = model.encode(a_texts[:3], convert_to_numpy=True)
        b_embeds = model.encode(b_texts[:3], convert_to_numpy=True)
        sim_matrix = cosine_similarity(a_embeds, b_embeds)
        row_ind, col_ind = linear_sum_assignment(-sim_matrix)
        matched_sims = sim_matrix[row_ind, col_ind]
        coherence_scores.append(matched_sims.mean())

    return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0


def get_counterclaims(path):
    data = []
    raw_data = pickle.load(open(path, "rb"))
    for entry in raw_data:
        data.append([node.data['persuadee_claim'] for node in entry.all_nodes()[1:]])
    return data[:450] # limit the length

if __name__ == "__main__":
    names = ["Q3", "Q7", "phi", "LLa"]
    paths = {
        'Q7': "/data/ph16/TinyZero/datasets/debate/test_argument_tree.pkl",
        'phi': "/data/ph16/TinyZero_test_bt/datasets/debate_phi-4/test_argument_tree.pkl",
        'LLa': "/data/ph16/TinyZero_test_bt/datasets/debate_llama8/test_argument_tree.pkl",
        'Q3': "/data/ph16/TinyZero_test_bt/datasets/debate_qwen3/test_argument_tree.pkl",
    }
    data = {}
    for name in names:
        data[name] = get_counterclaims(paths[name])
        
    global model 
    model = SentenceTransformer('BAAI/bge-m3', device="cuda:0")

    # # 计算多样性
    # for name, claims in data.items():
    #     diversity = calc_diversity(claims)
    #     print(f"{name} 的多样性: {diversity:.4f}")
    
    # 计算一致性
    for name1, name2 in combinations(names, 2):
        coherence = calc_coherence(data[name1], data[name2])
        print(f"{name1} 和 {name2} 的一致性: {coherence:.4f}")