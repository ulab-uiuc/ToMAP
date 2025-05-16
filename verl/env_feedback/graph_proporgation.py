import itertools
import math
import dataclasses
from dataclasses import dataclass
from typing import List, Union, Tuple
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import os
import json
from treelib import Tree
from openai import OpenAI
import requests
from verl.env_feedback.argument_graph import calc_argument_vectors, get_hidden_states, print_tree
from verl.env_feedback.debate_prompts import *
from verl.llm_agent.batch_inference import external_batch_inference
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from verl.utils.dataset.rl_dataset import collate_fn, tokenizer_wrapper
import pickle
import torch
import random

def calc_sim(a, b, method='euclidean', normalize=True):
    """计算两个向量的相似度
    
    Args:
        a (torch.Tensor): 第一个向量
        b (torch.Tensor): 第二个向量 
        method (str): 相似度计算方法，可选值：
            - 'cosine': 余弦相似度 (默认)
            - 'euclidean': 欧氏距离相似度
            - 'dot': 点积相似度
            - 'manhattan': 曼哈顿距离相似度
            - 'jaccard': 杰卡德相似度
        normalize (bool): 是否在计算之前对向量进行归一化
    
    Returns:
        float: 相似度值
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("输入必须是torch.Tensor类型")
    if a.dim() != 1 or b.dim() != 1:
        raise ValueError("输入tensor必须是一维的")
    if a.size(0) != b.size(0):
        raise ValueError("输入tensor的长度必须相同")
    
    # 对向量进行归一化预处理
    if normalize:
        a = torch.nn.functional.normalize(a, p=2, dim=0)
        b = torch.nn.functional.normalize(b, p=2, dim=0)

    if method == 'cosine':
        sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
    elif method == 'euclidean':
        distance = torch.sqrt(torch.sum((a - b) ** 2))
        sim = torch.exp(-distance)  # 将距离转换为相似度
    elif method == 'dot':
        # 如果已经归一化，直接计算点积
        sim = torch.dot(a, b)
    elif method == 'manhattan':
        distance = torch.sum(torch.abs(a - b))
        sim = 1 / (1 + distance)  # 将距离转换为相似度
    elif method == 'jaccard':
        threshold = 0.5  # 二值化阈值
        a_binary = (a > threshold).float()
        b_binary = (b > threshold).float()
        intersection = torch.sum(a_binary * b_binary)
        union = torch.sum((a_binary + b_binary) > 0)
        sim = intersection / union if union > 0 else torch.tensor(0.0)
    else:
        raise ValueError(f"不支持的相似度计算方法: {method}")

    return sim.item()



def predict_graph_nodes(tree, vectors):
    # tree是treelib对象，包含了所有节点
    # 每个节点对应一个初始向量
    # 预测label为node.data["persuadee_confidence"]，未知的初始值为-1
    # 图结构可以假设为完全图，用calc_sim计算边权
    # 使用你说的第一种方法完成计算
    pass


import numpy as np
from treelib import Tree

def predict_graph_nodes(tree, vectors):
    # 获取所有节点并建立索引映射
    nodes = tree.all_nodes()
    n = len(nodes)
    node_to_idx = {node.identifier: i for i, node in enumerate(nodes)}
    
    bias=0.5
    
    # 构建邻接矩阵 A（基于相似度）
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = calc_sim(vectors[i], vectors[j]) - bias
            A[i, j] = sim
            A[j, i] = sim
    
    print(A)
    # 计算度矩阵 D 和转移矩阵 P
    D = np.diag(np.sum(A, axis=1))
    D_inv = np.linalg.inv(D)
    P = np.dot(D_inv, A)
    
    # 初始化标签向量 Y 和已知标签掩码
    Y = np.zeros(n)
    Y_init = np.zeros(n)
    known_mask = np.zeros(n, dtype=bool)
    for node in nodes:
        idx = node_to_idx[node.identifier]
        if node.data["persuadee_confidence"] != -1:  # 假设 -1 表示未知节点
            Y[idx] = node.data["persuadee_confidence"]
            Y_init[idx] = node.data["persuadee_confidence"]
            known_mask[idx] = True
    
    # 设置超参数
    alpha = 1
    max_iter = 100
    tol = 1e-5
    
    # 标签传播迭代
    print(Y)
    for _ in range(max_iter):
        Y_new = alpha * np.dot(P, Y) + (1 - alpha) * Y_init
        Y_new[known_mask] = Y_init[known_mask]  # 固定已知标签节点
        if np.linalg.norm(Y_new - Y) < tol:  # 检查收敛
            break
        Y = Y_new.copy()
        print(Y)
    
    # 将预测值赋值回节点
    for node in nodes:
        idx = node_to_idx[node.identifier]
        node.data["persuadee_confidence"] = Y[idx]
    
    return tree



if __name__ == "__main__":
    tree = pickle.load(open("/data/ph16/TinyZero/datasets/debate/Qwen2.5-7B-Instruct_1_10_abd/test.pkl", "rb"))[0]
    vectors = torch.load('/home/ph16/TinyZero-dev/states.pt')[0]
    print("Ground Truth: ")
    print(np.array([node.data["persuadee_confidence"] for node in tree.all_nodes()]))
    orig_tree = copy.deepcopy(tree)
    for node in tree.all_nodes():
        if random.random() < 0.5:
            node.data["persuadee_confidence"] = -1
    print("Iterations: ")
    print(np.array([node.data["persuadee_confidence"] for node in tree.all_nodes()]))
    tree = predict_graph_nodes(tree, vectors)
    # print_tree(tree)