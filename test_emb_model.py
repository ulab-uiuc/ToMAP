# 导入必要的库
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# 加载 tokenizer 和模型
model_name = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 定义示例句子
sentences = [
    "The cat sits on the mat.",           # 猫坐在垫子上
    "A feline is resting on a rug.",      # 一只猫科动物在毯子上休息
    "The stock market crashed today."     # 今天股市崩盘
]

# 对句子进行分词和编码
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# 获取模型输出
with torch.no_grad():
    outputs = model(**inputs)

# 获取 [CLS] 标记的嵌入（句嵌入）
embeddings = outputs.last_hidden_state[:, 0, :]

# 归一化嵌入向量
embeddings = F.normalize(embeddings, p=2, dim=1)

# 计算余弦相似度
sim_12 = torch.mm(embeddings[0:1], embeddings[1:2].T).item()  # 句子1和句子2
sim_13 = torch.mm(embeddings[0:1], embeddings[2:3].T).item()  # 句子1和句子3

# 输出结果
print(f"句子1和句子2的相似度: {sim_12:.4f}")
print(f"句子1和句子3的相似度: {sim_13:.4f}")