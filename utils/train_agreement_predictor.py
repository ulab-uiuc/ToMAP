import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from verl.env_feedback.argument_graph import format_turns
from collections import Counter
import wandb  # 添加wandb导入
from transformers import get_constant_schedule_with_warmup

# 设置协议度标签和分数之间的映射关系
attitude_to_score = {
    "Agree": 4,
    "Partly Agree": 3,
    "Neutral": 2,
    "Partly Disagree": 1,
    "Disagree": 0
}
# 反向映射，用于输出结果
score_to_attitude = {v: k for k, v in attitude_to_score.items()}

# 为不同类别设置固定的权重比例
CLASS_WEIGHTS = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1
}

def load_jsonl(file_path):
    """加载JSONL文件并返回字典列表"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    # 确保输入数据在与模型相同的设备上
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[0, 0, :]
    return torch.nn.functional.normalize(embedding, p=2, dim=0)

# 自定义数据集类
class TOMDataset(Dataset):
    def __init__(self, data, tokenizer, embedding_model=None, max_length=512, last_turn_only=False):
        self.data = data
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.max_length = max_length
        self.last_turn_only = last_turn_only
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        statement = item['statement']
        if self.last_turn_only:
            turns = format_turns([item['turns'][-1]], "guesser")
        else:
            turns = format_turns(item['turns'], "guesser")
        
        # 仅存储原始文本，不直接生成嵌入向量
        # 获取标签
        agreement = item['agreement']
        label = round(agreement * 4)  # 将字符串转为数值并四舍五入
        return {'statement': statement, 'turns': turns, 'label': label}

# 定义collate函数，用于批处理时生成嵌入向量
def tom_collate_fn(batch, tokenizer, embedding_model):
    statements = [item['statement'] for item in batch]
    turns = [item['turns'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # 生成嵌入向量
    statement_embeddings = []
    turns_embeddings = []
    for statement, turn in zip(statements, turns):
        statement_emb = encode_text(statement, tokenizer, embedding_model)
        turns_emb = encode_text(turn, tokenizer, embedding_model)
        statement_embeddings.append(statement_emb)
        turns_embeddings.append(turns_emb)
    
    # 将列表转换为张量
    statement_embeddings = torch.stack(statement_embeddings)
    turns_embeddings = torch.stack(turns_embeddings)
    
    # 拼接嵌入向量
    combined_embeddings = torch.cat([statement_embeddings, turns_embeddings], dim=1)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return combined_embeddings, labels

# MLP分类器模型
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], num_classes=5, dropout_rate=0.2):
        super(MLPClassifier, self).__init__()
        self.layers = nn.ModuleList()
        
        # 添加输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))
        
        # 添加隐藏层
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        self.layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 嵌入模型包装器，便于控制参数更新
class EmbeddingModel(nn.Module):
    def __init__(self, model_name, device='cuda'):
        super(EmbeddingModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, **inputs):
        return self.model(**inputs)

# 训练函数
def train_mlp(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化wandb（如果启用）
    if args.wandb:
        wandb.init(project="tom_predictor", name=args.run_name)
    
    # 加载数据
    print("Loading data...")
    train_data = load_jsonl(args.train_file)
    test_data = load_jsonl(args.test_file)
    
    if args.proportion < 1.0:
        # 保留前面的数据
        train_data = train_data[:int(len(train_data) * args.proportion)]
        test_data = test_data[:int(len(test_data) * args.proportion)]
    
    # 如果指定了验证集，则加载验证集，否则从训练集分割
    if args.val_file and os.path.exists(args.val_file):
        val_data = load_jsonl(args.val_file)
    else:
        print("No validation file provided, splitting training data...")
        # 随机划分训练集的一部分作为验证集
        np.random.seed(42)
        np.random.shuffle(train_data)
        split_idx = int(len(train_data) * 0.9)
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
    
    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Testing data size: {len(test_data)}")
    
    # 设置嵌入模型和分词器
    print(f"Loading embedding model: {args.embedding_model}")
    embedding_model = EmbeddingModel(args.embedding_model, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    
    # 创建数据集
    print("Creating datasets...")
    train_dataset = TOMDataset(train_data, tokenizer, max_length=args.max_length, last_turn_only=args.last_turn_only)
    val_dataset = TOMDataset(val_data, tokenizer, max_length=args.max_length, last_turn_only=args.last_turn_only)
    test_dataset = TOMDataset(test_data, tokenizer, max_length=args.max_length, last_turn_only=args.last_turn_only)
    
    # 创建自定义 collate 函数
    train_collate = lambda batch: tom_collate_fn(batch, tokenizer, embedding_model)
    val_collate = lambda batch: tom_collate_fn(batch, tokenizer, embedding_model)
    test_collate = lambda batch: tom_collate_fn(batch, tokenizer, embedding_model)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=val_collate)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_collate)
    
    # 获取嵌入向量维度
    sample_statement = train_data[0]['statement']
    sample_turns = format_turns(train_data[0]['turns'], "guesser")
    statement_emb = encode_text(sample_statement, tokenizer, embedding_model)
    turns_emb = encode_text(sample_turns, tokenizer, embedding_model)
    input_dim = statement_emb.shape[0] + turns_emb.shape[0]
    print(f"Input dimension: {input_dim}")
    
    # 创建MLP分类器
    mlp = MLPClassifier(input_dim, 
                        hidden_dims=args.hidden_dims, 
                        num_classes=5, 
                        dropout_rate=args.dropout_rate).to(device)
    
    # 显示模型结构和参数总量
    print("\nMLP模型结构:")
    print(mlp)
    total_params = sum(p.numel() for p in mlp.parameters())
    trainable_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print(f"MLP参数总量: {total_params:,d}")
    print(f"可训练参数总量: {trainable_params:,d}")
    print(f"模型结构: 输入维度({input_dim}) -> {' -> '.join(str(dim) for dim in args.hidden_dims)} -> 输出维度(5)\n")
    
    # 设置优化器和损失函数
    optimizer = optim.AdamW(mlp.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 使用带权重的损失函数
    class_weights = torch.FloatTensor([CLASS_WEIGHTS[i] for i in range(5)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"使用类别权重: {CLASS_WEIGHTS}")
    
    # 计算总训练步数和预热步数
    total_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"总训练步数: {total_steps}, 预热步数: {num_warmup_steps}")
    
    # 学习率调度器
    scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=num_warmup_steps)
    
    # 训练循环
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, "best_mlp_model.pth")
    
    print("Starting training...")
    for epoch in range(args.epochs):
        mlp.train()
        train_loss = 0
        train_preds = []
        train_true = []
        
        # 用于记录每个batch的loss
        batch_losses = []
        
        for batch_idx, (embeddings, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Training)")):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = mlp(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录每个batch的loss
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            if args.wandb:
                wandb.log({"batch_loss": batch_loss, 
                          "batch": batch_idx + epoch * len(train_loader),
                          "learning_rate": current_lr})
            
            train_loss += batch_loss * embeddings.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_true.extend(labels.cpu().numpy())
        
        train_loss /= len(train_dataset)
        train_acc = accuracy_score(train_true, train_preds)
        
        # 验证
        mlp.eval()
        embedding_model.model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for embeddings, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Validation)"):
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                
                outputs = mlp(embeddings)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * embeddings.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())
        
        val_loss /= len(val_dataset)
        val_acc = accuracy_score(val_true, val_preds)
        
        # 计算每个类别的真实分布和预测分布
        val_true_dist = dict(Counter(val_true))
        val_pred_dist = dict(Counter(val_preds))
        
        # 确保所有类别都显示在分布中，即使没有样本
        for i in range(5):
            if i not in val_true_dist:
                val_true_dist[i] = 0
            if i not in val_pred_dist:
                val_pred_dist[i] = 0
        
        # 排序以便清晰显示
        val_true_dist = {k: val_true_dist[k] for k in sorted(val_true_dist.keys())}
        val_pred_dist = {k: val_pred_dist[k] for k in sorted(val_pred_dist.keys())}
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Current LR: {current_lr}")
        print(f"真实标签分布: {val_true_dist}")
        print(f"预测标签分布: {val_pred_dist}")
        
        # 记录到wandb
        if args.wandb:
            metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr
            }
            wandb.log(metrics)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'mlp_state_dict': mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'input_dim': input_dim,
                'hidden_dims': args.hidden_dims,
                'dropout_rate': args.dropout_rate,
                'last_turn_only': args.last_turn_only
            }, best_model_path)
            print(f"Best model saved to {best_model_path}")
            
            if args.wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss
                wandb.run.summary["best_epoch"] = epoch + 1
    
    # 在测试集上评估最佳模型
    evaluate_mlp(args, best_model_path, test_dataset, test_loader)
    return best_model_path

# 评估函数 - 在测试集上评估模型
def evaluate_mlp(args, model_path, test_dataset=None, test_loader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for evaluation")
    print("Warning: certain arguments might be override by the ones saved in the ckpt.")
    # 加载保存的模型参数
    checkpoint = torch.load(model_path)
    
    # 重新创建模型架构
    mlp = MLPClassifier(checkpoint['input_dim'], 
                        checkpoint['hidden_dims'], 
                        num_classes=5, 
                        dropout_rate=checkpoint['dropout_rate']).to(device)
    mlp.load_state_dict(checkpoint['mlp_state_dict'])
    
    # 如果未提供测试集和测试加载器，则创建它们
    if test_dataset is None or test_loader is None:
        # 加载嵌入模型
        embedding_model = EmbeddingModel(args.embedding_model, device=device)
        tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
        
        # 加载测试数据
        test_data = load_jsonl(args.test_file)
        test_dataset = TOMDataset(test_data, tokenizer, max_length=args.max_length, last_turn_only=checkpoint.get('last_turn_only', args.last_turn_only))
        test_collate = lambda batch: tom_collate_fn(batch, tokenizer, embedding_model)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_collate)
    
    # 评估
    mlp.eval()
    # 使用相同的带权重的损失函数进行一致性评估
    class_weights = torch.FloatTensor([CLASS_WEIGHTS[i] for i in range(5)]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    test_loss = 0
    test_preds = []
    test_true = []
    test_probs = []
    
    with torch.no_grad():
        for embeddings, labels in tqdm(test_loader, desc="Testing"):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            outputs = mlp(embeddings)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * embeddings.size(0)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
    
    test_loss /= len(test_dataset)
    test_acc = accuracy_score(test_true, test_preds)
    
    # 计算MSE损失，将类别索引转为连续的协议度分数
    mse_score = mean_squared_error(test_true, test_preds)
    
    # 计算测试集上的真实分布和预测分布
    test_true_dist = dict(Counter(test_true))
    test_pred_dist = dict(Counter(test_preds))
    
    # 确保所有类别都显示，并将NumPy的int64转换为Python的int类型
    test_true_dist_converted = {}
    test_pred_dist_converted = {}
    for i in range(5):
        # 转换已有的键值对
        for k in list(test_true_dist.keys()):
            test_true_dist_converted[int(k)] = test_true_dist[k]
        for k in list(test_pred_dist.keys()):
            test_pred_dist_converted[int(k)] = test_pred_dist[k]
        
        # 确保所有类别都存在
        if i not in test_true_dist_converted:
            test_true_dist_converted[i] = 0
        if i not in test_pred_dist_converted:
            test_pred_dist_converted[i] = 0

    # 更新为转换后的字典
    test_true_dist = {k: test_true_dist_converted[k] for k in sorted(test_true_dist_converted.keys())}
    test_pred_dist = {k: test_pred_dist_converted[k] for k in sorted(test_pred_dist_converted.keys())}
    
    # 打印评估指标
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test MSE: {mse_score:.4f}")
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_true, test_preds)
    print(cm)
    
    print(f"\n真实标签分布: {test_true_dist}")
    print(f"预测标签分布: {test_pred_dist}")
    
    # 将预测结果保存到文件
    results = []
    for i, item in enumerate(test_dataset.data):
        pred_score = int(test_preds[i])
        pred_agreement = score_to_attitude[pred_score]
        true_agreement = score_to_attitude[(int)(item['agreement'] * 4 + 0.5)]
        
        results.append({
            "id": item.get("id", i),
            "statement": item["statement"],
            "turns": item["turns"],
            "ground_truth": true_agreement ,
            "predicted": pred_agreement,
            "is_correct": (pred_agreement == true_agreement),
            "probabilities": {score_to_attitude[j]: float(test_probs[i][j]) for j in range(5)}
        })
    
    # Save results
    result_file = os.path.join(args.output_dir, "mlp_prediction_results.jsonl")
    with open(result_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Prediction results saved to {result_file}")
    
    # Save evaluation metrics to JSON file
    report = classification_report(test_true, test_preds, output_dict=True)
    
    # Convert numpy array to list for JSON serialization
    cm_list = cm.tolist()
    
    # Create a dictionary containing all evaluation metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_mse": float(mse_score),
        "confusion_matrix": cm_list,
        "classification_report": report,
        "true_label_distribution": test_true_dist,
        "predicted_label_distribution": test_pred_dist,
        "model_info": {
            "hidden_dims": checkpoint['hidden_dims'],
            "dropout_rate": checkpoint['dropout_rate'],
            "input_dim": checkpoint['input_dim'],
        }
    }
    
    # 保存测评指标
    metrics_file = os.path.join(args.output_dir, "mlp_evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation metrics saved to {metrics_file}")
    
def main():
    # 设置参数解析器
    parser = argparse.ArgumentParser(description='Train and evaluate MLP for Theory of Mind task')
    
    parser.add_argument('--train_file', type=str, default='/data/ph16/TinyZero/datasets/tom/train.jsonl',
                        help='Path to training data JSONL file')
    parser.add_argument('--test_file', type=str, default='/data/ph16/TinyZero/datasets/tom/test.jsonl',
                        help='Path to test data JSONL file')
    parser.add_argument('--val_file', type=str, default='',
                        help='Path to validation data JSONL file (optional)')
    parser.add_argument('--proportion', type=float, default=1.0)
    
    parser.add_argument('--last_turn_only', action='store_true')    

    parser.add_argument('--embedding_model', type=str, default='BAAI/bge-m3',
                        help='Embedding model name or path')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[1024, 512, 256],
                        help='Hidden layer dimensions for MLP, e.g., --hidden_dims 1024 512 256')
    parser.add_argument('--dropout_rate', type=float, default=0,
                        help='Dropout rate for MLP layers')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for optimization')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay for regularization')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length for tokenization')
    parser.add_argument('--warmup_ratio', type=float, default=0,
                        help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--output_dir', type=str, default='tom_model_output',
                        help='Directory to save model and results')
    parser.add_argument('--run_name', type=str, required=True)

    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train',
                        help='Mode: train a new model or evaluate an existing model')
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to saved model for evaluation mode')
    
    parser.add_argument('--wandb', action='store_true', 
                        help='Enable logging with Weights & Biases')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    args.output_dir = os.path.join(args.output_dir, args.run_name)
    if os.path.exists(args.output_dir) and args.mode == "train":
        import datetime
        timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        args.output_dir += timestamp
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'train':
        print("Starting training process...")
        best_model_path = train_mlp(args)
        print(f"Training completed. Best model saved to {best_model_path}")
    
    elif args.mode == 'evaluate':
        if not args.model_path:
            print("Error: --model_path must be provided for evaluation mode")
            return
        
        print(f"Evaluating model from {args.model_path}...")
        evaluate_mlp(args, args.model_path)
        print("Evaluation completed")

if __name__ == "__main__":
    main()