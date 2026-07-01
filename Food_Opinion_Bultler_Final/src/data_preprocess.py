import pandas as pd
import re
import jieba
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch

# 全局配置
TEXT_COL = "text"
ASPECT_COL = "dim_labels"
SENTI_COL = "sentiment_labels"
RISK_COL = "risk_label"
MAX_TEXT_LEN = 500
ASPECT_NUM = 8  # 8个评价维度
RISK_NUM = 4    # 4级风险等级

# 文本清洗规则
def clean_text(text):
    """文本清洗：去除特殊符号、换行、多余空格"""
    text = str(text)
    text = re.sub(r'[\n\r\t]', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。？！：；""''()（）、]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 维度标签转多标签二值化
def aspect_str2tensor(aspect_str):
    """维度标签字符串"0,3" → 8维二值张量"""
    tensor = torch.zeros(ASPECT_NUM, dtype=torch.float32)
    if pd.isna(aspect_str) or str(aspect_str).strip() == "":
        return tensor
    for idx in str(aspect_str).split(","):
        try:
            idx = int(idx.strip())
            if 0 <= idx < ASPECT_NUM:
                tensor[idx] = 1.0
        except:
            continue
    return tensor

# 情感标签转张量
def senti_str2tensor(senti_str):
    """情感标签字符串"0,1,1..." → 8维张量"""
    tensor = torch.zeros(ASPECT_NUM, dtype=torch.float32)
    if pd.isna(senti_str) or str(senti_str).strip() == "":
        return tensor
    arr = str(senti_str).split(",")
    for i in range(min(len(arr), ASPECT_NUM)):
        try:
            tensor[i] = float(arr[i].strip())
        except:
            continue
    return tensor

# 张量转标签字符串
def tensor2aspect_str(tensor):
    """8维张量 → 维度标签字符串"0,3" """
    indices = torch.where(tensor > 0.5)[0].cpu().numpy()
    return ",".join([str(i) for i in indices]) if len(indices) > 0 else ""

def tensor2senti_str(tensor):
    """8维张量 → 情感标签字符串"0,1,1..." """
    arr = (tensor > 0.5).int().cpu().numpy()
    return ",".join([str(i) for i in arr])

# 加载并预处理数据集
def load_and_preprocess_data(file_path):
    """加载数据集，清洗文本，过滤长度，划分训练测试集"""
    # 读取数据
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    # 文本清洗
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
    # 过滤文本长度≤500
    df = df[df[TEXT_COL].astype(str).apply(len) <= MAX_TEXT_LEN].reset_index(drop=True)
    # 标签转换
    df['aspect_tensor'] = df[ASPECT_COL].apply(aspect_str2tensor)
    df['senti_tensor'] = df[SENTI_COL].apply(senti_str2tensor)
    df['risk_label'] = df[RISK_COL].astype(int)
    # 划分训练测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[RISK_COL])
    print(f"数据集加载完成：总样本{len(df)}条，训练集{len(train_df)}条，测试集{len(test_df)}条")
    return train_df, test_df

# 分词函数（用于基线模型）
def tokenize_text(text):
    """中文分词，返回空格分隔的分词结果"""
    return " ".join(jieba.lcut(clean_text(text)))