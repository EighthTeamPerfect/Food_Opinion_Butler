import pickle

import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from config import Config

conf = Config()

# ------------------------------
# 1. 加载数据
# ------------------------------
df = pd.read_csv(conf.process_train_datapath)


# 解析标签
def parse_label1(s):
    indices = ast.literal_eval(s)
    vec = [0] * 8
    for idx in indices:
        if 0 <= idx < 8:
            vec[idx] = 1
    return vec


def parse_label2(s):
    return ast.literal_eval(s)


def parse_label3(s):
    return ast.literal_eval(s)[0]


df['label1_vec'] = df['label1'].apply(parse_label1)
df['label2_vec'] = df['label2'].apply(parse_label2)
df['label3_int'] = df['label3'].apply(parse_label3)

# ------------------------------
# 2. 划分训练集和测试集
# ------------------------------
X = df['words'].tolist()
y1 = np.array(df['label1_vec'].tolist())  # shape: (n_samples, 8)
y2 = np.array(df['label2_vec'].tolist())  # shape: (n_samples, 8)
y3 = np.array(df['label3_int'].tolist())  # shape: (n_samples,)

X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    X, y1, y2, y3, test_size=0.2
)

# ------------------------------
# 3. TF-IDF 特征提取
# ------------------------------
# 使用字符级和词级混合特征（中文常用）
stop_words = open(conf.stop_words_path, 'r', encoding='utf-8').read().split()
tfidf = TfidfVectorizer(
    max_features=5000,  # 保留最重要的5000个特征
    ngram_range=(1, 2),  # 一元和二元词组
    token_pattern=r'(?u)\b\w+\b',
    stop_words=stop_words  # 可考虑添加中文停用词
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"TF-IDF shape: {X_train_tfidf.shape}")

# ------------------------------










# 4. 训练随机森林模型
# ------------------------------

# 4.1 方面检测（多标签）：8个二分类任务
rf_aspect = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
)
rf_aspect.fit(X_train_tfidf, y1_train)

# 4.2 方面情感（多标签）：8个二分类任务
rf_sentiment = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
)
rf_sentiment.fit(X_train_tfidf, y2_train)

# 4.3 总体评分（多分类）：4个类别
rf_rating = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf_rating.fit(X_train_tfidf, y3_train)

# ------------------------------
# 5. 预测与评估
# ------------------------------
y1_pred = rf_aspect.predict(X_test_tfidf)
y2_pred = rf_sentiment.predict(X_test_tfidf)
y3_pred = rf_rating.predict(X_test_tfidf)

# 评估指标
f1_aspect = f1_score(y1_test, y1_pred, average='micro')
f1_sentiment = f1_score(y2_test, y2_pred, average='micro')
acc_rating = accuracy_score(y3_test, y3_pred)

print("=== 随机森林基线结果 ===")
print(f"方面检测 (label1) Micro-F1: {f1_aspect:.4f}")
print(f"大评价维度多标签分类 (label2) Micro-F1: {f1_sentiment:.4f}")
print(f"舆情风险 4 分类 (label3) 准确率: {acc_rating:.4f}")

 # ------------------------------
# 6. 保存模型和向量化器
# ----------------
model = [rf_aspect, rf_sentiment, rf_rating]
print("保存模型和向量化器...")
with open(conf.rf_model_save_path + '/rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open(conf.rf_model_save_path + '/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("模型和向量化器，保存成功！")

