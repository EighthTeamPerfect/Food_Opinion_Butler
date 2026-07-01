import fasttext
import pandas as pd
from data_preprocess_addpath import load_and_preprocess_data, tokenize_text, TEXT_COL, ASPECT_NUM, RISK_NUM

# 全局配置
MODEL_SAVE_PATH = "../models/fasttext_model.bin"
TRAIN_TXT_PATH = "../data/fasttext_train.txt"


# 生成FastText训练数据
def generate_fasttext_data(df, save_path):
    """生成FastText要求的标签格式数据：__label__xxx 文本内容"""
    lines = []
    for _, row in df.iterrows():
        # 维度标签
        aspect_labels = [f"__label__aspect_{i}" for i in range(ASPECT_NUM) if row['aspect_tensor'][i] == 1]
        # 风险标签
        risk_label = f"__label__risk_{row['risk_label']}"
        # 拼接行
        line = f"{' '.join(aspect_labels)} {risk_label} {tokenize_text(row[TEXT_COL])}\n"
        lines.append(line)
    # 保存文件
    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"FastText训练数据已生成：{save_path}")


# 训练FastText模型
def train_fasttext_model(train_df, test_df):
    # 生成训练数据
    generate_fasttext_data(train_df, TRAIN_TXT_PATH)
    # 训练模型
    model = fasttext.train_supervised(
        input=TRAIN_TXT_PATH,
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        bucket=200000,
        dim=100,
        loss="ova"  # 多标签分类用ova
    )
    # 模型评估
    test_texts = [tokenize_text(text) for text in test_df[TEXT_COL]]
    test_labels = []
    for _, row in test_df.iterrows():
        labels = [f"aspect_{i}" for i in range(ASPECT_NUM) if row['aspect_tensor'][i] == 1]
        labels.append(f"risk_{row['risk_label']}")
        test_labels.append(labels)

    # 预测
    pred_labels = []
    for text in test_texts:
        pred = model.predict(text, k=10)
        pred_labels.append([label.replace("__label__", "") for label in pred[0]])

    # 计算F1
    f1 = 0
    for true, pred in zip(test_labels, pred_labels):
        tp = len(set(true) & set(pred))
        fp = len(set(pred) - set(true))
        fn = len(set(true) - set(pred))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 += 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f1 /= len(test_labels)
    print(f"FastText模型F1分数: {f1:.4f}")

    # 保存模型
    model.save_model(MODEL_SAVE_PATH)
    print(f"FastText模型已保存至: {MODEL_SAVE_PATH}")
    return model


# 预测函数
def predict_fasttext(text, model):
    """单文本预测，返回维度标签、风险等级"""
    tokenized_text = tokenize_text(text)
    pred = model.predict(tokenized_text, k=10)
    labels = [label.replace("__label__", "") for label in pred[0]]
    # 解析维度标签
    aspect_list = [label.split("_")[1] for label in labels if label.startswith("aspect_")]
    aspect_str = ",".join(aspect_list)
    # 解析风险标签
    risk_id = [int(label.split("_")[1]) for label in labels if label.startswith("risk_")][0]
    return {
        "aspect_label_id": aspect_str,
        "risk_label_id": risk_id
    }


# 主训练入口
if __name__ == "__main__":
    train_df, test_df = load_and_preprocess_data("../data/合并标注_SOP格式_长度_500.csv")
    train_fasttext_model(train_df, test_df)