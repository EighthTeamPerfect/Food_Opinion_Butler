import pandas as pd
import re
import jieba
import torch
from sklearn.model_selection import train_test_split

# 维度关键词规则
# ASPECT_RULES = {
#     0: {"pos": ["好吃","美味","入味","新鲜","赞"], "neg": ["难吃","太咸","发苦","腥","油腻"]},
#     1: {"pos": ["新鲜","嫩","活的"], "neg": ["不新鲜","变质","臭","发酸","放久"]},
#     2: {"pos": ["干净","宽敞","安静"], "neg": ["脏","苍蝇","拥挤","油烟大"]},
#     3: {"pos": ["热情","周到","及时"], "neg": ["态度差","不理人","凶","欺骗"]},
#     4: {"pos": ["上菜快","不用等"], "neg": ["等很久","上菜慢","催单没用"]},
#     5: {"pos": ["划算","实惠","量大"], "neg": ["太贵","不值","分量少"]},
#     6: {"pos": ["不漏","包装完好"], "neg": ["洒了","破","缺餐具"]},
#     7: {"pos": ["卫生安全"], "neg": ["头发","虫子","拉肚子","食物中毒","异味"]}
# }
ASPECT_RULES = {
    0: {"name": "菜品口味", "pos": ["好吃","美味","入味","新鲜","赞"], "neg": ["难吃","太咸","发苦","腥","油腻"]},
    1: {"name": "食材新鲜度", "pos": ["新鲜","嫩","活的"], "neg": ["不新鲜","变质","臭","发酸","放久"]},
    2: {"name": "门店环境", "pos": ["干净","宽敞","安静"], "neg": ["脏","苍蝇","拥挤","油烟大"]},
    3: {"name": "服务态度", "pos": ["热情","周到","及时"], "neg": ["态度差","不理人","凶","欺骗"]},
    4: {"name": "上菜速度", "pos": ["上菜快","不用等"], "neg": ["等很久","上菜慢","催单没用"]},
    5: {"name": "性价比价格", "pos": ["划算","实惠","量大"], "neg": ["太贵","不值","分量少"]},
    6: {"name": "外卖包装", "pos": ["不漏","包装完好"], "neg": ["洒了","破","缺餐具"]},
    7: {"name": "食品安全", "pos": ["卫生安全"], "neg": ["头发","虫子","拉肚子","食物中毒","异味"]}
}

# 风险等级文字映射
RISK_RULES = {
    0: "无风险好评",
    1: "普通轻度吐槽",
    2: "中度差评投诉",
    3: "高危食品安全舆情"
}

# ====================== 全局统一配置区（所有路径、超参在这里修改） ======================
# 1. 数据列名配置（匹配你的csv字段）
TEXT_COL = "text"
ASPECT_COL = "dim_labels"
SENTI_COL = "sentiment_labels"
RISK_COL = "risk_label"

# 2. 模型/数据超参
MAX_TEXT_LEN = 256    # 过滤超过该长度文本
ASPECT_NUM = 8        # 评价维度总数
RISK_NUM = 4          # 风险等级类别数
TEST_SPLIT_RATIO = 0.2  # 测试集划分比例
RANDOM_SEED = 42        # 随机种子固定复现

# 3. 文件路径统一配置（只需改这里切换数据集）
DATA_FOLDER = "../data/"  # 数据根目录
# 主标注数据集
MAIN_CSV = DATA_FOLDER + "合并标注_SOP格式_长度_500.csv"
# 种子标注数据集
SEED_CSV = DATA_FOLDER + "seed_label03.csv"
# FastText训练文件输出路径
FASTTEXT_TRAIN_TXT = DATA_FOLDER + "fasttext_train.txt"
# ==================================================================================

def safe_read_csv(file_path):
    """兼容utf-8-sig/gbk/gb2312，解决csv乱码问题"""
    encodings = ["utf-8-sig", "gbk", "gb2312"]
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"读取成功 | 文件：{file_path} | 编码：{enc}")
            return df
        except UnicodeDecodeError:
            continue
    raise Exception(f"文件{file_path}无法以utf-8/gbk/gb2312读取，请检查文件")

def clean_text(text):
    """文本清洗：去除换行、特殊脏符号、多余空格"""
    text = str(text)
    # 替换换行制表符
    text = re.sub(r'[\n\r\t]', ' ', text)
    # 仅保留中文、英文、数字、常用标点
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。？！：；""''()（）、]', '', text)
    # 多个空格合并为单个
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def aspect_str2tensor(aspect_str):
    """维度标签字符串 0,1,3 → torch 8维0/1张量"""
    tensor = torch.zeros(ASPECT_NUM, dtype=torch.float32)
    if pd.isna(aspect_str) or str(aspect_str).strip() == "":
        return tensor
    for idx_str in str(aspect_str).split(","):
        try:
            idx = int(idx_str.strip())
            if 0 <= idx < ASPECT_NUM:
                tensor[idx] = 1.0
        except ValueError:
            continue
    return tensor

def senti_str2tensor(senti_str):
    """情感标签字符串 1,0,1... → torch 8维张量"""
    tensor = torch.zeros(ASPECT_NUM, dtype=torch.float32)
    if pd.isna(senti_str) or str(senti_str).strip() == "":
        return tensor
    arr = str(senti_str).split(",")
    valid_len = min(len(arr), ASPECT_NUM)
    for i in range(valid_len):
        try:
            tensor[i] = float(arr[i].strip())
        except ValueError:
            continue
    return tensor

def tensor2aspect_str(tensor):
    """张量转回逗号分隔维度字符串"""
    indices = torch.where(tensor > 0.5)[0].cpu().numpy()
    return ",".join([str(i) for i in indices]) if len(indices) > 0 else ""

def tensor2senti_str(tensor):
    """张量转回逗号分隔情感字符串"""
    arr = (tensor > 0.5).int().cpu().numpy()
    return ",".join([str(i) for i in arr])

def tokenize_text(text):
    """jieba分词，返回空格分隔字符串，供FastText/RF使用"""
    clean_txt = clean_text(text)
    seg_list = jieba.lcut(clean_txt)
    return " ".join(seg_list)

def load_and_preprocess_data(file_path=MAIN_CSV):
    """
    统一数据加载&预处理入口
    :param file_path: 传入csv路径，默认使用MAIN_CSV全局配置
    :return: train_df, test_df 训练集、测试集
    """
    # 读取csv
    df = safe_read_csv(file_path)
    # 文本清洗
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
    # 过滤超长文本
    df = df[df[TEXT_COL].astype(str).apply(len) <= MAX_TEXT_LEN].reset_index(drop=True)
    # 标签转张量
    df["aspect_tensor"] = df[ASPECT_COL].apply(aspect_str2tensor)
    df["senti_tensor"] = df[SENTI_COL].apply(senti_str2tensor)
    df[RISK_COL] = df[RISK_COL].astype(int)
    # 分层划分训练/测试集
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=df[RISK_COL]
    )
    print(f"\n数据集处理完成：")
    print(f"原始总样本：{len(df)}")
    print(f"训练集：{len(train_df)} | 测试集：{len(test_df)}")
    return train_df, test_df

# 测试入口
if __name__ == "__main__":
    train_data, test_data = load_and_preprocess_data()
    print("\n训练集前5行预览：")
    print(train_data[[TEXT_COL, ASPECT_COL, SENTI_COL, RISK_COL]].head())