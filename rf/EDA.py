import re

import jieba
import pandas as pd
from config import Config

conf = Config()
current_path = conf.train_datapath
# 读取整个文件
with open(current_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 正则匹配：评论 + 三个标签（标签1、标签2可为多个数字，标签3为单个数字）
pattern = r'(.+?)\t([\d,\s]+)\t([\d,\s]+)\t(\d+)'
records = re.findall(pattern, content, re.DOTALL)

# 构建DataFrame
data = []
for rec in records:
    text = rec[0].strip()  # 评论
    # 去除多余换行、空白
    text = re.sub(r'\s+', ' ', text)  # 合并空白
    # 标签处理
    label1 = [int(x.strip()) for x in rec[1].split(',') if x.strip()]
    label2 = [int(x.strip()) for x in rec[2].split(',') if x.strip()]
    label3 = [int(x.strip()) for x in rec[3].split(',') if x.strip()]
    # label3 = int(rec[3].strip())
    data.append([text, label1, label2, label3])

df = pd.DataFrame(data, columns=['text', 'label1', 'label2', 'label3'])


def cut_sentence(s):
    """对输入文本进行结巴分词，并限制前30个词"""
    return ' '.join(list(jieba.cut(s)))  # 分词后取前30个词并a用空格连接


df['words'] = df['text'].apply(cut_sentence)  # 对每行文本进行分词并存储到words列
print("*" * 60)
df = df[['words', 'label1', 'label2', 'label3']]
print(df.head())

if "train" in current_path:
    df.to_csv(conf.process_train_datapath, index=False)  # 将处理后的数据保存到CSV文件
    print(f"train数据已经处理完成，已经成功保存至：{conf.process_train_datapath}")
elif "test" in current_path:
    df.to_csv(conf.process_test_datapath, index=False)  # 将处理后的数据保存到CSV文件
    print(f"test数据已经处理完成，已经成功保存至：{conf.process_test_datapath}")
elif "dev" in current_path:
    df.to_csv(conf.process_dev_datapath, index=False)  # 将处理后的数据保存到CSV文件
    print(f"dev数据已经处理完成，已经成功保存至：{conf.process_dev_datapath}")
