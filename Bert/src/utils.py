# 文件名：utils.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import time
from datetime import timedelta
from config import Config
import time
from torch.nn.utils.rnn import pad_sequence
conf=Config()
def get_time_diff(start_time):
    end_time = time.time()
    # 计算时间差（秒），转换为毫秒（1秒 = 1000毫秒）
    return (end_time - start_time) * 1000

def load_raw_data(file_path):
    """
    读取原始数据文件，解析为文本和标签。

    参数：
        file_path (str): 数据文件路径（如dev2.txt）。

    返回：
        List[Tuple[str, int]]: 包含(文本, 标签)的列表。
    """
    data = []
    with open(file_path, "r", encoding="UTF-8") as f:
        for line in tqdm(f, desc="Loading data"):

            line = line.strip()
            #查看是否为空
            if not line:
                continue
            parts = line.split("\t")
            text, aspect_label_id, senti_label, risk_label_id = parts[0], parts[1], parts[2], parts[3]
            data.append((text,aspect_label_id, senti_label, risk_label_id))
    print(data[:5])
    return data


class TextDataset(Dataset):
    """
    自定义TextDataset，存储原始文本和标签，用于BERT分类任务。
    """

    def __init__(self, data):
        """
        参数：
            data (List[Tuple[str, int]]): 原始数据，包含(文本, 标签)的列表。
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        参数：
            idx (int): 样本索引。
        """
        text=self.data[idx][0]
        aspect_label_id=self.data[idx][1]
        senti_label = self.data[idx][2]
        risk_label_id = self.data[idx][3]
        return text,aspect_label_id, senti_label, risk_label_id
def collate_fn(batch):
    # 提取文本和标签
    texts = [item[0] for item in batch]
    aspect_label_str = [item[1] for item in batch]
    senti_label_str = [item[2] for item in batch]
    risk_label_str = [item[3] for item in batch]

    # BERT分词编码
    text_tokens = conf.tokenizer(texts, padding=True, truncation=True, max_length=conf.pad_size)
    input_ids = torch.tensor(text_tokens["input_ids"])
    attention_mask = torch.tensor(text_tokens["attention_mask"])

    # ----------------------标签处理（修复(空)报错 + 变长标签padding）----------------------
    # 1. 处理aspect多标签
    aspect_label_list = []
    for s in aspect_label_str:
        # 判断标签是否为空标识
        if s.strip() == "(空)":
            nums = []
        else:
            # 过滤空白字符、转数字
            nums = [int(x.strip()) for x in s.split(",") if x.strip()]
        aspect_label_list.append(torch.tensor(nums, dtype=torch.long))
    # 填充0补齐长度
    aspect_label_ids = pad_sequence(aspect_label_list, batch_first=True, padding_value=0)

    # 2. 处理senti多标签
    senti_label_list = []
    for s in senti_label_str:
        if s.strip() == "(空)":
            nums = []
        else:
            nums = [int(x.strip()) for x in s.split(",") if x.strip()]
        senti_label_list.append(torch.tensor(nums, dtype=torch.long))
    senti_label = pad_sequence(senti_label_list, batch_first=True, padding_value=0)

    # 3. risk单分类标签（如果risk也有(空)，同步加判断）
    risk_label_list = []
    for s in risk_label_str:
        if s.strip() == "(空)":
            risk_label_list.append(0)  # 空标签默认0，按需修改
        else:
            risk_label_list.append(int(s.strip()))
    risk_label_id = torch.tensor(risk_label_list, dtype=torch.long)

    return input_ids, attention_mask, aspect_label_ids, senti_label, risk_label_id


# 示例用法

# 构建数据的Dataloader
def build_dataloader():
    # 1.数据格式调整   text \t label ===> [（text,label）]
    train_data = load_raw_data(conf.train_path)
    test_data = load_raw_data(conf.test_path)
    dev_data = load_raw_data(conf.dev_path)

    # 2.构建数据集对象 TextDataset   __getitem__
    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)
    dev_dataset = TextDataset(dev_data)

    # 3.Dataloader构建
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=conf.batch_size,collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=conf.batch_size,collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=conf.batch_size,collate_fn=collate_fn)

    return train_loader,test_loader,dev_loader


if __name__ == "__main__":
    # 记录开始时间
    # start_time = time.time()
    # print(load_raw_data(conf.train_path)[:5])
    # # # 构建 DataLoader
    # train_dataloader,test_dataloader,dev_dataloader = build_dataloader()
    # # print(f'训练集数量：{len(train_dataloader.dataset)}')
    # # print(f'测试集数量：{len(test_dataloader.dataset)}')
    # # print(f'验证集数量：{len(dev_dataloader.dataset)}')
    #
    # # # #遍历 DataLoader
    # for batch in train_dataloader:
    #     input_ids, attention_mask, labels = batch
    #     print("input_ids=>",input_ids.tolist())
    #     print("labels=>",labels.tolist())
    #     print("attention_mask=>",attention_mask.tolist())
    #     breakpoint()
    #     # print("Input IDs:", input_ids.shape)
    #     # print("Attention Mask:", attention_mask.shape)
    #     # print("Labels:", labels.shape)


    # train_dataset = TextDataset(data)
    # print(train_dataset)
    data = load_raw_data(conf.train_path)
    train_dataset = TextDataset(data)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=conf.batch_size,collate_fn=collate_fn)
    for batch in train_loader:
        input_ids, attention_mask, aspect_label_id, senti_label, risk_label_id = batch
        print("input_ids=>",input_ids.tolist())
        print("labels=>",aspect_label_id.tolist())
        print("attention_mask=>",attention_mask.tolist())
        print("senti_label=>",senti_label.tolist())
        print("risk_label_id=>",risk_label_id.tolist())


