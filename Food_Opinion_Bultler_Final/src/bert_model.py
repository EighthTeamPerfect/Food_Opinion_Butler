import torch
import torch.nn as nn
# 新增
from torch.optim import AdamW
# 分层导入transformers组件
from transformers import BertTokenizer, BertModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from data_preprocess_addpath import load_and_preprocess_data, TEXT_COL, ASPECT_NUM, RISK_NUM, MAX_TEXT_LEN

# 全局配置
PRETRAIN_MODEL_PATH = "../models/bert-base-chinese"  # 轻量CPU友好模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 5e-5
MODEL_SAVE_PATH = "../models/bert_multi_task/"

# 多任务BERT模型
class MultiTaskBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(PRETRAIN_MODEL_PATH, ignore_mismatched_sizes=True)
        self.dropout = nn.Dropout(0.1)
        hidden_dim = self.bert.config.hidden_size
        # 三任务输出头
        self.aspect_head = nn.Linear(hidden_dim, ASPECT_NUM)  # 8维度多标签分类
        self.senti_head = nn.Linear(hidden_dim, ASPECT_NUM)   # 8维度情感极性分类
        self.risk_head = nn.Linear(hidden_dim, RISK_NUM)       # 4级风险分类

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = self.dropout(out.pooler_output)
        logit_aspect = self.aspect_head(cls_out)
        logit_senti = self.senti_head(cls_out)
        logit_risk = self.risk_head(cls_out)
        return logit_aspect, logit_senti, logit_risk

# 数据集类
class CommentDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[TEXT_COL])
        encode = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=MAX_TEXT_LEN, return_tensors="pt"
        )
        input_ids = encode["input_ids"][0]
        mask = encode["attention_mask"][0]
        asp_t = row['aspect_tensor']
        sen_t = row['senti_tensor']
        risk_t = torch.tensor(row['risk_label'], dtype=torch.long)
        return input_ids, mask, asp_t, sen_t, risk_t

# 训练函数
def train_bert_model(train_df, test_df):
    # 初始化
    tokenizer = BertTokenizer.from_pretrained(PRETRAIN_MODEL_PATH)
    model = MultiTaskBert().to(DEVICE)
    # 损失函数
    loss_aspect = nn.BCEWithLogitsLoss()
    loss_senti = nn.BCEWithLogitsLoss()
    loss_risk = nn.CrossEntropyLoss()
    # 优化器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    # 学习率调度
    total_steps = len(train_df) // BATCH_SIZE * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # 数据加载
    train_ds = CommentDataset(train_df, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_ds = CommentDataset(test_df, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} 训练"):
            input_ids, mask, asp_t, sen_t, risk_t = batch
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            asp_t = asp_t.to(DEVICE)
            sen_t = sen_t.to(DEVICE)
            risk_t = risk_t.to(DEVICE)

            optimizer.zero_grad()
            la, ls, lr = model(input_ids, mask)
            l1 = loss_aspect(la, asp_t)
            l2 = loss_senti(ls, sen_t)
            l3 = loss_risk(lr, risk_t)
            loss = l1 + l2 + l3
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} 验证"):
                input_ids, mask, asp_t, sen_t, risk_t = batch
                input_ids = input_ids.to(DEVICE)
                mask = mask.to(DEVICE)
                asp_t = asp_t.to(DEVICE)
                sen_t = sen_t.to(DEVICE)
                risk_t = risk_t.to(DEVICE)
                la, ls, lr = model(input_ids, mask)
                l1 = loss_aspect(la, asp_t)
                l2 = loss_senti(ls, sen_t)
                l3 = loss_risk(lr, risk_t)
                total_val_loss += (l1 + l2 + l3).item()
        avg_val_loss = total_val_loss / len(test_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}")

    # 保存模型
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"BERT多任务模型已保存至: {MODEL_SAVE_PATH}")
    return model, tokenizer

# 预测函数
def predict_bert(text, model, tokenizer):
    """单文本预测，返回维度标签、情感标签、风险等级、置信度"""
    model.eval()
    with torch.no_grad():
        encode = tokenizer(
            text, truncation=True, padding="max_length",
            max_length=MAX_TEXT_LEN, return_tensors="pt"
        )
        input_ids = encode["input_ids"].to(DEVICE)
        mask = encode["attention_mask"].to(DEVICE)
        la, ls, lr = model(input_ids, mask)
        # 概率计算
        prob_asp = torch.sigmoid(la)
        prob_sen = torch.sigmoid(ls)
        prob_risk = torch.softmax(lr, dim=-1)
        # 预测结果
        pred_asp = (prob_asp > 0.5).float()
        pred_sen = (prob_sen > 0.5).float()
        pred_risk = torch.argmax(prob_risk, dim=-1)
        # 置信度
        conf = prob_asp.max(dim=1)[0] * prob_sen.max(dim=1)[0] * prob_risk.max(dim=1)[0]
        # 转标签字符串
        aspect_str = ",".join([str(i) for i in torch.where(pred_asp[0] > 0.5)[0].cpu().numpy()])
        senti_str = ",".join([str(i) for i in (pred_sen[0] > 0.5).int().cpu().numpy()])
        return {
            "aspect_label_id": aspect_str,
            "senti_label": senti_str,
            "risk_label_id": int(pred_risk[0]),
            "confidence": float(conf[0])
        }

# 主训练入口
if __name__ == "__main__":
    train_df, test_df = load_and_preprocess_data("../data/合并标注_SOP格式_长度_50.csv")
    train_bert_model(train_df, test_df)