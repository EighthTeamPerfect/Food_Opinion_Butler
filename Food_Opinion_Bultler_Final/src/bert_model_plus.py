import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm
from data_preprocess_addpath import load_and_preprocess_data, TEXT_COL, ASPECT_NUM, RISK_NUM, MAX_TEXT_LEN

# ===================== 全局统一超参配置（集中修改，无需散落在代码） =====================
# 预训练模型路径
PRETRAIN_MODEL_PATH = "../models/bert-base-chinese"
# 模型保存路径，自动创建文件夹
MODEL_SAVE_PATH = "../models/bert_multi_task/"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
WEIGHT_SAVE_FILE = os.path.join(MODEL_SAVE_PATH, "pytorch_model.bin")

# 设备自动识别
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 训练超参
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0       # 梯度裁剪防止爆炸
GRAD_ACCUM_STEPS = 2       # 梯度累积，小显存模拟大batch
LOSS_WEIGHT_ASPECT = 1.0   # 多任务损失权重平衡
LOSS_WEIGHT_SENTI = 1.0
LOSS_WEIGHT_RISK = 1.2
# DataLoader配置（Windows兼容）
WORKER_NUM = 0 if os.name == "nt" else 4
PIN_MEM = True if torch.cuda.is_available() else False
# ======================================================================================

# 多任务BERT网络（结构不变，兼容原有推理代码）
class MultiTaskBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(PRETRAIN_MODEL_PATH, ignore_mismatched_sizes=True)
        self.dropout = nn.Dropout(0.1)
        hidden_dim = self.bert.config.hidden_size
        # 三任务输出头
        self.aspect_head = nn.Linear(hidden_dim, ASPECT_NUM)
        self.senti_head = nn.Linear(hidden_dim, ASPECT_NUM)
        self.risk_head = nn.Linear(hidden_dim, RISK_NUM)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = self.dropout(out.pooler_output)
        logit_aspect = self.aspect_head(cls_out)
        logit_senti = self.senti_head(cls_out)
        logit_risk = self.risk_head(cls_out)
        # 同时返回三个任务输出
        return logit_aspect, logit_senti, logit_risk

# 数据集类（增加张量类型强校验，防止loss报错）
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
        # 强制转为float张量，避免BCE loss类型报错
        asp_t = torch.tensor(row["aspect_tensor"], dtype=torch.float32)
        sen_t = torch.tensor(row["senti_tensor"], dtype=torch.float32)
        risk_t = torch.tensor(row["risk_label"], dtype=torch.long)
        return input_ids, mask, asp_t, sen_t, risk_t

# 分层优化器分组：bert bias/layernorm不做weight decay
def get_optim_groups(model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_groups = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    return optimizer_groups

# 训练主函数（混合精度+梯度累积+损失加权+梯度裁剪）
def train_bert_model(train_df, test_df):
    # 初始化分词器与模型
    tokenizer = BertTokenizer.from_pretrained(PRETRAIN_MODEL_PATH)
    model = MultiTaskBert().to(DEVICE)
    # 混合精度缩放器（GPU加速必备）
    scaler = GradScaler(enabled=torch.cuda.is_available())

    # 损失函数
    loss_aspect = nn.BCEWithLogitsLoss()
    loss_senti = nn.BCEWithLogitsLoss()
    loss_risk = nn.CrossEntropyLoss()

    # 分层AdamW优化器
    opt_groups = get_optim_groups(model)
    optimizer = AdamW(opt_groups, lr=LEARNING_RATE)

    # 学习率调度
    # 计算总训练步数
    total_train_steps = (len(train_df) // BATCH_SIZE) // GRAD_ACCUM_STEPS * EPOCHS
    # 学习率调度器
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_train_steps * 0.1),
        num_training_steps=total_train_steps
    )

    # 构建DataLoader（提速配置）
    train_ds = CommentDataset(train_df, tokenizer)
    test_ds = CommentDataset(test_df, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKER_NUM, pin_memory=PIN_MEM)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKER_NUM, pin_memory=PIN_MEM)

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        step_count = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} 训练")
        for batch in train_pbar:
            input_ids, mask, asp_t, sen_t, risk_t = batch
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            asp_t = asp_t.to(DEVICE)
            sen_t = sen_t.to(DEVICE)
            risk_t = risk_t.to(DEVICE)

            # 混合精度前向传播
            with autocast(enabled=torch.cuda.is_available()):
                la, ls, lr = model(input_ids, mask)
                l1 = loss_aspect(la, asp_t) * LOSS_WEIGHT_ASPECT
                l2 = loss_senti(ls, sen_t) * LOSS_WEIGHT_SENTI
                l3 = loss_risk(lr, risk_t) * LOSS_WEIGHT_RISK
                batch_loss = l1 + l2
                # 梯度累积：loss除以累积步数
                batch_loss = batch_loss / GRAD_ACCUM_STEPS

            # 反向传播缩放梯度
            scaler.scale(batch_loss).backward()
            step_count += 1

            # 每累积N步更新参数
            if step_count % GRAD_ACCUM_STEPS == 0:
                # 梯度裁剪防止爆炸NaN
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_train_loss += batch_loss.item() * GRAD_ACCUM_STEPS
            train_pbar.set_postfix({"batch_loss": f"{batch_loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段（关闭梯度，节省显存）
        model.eval()
        total_val_loss = 0.0
        val_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{EPOCHS} 验证")
        with torch.no_grad():
            for batch in val_pbar:
                input_ids, mask, asp_t, sen_t, risk_t = batch
                input_ids = input_ids.to(DEVICE)
                mask = mask.to(DEVICE)
                asp_t = asp_t.to(DEVICE)
                sen_t = sen_t.to(DEVICE)
                risk_t = risk_t.to(DEVICE)
                la, ls, lr = model(input_ids, mask)
                l1 = loss_aspect(la, asp_t) * LOSS_WEIGHT_ASPECT
                l2 = loss_senti(ls, sen_t) * LOSS_WEIGHT_SENTI
                l3 = loss_risk(lr, risk_t) * LOSS_WEIGHT_RISK
                val_loss = l1 + l2 + l3
                total_val_loss += val_loss.item()
                val_pbar.set_postfix({"val_loss": f"{val_loss.item():.4f}"})
        avg_val_loss = total_val_loss / len(test_loader)
        print(f"\n==== Epoch {epoch+1} 训练总结 ====")
        print(f"训练平均损失: {avg_train_loss:.4f} | 验证平均损失: {avg_val_loss:.4f}")

    # 【修复原save_pretrained报错】标准保存方式，完美匹配inference加载逻辑
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    torch.save(model.state_dict(), WEIGHT_SAVE_FILE)
    print(f"\n✅ 模型训练完成，权重保存路径: {WEIGHT_SAVE_FILE}")
    return model, tokenizer

# 推理预测函数（无改动，保持原有输出格式兼容流水线）
def predict_bert(text, model, tokenizer):
    model.eval()
    with torch.no_grad():
        encode = tokenizer(
            text, truncation=True, padding="max_length",
            max_length=MAX_TEXT_LEN, return_tensors="pt"
        )
        input_ids = encode["input_ids"].to(DEVICE)
        mask = encode["attention_mask"].to(DEVICE)
        la, ls, lr = model(input_ids, mask)
        prob_asp = torch.sigmoid(la)
        prob_sen = torch.sigmoid(ls)
        prob_risk = torch.softmax(lr, dim=-1)
        pred_asp = (prob_asp > 0.5).float()
        pred_sen = (prob_sen > 0.5).float()
        pred_risk = torch.argmax(prob_risk, dim=-1)
        conf = prob_asp.max(dim=1)[0] * prob_sen.max(dim=1)[0] * prob_risk.max(dim=1)[0]
        aspect_str = ",".join([str(i) for i in torch.where(pred_asp[0] > 0.5)[0].cpu().numpy()])
        senti_str = ",".join([str(i) for i in (pred_sen[0] > 0.5).int().cpu().numpy()])
        return {
            "aspect_label_id": aspect_str,
            "senti_label": senti_str,
            "risk_label_id": int(pred_risk[0]),
            "confidence": float(conf[0])
        }

# 程序入口
if __name__ == "__main__":
    train_df, test_df = load_and_preprocess_data("../data/合并标注_SOP格式_长度_50.csv")
    train_bert_model(train_df, test_df)