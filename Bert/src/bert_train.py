"""
BERT 多任务训练器 — 仅使用指数学习率衰减
Author: 郎朗
"""
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from sklearn.metrics import f1_score
from config import Config
from bert_model import MuliTaskBertClassfier
from utils import build_dataloader

conf = Config()


def convert_aspect_to_multihot(indices_tensor, num_dims=8):
    """稀疏索引 [0,5] → 独热向量 [1,0,0,0,0,1,0,0]"""
    batch_size = indices_tensor.size(0)
    multihot = torch.zeros(batch_size, num_dims)
    for i in range(batch_size):
        for idx in indices_tensor[i]:
            idx = idx.item()
            if idx != 0 or (indices_tensor[i] == 0).all():
                if 0 <= idx < num_dims:
                    multihot[i, idx] = 1.0
    # 非 zero-padding 的位置
    mask = indices_tensor != 0
    for i in range(batch_size):
        for j in range(indices_tensor.size(1)):
            if mask[i, j]:
                idx = indices_tensor[i, j].item()
                if 0 <= idx < num_dims:
                    multihot[i, idx] = 1.0
    return multihot


def convert_senti_to_multihot(indices_tensor, num_dims=8):
    """
    将情感标签转换为多独热向量（float 格式，供 BCE 损失使用）。

    背景：
        与 aspect 不同，senti_label 本身就是全向量格式，例如 "0,1,1,1,1,1,1,1"。
        utils.py 的 collate_fn 按逗号 split 后直接得到 8 个 int 值 [0,1,1,1,1,1,1,1]。
        本函数只需将其转为 float tensor 即可（BCE 需要 float 标签）。

    参数：
        indices_tensor (Tensor): shape [batch, 8]，每行是 8 个 0/1 值
        num_dims (int): 期望的向量长度，默认 8

    返回：
        Tensor: shape [batch, 8]，float 类型的多独热向量

    注意：
        如果列数与 num_dims 不匹配（异常情况），退化为全 1 向量作为兜底。
    """
    batch_size = indices_tensor.size(0)
    vec_len = indices_tensor.size(1)
    if vec_len == num_dims:
        # 8 列 → 直接就是全向量，只需转 float 类型（BCE 损失要求 float）
        return indices_tensor.float()
    else:
        # 列数不对（数据异常），退化为全 1 向量，避免训练崩溃
        return torch.ones(batch_size, num_dims)


class Trainer:
    """
    多任务 BERT 训练器 — 两阶段训练策略 + 指数学习率衰减。

    两阶段策略说明：
        Stage 1（冻结BERT）：只训练三个分类头，用较大学习率快速收敛，避免随机初始化的头拖累 BERT。
        Stage 2（解冻BERT）  ：解冻全部参数，使用分层学习率（BERT底层小、顶层大、分类头最大），
                              让 BERT 逐步适应下游任务，同时防止预训练知识被遗忘。
    """

    def __init__(self):
        # 训练设备（GPU 或 CPU）
        self.device = conf.device
        # 初始化多任务 BERT 模型并移至目标设备
        self.model = MuliTaskBertClassfier().to(self.device)

        # 构建训练集、测试集、验证集的 DataLoader
        self.train_loader, self.test_loader, self.dev_loader = build_dataloader()

        # 记录历史最优验证损失和对应 epoch（用于早停/保存最优模型）
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    # ============================================================
    # 阶段一: 冻结 BERT，仅训练 3 个分类头
    #
    # 目的：分类头是随机初始化的，如果直接和 BERT 一起训练，
    #       会有大的梯度反传到 BERT 导致预训练知识被破坏。
    #       因此先冻结 BERT，用较大学习率（1e-3）单独训练分类头至基本收敛。
    # ============================================================
    def train_stage1(self, epochs=3, lr=1e-3, gamma=0.95):
        """
        Stage 1: 冻结 BERT 主体，仅训练三个分类头。

        参数：
            epochs (int): 阶段一训练轮数，默认 3
            lr (float): 学习率，默认 1e-3（比 Stage 2 大，因为只训练线性层，收敛快）
            gamma (float): ExponentialLR 衰减系数，lr_new = lr * gamma，默认 0.95
        """
        print("=" * 56)
        print("  Stage 1: Freeze BERT → Train Heads only")
        print(f"  lr={lr}  gamma={gamma}  epochs={epochs}")
        print("=" * 56)

        # 冻结 BERT 所有参数：不计算梯度、不更新权重
        for param in self.model.bert.parameters():
            param.requires_grad = False

        # 只优化三个分类头的参数（dim_head、sentiment_head、risk_head）
        optimizer = AdamW(
            list(self.model.dim_head.parameters())        # 维度分类头
            + list(self.model.sentiment_head.parameters())  # 情感极性头
            + list(self.model.risk_head.parameters()),       # 舆情风险头
            lr=lr,
        )
        # 指数衰减调度器：每个 epoch 后 lr = lr * gamma
        scheduler = ExponentialLR(optimizer, gamma=gamma)

        # 调用通用训练循环
        return self._run_loop(optimizer, scheduler, epochs, "Stage1")

    # ============================================================
    # 阶段二: 解冻全部，分层学习率
    #
    # 目的：Stage 1 中分类头已基本收敛，现在解冻 BERT 进行整体微调。
    #       使用「分层学习率」策略：BERT 底层（Embeddings）学习率最小，
    #       逐层递增到顶层（Encoder Layers），分类头学习率最大。
    #       这样底层通用语义改变少，高层任务特征调整多，避免灾难性遗忘。
    # ============================================================
    def train_stage2(self, epochs=5, base_lr=2e-5, head_lr=1e-4, gamma=0.95):
        """
        Stage 2: 解冻 BERT，使用分层学习率进行整体微调。

        参数：
            epochs (int): 阶段二训练轮数，默认 5
            base_lr (float): BERT 顶层的基础学习率，默认 2e-5
            head_lr (float): 分类头学习率，默认 1e-4（比 BERT 大，因为头仍需更多调整）
            gamma (float): ExponentialLR 衰减系数，默认 0.95
        """
        print("\n" + "=" * 56)
        print("  Stage 2: Unfreeze BERT → Layer-wise LR")
        print(f"  base_lr={base_lr}  head_lr={head_lr}  gamma={gamma}  epochs={epochs}")
        print("=" * 56)

        # 解冻 BERT 所有参数：恢复梯度计算和权重更新
        for param in self.model.bert.parameters():
            param.requires_grad = True

        # 获取 BERT 的 Transformer 层数（bert-base 为 12 层）
        num_layers = self.model.bert.config.num_hidden_layers  # 12

        param_groups = []
        # --- 分层学习率策略 ---
        # 1. Embeddings 层 → base_lr 的 30%（最底层，最接近通用语义，改动应最小）
        param_groups.append({
            "params": self.model.bert.embeddings.parameters(),
            "lr": base_lr * 0.30,
        })
        # 2. Encoder 12 层 → 从 base_lr 的 40% 线性递增到 95%
        #    底层（layer 0）→ 40%，顶层（layer 11）→ 95%
        for i, layer in enumerate(self.model.bert.encoder.layer):
            ratio = 0.40 + (0.55 * i / (num_layers - 1))
            param_groups.append({
                "params": layer.parameters(),
                "lr": base_lr * ratio,
            })
        # 3. Pooler 层 → base_lr（100%）
        param_groups.append({
            "params": self.model.bert.pooler.parameters(),
            "lr": base_lr,
        })
        # 4. 三个分类头 → head_lr（最大，因为需要适配下游任务）
        param_groups.append({
            "params": list(self.model.dim_head.parameters())
                    + list(self.model.sentiment_head.parameters())
                    + list(self.model.risk_head.parameters()),
            "lr": head_lr,
        })

        # 用所有参数组构建优化器（每组有不同学习率）
        optimizer = AdamW(param_groups)
        scheduler = ExponentialLR(optimizer, gamma=gamma)

        return self._run_loop(optimizer, scheduler, epochs, "Stage2")

    # ============================================================
    # 训练循环（Stage 1 和 Stage 2 共用）
    #
    # 每个 epoch 的流程：
    #   1. 遍历训练集 → 前向 → 计算损失 → 反向传播 → 更新权重
    #   2. 在验证集上评估 → 计算 val_loss 和 F1 指标
    #   3. 学习率衰减（ExponentialLR）
    #   4. 如果 val_loss 创新低 → 保存最优模型
    # ============================================================
    def _run_loop(self, optimizer, scheduler, epochs, tag):
        """
        通用训练循环，供 Stage 1 和 Stage 2 调用。

        参数：
            optimizer: 已配置好参数组的优化器
            scheduler: 学习率调度器
            epochs (int): 训练轮数
            tag (str): 阶段标签（"Stage1" / "Stage2"），用于日志输出

        返回：
            dict: 训练历史记录（train_loss, val_loss, 各任务 F1）
        """
        # 初始化历史记录字典，用于保存每个 epoch 的指标
        history = {"train_loss": [], "val_loss": [],
                   "dim_f1": [], "risk_f1": [], "sent_f1": []}

        for epoch in range(epochs):
            # 设置为训练模式（启用 Dropout 和 BatchNorm 更新）
            self.model.train()
            losses = []

            # 用 tqdm 显示训练进度条
            pbar = tqdm(self.train_loader, desc=f"{tag} E{epoch+1}/{epochs}")
            for batch in pbar:
                # 解包 batch：input_ids, attention_mask, aspect_ids, senti_ids, risk_ids
                input_ids, attention_mask, aspect_ids, senti_ids, risk_ids = batch
                # 将输入张量移至目标设备（GPU/CPU）
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                # --- 标签转换 ---
                # aspect 稀疏索引 → 多独热向量（如 [1,3] → [0,1,0,1,0,0,0,0]）
                dim_labels = convert_aspect_to_multihot(aspect_ids).to(self.device)
                # senti 全向量 → float tensor（如 [0,1,1,0,1,1,0,1]）
                senti_labels = convert_senti_to_multihot(senti_ids).to(self.device)
                # risk 单分类标签（如 2）
                risk_labels = risk_ids.to(self.device)

                # --- 前向传播 ---
                # model 返回 {"dimensions":..., "sentiments":..., "risk":...}
                outputs = self.model(input_ids, attention_mask)

                # --- 计算损失 ---
                # 三任务加权求和：dim_loss + sent_loss + risk_loss
                loss, loss_dict = self._compute_loss(
                    outputs, dim_labels, senti_labels, risk_labels
                )

                # --- 反向传播 ---
                optimizer.zero_grad()  # 清空上一步的残余梯度
                loss.backward()        # 反向传播，计算梯度
                # 梯度裁剪：防止梯度爆炸，将梯度范数限制在 1.0 以内
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()       # 更新参数

                # 记录当前 batch 的 loss
                losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.3f}")

            # --- Epoch 结束，计算平均训练损失 ---
            avg_train = sum(losses) / len(losses)
            history["train_loss"].append(avg_train)

            # --- 验证集评估 ---
            val_loss, metrics = self._evaluate()
            history["val_loss"].append(val_loss)
            history["dim_f1"].append(metrics["dim_f1"])
            history["risk_f1"].append(metrics["risk_f1"])
            history["sent_f1"].append(metrics["sent_f1"])

            # ★ 指数衰减：每个 epoch 结束后衰减一次学习率
            scheduler.step()
            # 读取当前学习率（取第一个参数组的学习率作为代表）
            current_lr = optimizer.param_groups[0]["lr"]

            # 打印本 epoch 的训练摘要
            print(f"  {tag} E{epoch+1} | "
                  f"train_loss={avg_train:.4f} val_loss={val_loss:.4f} | "
                  f"dim_f1={metrics['dim_f1']:.3f} "
                  f"sent_f1={metrics['sent_f1']:.3f} "
                  f"risk_f1={metrics['risk_f1']:.3f} | lr={current_lr:.2e}")

            # --- 保存最优模型 ---
            # 如果当前验证损失低于历史最优，则保存模型权重
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                torch.save(self.model.state_dict(), conf.model_save_path)
                print(f"  >> Best model saved")

        return history

    # ============================================================
    # 损失计算
    #
    # 三个任务头的损失：
    #   Head 1（维度分类）: BCE with logits，对 8 个维度做多标签二分类
    #   Head 2（情感极性）: 带 mask 的 BCE，只对「提及的维度」计算情感损失（未提及的不算）
    #   Head 3（舆情风险）: 交叉熵 CE，4 分类
    # 总损失 = dim_loss + sent_loss + risk_loss（等权求和）
    # ============================================================
    def _compute_loss(self, outputs, dim_labels, senti_labels, risk_labels):
        """
        计算三任务总损失。

        参数：
            outputs (dict): 模型前向输出，包含 "dimensions"、"sentiments"、"risk" 三个 logits
            dim_labels (Tensor): [batch, 8] 维度多独热标签
            senti_labels (Tensor): [batch, 8] 情感多独热标签
            risk_labels (Tensor): [batch] 风险等级标签（0~3）

        返回：
            total_loss (Tensor): 三任务损失之和（标量）
            loss_dict (dict): 各任务单独的损失值（float，用于日志）
        """
        # Head 1: 维度多标签 BCE（with logits）
        # 直接对 8 个维度独立做二分类，用 sigmoid + BCE
        dim_loss = nn.functional.binary_cross_entropy_with_logits(
            outputs["dimensions"], dim_labels
        )

        # Head 2: 情感极性 BCE（带 mask）
        # 关键：只在「文本提及的维度」上计算情感损失，未提及的维度不参与
        # 例如文本只提到了"口味"和"价格"，那么只在这 2 个维度上计算情感损失
        mask = dim_labels  # 用维度标签作为 mask（1=提及，0=未提及）
        # reduction="none" 先逐元素计算损失，不做聚合
        sent_loss_full = nn.functional.binary_cross_entropy_with_logits(
            outputs["sentiments"], senti_labels, reduction="none"
        )
        # 用 mask 加权求和后取均值（只保留提及维度的损失）
        sent_loss = (sent_loss_full * mask).sum() / (mask.sum() + 1e-8)

        # Head 3: 舆情风险 CE（4 分类交叉熵）
        risk_loss = nn.functional.cross_entropy(
            outputs["risk"], risk_labels
        )

        # 总损失：三任务等权相加
        total = dim_loss + sent_loss + risk_loss
        return total, {
            "dim": dim_loss.item(),
            "sent": sent_loss.item(),
            "risk": risk_loss.item(),
        }

    # ============================================================
    # 评估函数
    #
    # 在验证集（dev_loader）上计算：
    #   1. 总损失（用于早停和最优模型保存的判断依据）
    #   2. 三个任务的 macro-F1 指标（维度、情感、风险）
    # ============================================================
    def _evaluate(self):
        """
        在验证集上评估模型性能。

        返回：
            avg_loss (float): 验证集平均损失
            metrics (dict): 包含 dim_f1、sent_f1、risk_f1 三个指标
        """
        # 切换到评估模式（关闭 Dropout，BatchNorm 使用统计量）
        self.model.eval()
        total_loss = 0.0
        # 初始化各任务的预测值和真实值列表（后续合并后计算 F1）
        all_dim_preds, all_dim_trues = [], []
        all_sent_preds, all_sent_trues = [], []
        all_risk_preds, all_risk_trues = [], []

        with torch.no_grad():  # 评估时不计算梯度，节省显存和计算
            for batch in self.dev_loader:
                input_ids, attention_mask, aspect_ids, senti_ids, risk_ids = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                # 标签转换（与训练时一致）
                dim_labels = convert_aspect_to_multihot(aspect_ids).to(self.device)
                senti_labels = convert_senti_to_multihot(senti_ids).to(self.device)
                risk_labels = risk_ids.to(self.device)

                # 前向传播
                outputs = self.model(input_ids, attention_mask)
                # 计算损失（只取 loss_dict 用于累计，不需要梯度）
                _, loss_dict = self._compute_loss(
                    outputs, dim_labels, senti_labels, risk_labels
                )
                # 累加总损失（loss_dict 中没有 "total" 键时手动求和）
                total_loss += loss_dict["total"] if "total" in loss_dict else (
                    loss_dict["dim"] + loss_dict["sent"] + loss_dict["risk"]
                )

                # --- 预测 ---
                # 维度预测：sigmoid > 0.5 → 1，否则 0
                dim_pred = (torch.sigmoid(outputs["dimensions"]) > 0.5).int().cpu()
                # 情感预测：sigmoid > 0.5 → 1，否则 0
                sent_pred = (torch.sigmoid(outputs["sentiments"]) > 0.5).int().cpu()
                # 风险预测：取 logits 最大的类别
                risk_pred = torch.argmax(outputs["risk"], dim=1).cpu()

                # 收集本 batch 的预测和真实值
                all_dim_preds.append(dim_pred)
                all_dim_trues.append(dim_labels.cpu())
                all_sent_preds.append(sent_pred)
                all_sent_trues.append(senti_labels.cpu())
                all_risk_preds.append(risk_pred)
                all_risk_trues.append(risk_labels.cpu())

        # --- 合并所有 batch 的结果 ---
        dim_pred = torch.cat(all_dim_preds).numpy()
        dim_true = torch.cat(all_dim_trues).numpy()
        sent_pred = torch.cat(all_sent_preds).numpy()
        sent_true = torch.cat(all_sent_trues).numpy()
        risk_pred = torch.cat(all_risk_preds).numpy()
        risk_true = torch.cat(all_risk_trues).numpy()

        # --- 计算 macro-F1 ---
        # 维度 F1：对所有 8 个维度计算 macro-F1
        dim_f1 = f1_score(dim_true, dim_pred, average="macro", zero_division=0)
        # 情感 F1：只在「提及的维度」上计算（mask = dim_true == 1）
        mask = dim_true == 1
        sent_f1 = f1_score(sent_true[mask], sent_pred[mask], average="macro", zero_division=0) if mask.sum() > 0 else 0.0
        # 风险 F1：4 分类的 macro-F1
        risk_f1 = f1_score(risk_true, risk_pred, average="macro", zero_division=0)

        # 返回平均损失和指标
        return total_loss / len(self.dev_loader), {
            "dim_f1": dim_f1,
            "sent_f1": float(sent_f1),
            "risk_f1": risk_f1,
        }

    # ============================================================
    # 完整训练（两阶段入口）
    #
    # 按顺序执行 Stage 1 → Stage 2，最终输出最优模型信息。
    # 所有超参数均可通过 train() 的参数覆盖默认值。
    # ============================================================
    def train(self,
              stage1_epochs=3, stage1_lr=1e-3, stage1_gamma=0.95,
              stage2_epochs=5, stage2_base_lr=2e-5, stage2_head_lr=1e-4, stage2_gamma=0.95):
        """
        完整两阶段训练入口。

        参数：
            stage1_epochs (int): Stage 1 训练轮数，默认 3
            stage1_lr (float): Stage 1 学习率，默认 1e-3
            stage1_gamma (float): Stage 1 学习率衰减系数，默认 0.95
            stage2_epochs (int): Stage 2 训练轮数，默认 5
            stage2_base_lr (float): Stage 2 BERT 基础学习率，默认 2e-5
            stage2_head_lr (float): Stage 2 分类头学习率，默认 1e-4
            stage2_gamma (float): Stage 2 学习率衰减系数，默认 0.95

        返回：
            dict: {"stage1": h1, "stage2": h2}，包含两阶段的训练历史
        """
        # Stage 1: 冻结 BERT，仅训练分类头
        h1 = self.train_stage1(
            epochs=stage1_epochs, lr=stage1_lr, gamma=stage1_gamma
        )
        # Stage 2: 解冻 BERT，分层学习率整体微调
        h2 = self.train_stage2(
            epochs=stage2_epochs, base_lr=stage2_base_lr,
            head_lr=stage2_head_lr, gamma=stage2_gamma,
        )
        # 打印最终训练结果摘要
        print(f"\nBest val_loss={self.best_val_loss:.4f} @ epoch {self.best_epoch}")
        print(f"Model saved to: {conf.model_save_path}")
        return {"stage1": h1, "stage2": h2}


if __name__ == "__main__":
    # 创建训练器实例并启动两阶段训练
    trainer = Trainer()
    trainer.train()
