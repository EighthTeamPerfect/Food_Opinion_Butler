"""
bert_model -



Author:郎朗
Version:0.0.1
Date:2026/6/30
"""
import torch
import torch.nn as nn
from transformers import BertModel
from config import Config
config = Config()
class MuliTaskBertClassfier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = config.bert_model
        self.dropout = nn.Dropout(config.dropout)
        #  HEAD 1  八维多标签
        self.dim_head = nn.Linear(self.bert.config.hidden_size, config.num_dimensions)
        # HEAD 2 八维情感极性
        self.sentiment_head = nn.Linear(self.bert.config.hidden_size, config.num_dimensions)
        # Head 3: 舆情风险四分类
        self.risk_head = nn.Linear(self.bert.config.hidden_size, config.num_risk_levels)
    def forward(self, input_ids,attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)    # 维度 batch,768
        return {
            "dimensions": self.dim_head(pooled),         #维度  batch, 8
            "sentiments": self.sentiment_head(pooled),   #维度  batch, 4
            "risk": self.risk_head(pooled),              #维度  batch, 4
        }
if __name__ == '__main__':
    model = MuliTaskBertClassfier()
    print(model)