"""
bert_predict -



Author:郎朗
Version:0.0.1
Date:2026/6/30
"""
import torch
import torch.nn as nn
from config import Config
from bert_model import MuliTaskBertClassfier
from config import Config

config = Config()
model = MuliTaskBertClassfier()
model.load_state_dict(torch.load(config.model_save_path, map_location="cpu"))

def predict(data, threshold=0.5):
    model.eval()
    text = data["text"]
    text_token = config.tokenizer.encode_plus(
        text, return_tensors="pt", max_length=config.pad_size,
        truncation=True, padding="max_length"
    )
    input_ids = text_token["input_ids"]  # [1, seq_len]
    attention_mask = text_token["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        dim_preds = (torch.sigmoid(outputs["dimensions"]) > threshold).int()
        sent_preds = (torch.sigmoid(outputs["sentiments"]) > threshold).int()
        risk_preds = torch.argmax(outputs["risk"], dim=1)
        print(dim_preds[0].tolist())
        print(sent_preds[0].tolist())
        print(risk_preds)
        print(risk_preds.item())
    #根据模型预测结果，返回结果
    #维度提及

    #情感级性

    #舆情等级
    risk_preds_str = config.risk_label_id_class[risk_preds.item()]
    print(risk_preds_str)
    return {
        "维度提及": dim_preds[0].tolist(),
        "情感极性": sent_preds[0].tolist(),
        "舆情等级": risk_preds_str,
    }


if __name__ == '__main__':


    # data = {"text": "晚上没吃饭 路过这不知道还开没开门 就看了看 九点多了 还在营业 挺好的，不过东西已经不多了 反正也不是太饿 就随便拿了几样吃，味道还行 不知道是那种调料放多了 稍微有点咸，价钱好像稍微贵点 不知不觉的麻辣烫的价钱也好像涨了那么多"}
    # data = {"text": "菜品口味 好,食材新鲜度:好,服务态度:好,食品很卫生"}
    # data = {"text": "第一次来他家吃,饭菜口味很好,点了很多,性价比也很高"}
    data = {"text": "第一次来他家吃,饭菜口味很好,点了很多"}
    result = predict(data)
    print(result)

# 菜品口味
# 食材新鲜度
# 门店环境
# 服务态度
# 上菜速度
# 性价比价格
# 外卖包装
# 食品安全卫生