# 串联文本清洗→模型推理→标签后处理，提供统一的推理接口
import torch
# from transformers import BertTokenizer, BertModel
from transformers import BertTokenizer
# 先加载tokenizer
BERT_MODEL_PATH = "../../bert-base-chinese"
models["bert_tokenizer"] = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
# 初始化网络结构
bert_model = MultiTaskBert().to(DEVICE)
# 加载本地权重文件
weight_file = f"{BERT_MODEL_PATH}/pytorch_model.bin"
bert_model.load_state_dict(torch.load(weight_file, map_location=DEVICE))
bert_model.eval()
models["bert_model"] = bert_model
print("✅ BERT模型加载成功")

from data_preprocess_addpath import clean_text, tensor2aspect_str, tensor2senti_str
from bert_model import MultiTaskBert, predict_bert
from rf_model import predict_rf
from fasttext_model import predict_fasttext
import joblib
import fasttext

# 全局配置
BERT_MODEL_PATH = "../models/bert_multi_task/"
RF_MODEL_PATH = "../models/rf_model.pkl"
FASTTEXT_MODEL_PATH = "../models/fasttext_model.bin"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 移除错误导入
# from data_preprocess_addpath import ... , ASPECT_RULES, RISK_RULES

# 本地定义规则
ASPECT_RULES = {
    0: {"pos": ["好吃","美味","入味","新鲜","赞"], "neg": ["难吃","太咸","发苦","腥","油腻"]},
    1: {"pos": ["新鲜","嫩","活的"], "neg": ["不新鲜","变质","臭","发酸","放久"]},
    2: {"pos": ["干净","宽敞","安静"], "neg": ["脏","苍蝇","拥挤","油烟大"]},
    3: {"pos": ["热情","周到","及时"], "neg": ["态度差","不理人","凶","欺骗"]},
    4: {"pos": ["上菜快","不用等"], "neg": ["等很久","上菜慢","催单没用"]},
    5: {"pos": ["划算","实惠","量大"], "neg": ["太贵","不值","分量少"]},
    6: {"pos": ["不漏","包装完好"], "neg": ["洒了","破","缺餐具"]},
    7: {"pos": ["卫生安全"], "neg": ["头发","虫子","拉肚子","食物中毒","异味"]}
}
RISK_RULES = {
    0: "无风险好评",
    1: "普通轻度吐槽",
    2: "中度差评投诉",
    3: "高危食品安全舆情"
}

# 加载所有模型
def load_models():
    """加载所有预训练模型，返回模型字典"""
    models = {}
    # 加载BERT模型
    try:
        models["bert_tokenizer"] = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        models["bert_model"] = MultiTaskBert().to(DEVICE)
        models["bert_model"].load_state_dict(torch.load(f"{BERT_MODEL_PATH}/pytorch_model.bin", map_location=DEVICE))
        models["bert_model"].eval()
        print("✅ BERT模型加载成功")
    except Exception as e:
        print(f"❌ BERT模型加载失败: {e}")

    # 加载RF模型
    try:
        rf_data = joblib.load(RF_MODEL_PATH)
        models["rf_vectorizer"] = rf_data["vectorizer"]
        models["rf_aspect_model"] = rf_data["aspect_model"]
        models["rf_risk_model"] = rf_data["risk_model"]
        print("✅ RF模型加载成功")
    except Exception as e:
        print(f"❌ RF模型加载失败: {e}")

    # 加载FastText模型
    try:
        models["fasttext_model"] = fasttext.load_model(FASTTEXT_MODEL_PATH)
        print("✅ FastText模型加载成功")
    except Exception as e:
        print(f"❌ FastText模型加载失败: {e}")

    return models


# 标签后处理规则校验
def post_process_labels(text, pred_result):
    """基于SOP规则对预测结果做兜底校验，修正明显错误"""
    text = str(text).lower()
    # 规则1：命中食品安全负面关键词，强制风险等级3
    if any(keyword in text for keyword in ASPECT_RULES[7]["neg"]):
        pred_result["risk_label_id"] = 3
        if "7" not in pred_result["aspect_label_id"]:
            pred_result["aspect_label_id"] += ",7" if pred_result["aspect_label_id"] else "7"
    # 规则2：无任何负面维度，强制风险等级0
    neg_aspect = []
    for aid in range(8):
        if str(aid) in pred_result["aspect_label_id"] and pred_result["senti_label"].split(",")[aid] == "0":
            neg_aspect.append(aid)
    if len(neg_aspect) == 0:
        pred_result["risk_label_id"] = 0
    # 规则3：2个及以上负面维度，强制风险等级2
    if len(neg_aspect) >= 2:
        pred_result["risk_label_id"] = max(pred_result["risk_label_id"], 2)
    return pred_result


# 端到端推理函数
def inference_pipeline(text, models, model_type="bert"):
    """
    端到端推理Pipeline
    :param text: 原始评论文本
    :param models: 加载的模型字典
    :param model_type: 选择模型：bert/rf/fasttext
    :return: 完整的预测结果
    """
    # 1. 文本清洗
    clean_text = clean_text(text)
    if len(clean_text) == 0:
        return {"error": "文本内容为空"}

    # 2. 模型推理
    if model_type == "bert":
        if "bert_model" not in models:
            return {"error": "BERT模型未加载"}
        pred_result = predict_bert(clean_text, models["bert_model"], models["bert_tokenizer"])
    elif model_type == "rf":
        if "rf_aspect_model" not in models:
            return {"error": "RF模型未加载"}
        pred_result = predict_rf(clean_text, models["rf_vectorizer"], models["rf_aspect_model"],
                                 models["rf_risk_model"])
        pred_result["senti_label"] = "1,1,1,1,1,1,1,1"  # RF模型无情感预测，默认正向
        pred_result["confidence"] = 0.8
    elif model_type == "fasttext":
        if "fasttext_model" not in models:
            return {"error": "FastText模型未加载"}
        pred_result = predict_fasttext(clean_text, models["fasttext_model"])
        pred_result["senti_label"] = "1,1,1,1,1,1,1,1"
        pred_result["confidence"] = 0.75
    else:
        return {"error": "不支持的模型类型"}

    # 3. 标签后处理规则校验
    pred_result = post_process_labels(clean_text, pred_result)
    # 4. 补充文本信息
    pred_result["raw_text"] = text
    pred_result["clean_text"] = clean_text
    pred_result["text_length"] = len(clean_text)
    return pred_result


# 主入口
if __name__ == "__main__":
    models = load_models()
    test_text = "这家店的菜太咸了，服务员态度很差，等了一个小时才上菜，再也不来了！"
    result = inference_pipeline(test_text, models, model_type="bert")
    print("推理结果：", result)