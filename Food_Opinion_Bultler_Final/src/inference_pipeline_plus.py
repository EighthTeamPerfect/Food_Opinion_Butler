import torch
import joblib
import fasttext
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 导入公共预处理工具
from data_preprocess_addpath import clean_text, tensor2aspect_str, tensor2senti_str
# 导入各模型预测函数
from bert_model import MultiTaskBert, predict_bert
from rf_model import predict_rf
from fasttext_model import predict_fasttext

# ===================== 全局固定配置 =====================
# 设备自动识别
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 模型文件路径
BERT_MODEL_PATH = "../models/bert_multi_task/"
RF_MODEL_PATH = "../models/rf_model.pkl"
FASTTEXT_MODEL_PATH = "../models/fasttext_model.bin"

# 内置行业关键词规则（解决导入缺失ASPECT_RULES/RISK_RULES报错）
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
# =======================================================

def load_models():
    """统一加载全部模型，返回models字典，修复models未定义问题"""
    models = {}
    # 1. 加载BERT多任务模型
    try:
        # 加载分词器
        models["bert_tokenizer"] = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
        # 初始化网络结构
        bert_model = MultiTaskBert().to(DEVICE)
        # 加载本地训练权重
        weight_file = f"{BERT_MODEL_PATH}/pytorch_model.bin"
        bert_model.load_state_dict(torch.load(weight_file, map_location=DEVICE))
        bert_model.eval()
        models["bert_model"] = bert_model
        print("✅ BERT模型加载成功")
    except Exception as e:
        print(f"❌ BERT模型加载失败: {str(e)}")

    # 2. 加载RF随机森林模型
    try:
        rf_data = joblib.load(RF_MODEL_PATH)
        models["rf_vectorizer"] = rf_data["vectorizer"]
        models["rf_aspect_model"] = rf_data["aspect_model"]
        models["rf_risk_model"] = rf_data["risk_model"]
        print("✅ RF模型加载成功")
    except Exception as e:
        print(f"❌ RF模型加载失败: {str(e)}")

    # 3. 加载FastText轻量模型
    try:
        models["fasttext_model"] = fasttext.load_model(FASTTEXT_MODEL_PATH)
        print("✅ FastText模型加载成功")
    except Exception as e:
        print(f"❌ FastText模型加载失败: {str(e)}")

    return models

def post_process_labels(raw_text, pred_result):
    """标签后处理：基于SOP规则兜底修正预测结果"""
    text_lower = str(raw_text).lower()
    # 规则1：命中食品安全负面词，强制拉高风险+补充食品安全维度
    if any(word in text_lower for word in ASPECT_RULES[7]["neg"]):
        pred_result["risk_label_id"] = 3
        if "7" not in pred_result["aspect_label_id"]:
            pred_result["aspect_label_id"] = pred_result["aspect_label_id"] + ",7" if pred_result["aspect_label_id"] else "7"

    # 拆分情感标签
    senti_list = pred_result["senti_label"].split(",") if pred_result["senti_label"] else []
    neg_dim_count = 0
    aspect_ids = pred_result["aspect_label_id"].split(",") if pred_result["aspect_label_id"] else []
    for aid_str in aspect_ids:
        if not aid_str:
            continue
        aid = int(aid_str)
        if aid < len(senti_list) and senti_list[aid] == "0":
            neg_dim_count += 1

    # 规则2：无任何负面维度，风险强制0
    if neg_dim_count == 0:
        pred_result["risk_label_id"] = 0
    # 规则3：≥2个负面维度，风险至少2级
    if neg_dim_count >= 2 and pred_result["risk_label_id"] < 2:
        pred_result["risk_label_id"] = 2
    return pred_result

def inference_pipeline(raw_text, models, model_type="bert"):
    """
    端到端推理流水线
    :param raw_text: 用户原始评论文本
    :param models: load_models() 返回的模型字典
    :param model_type: bert / rf / fasttext
    :return: 完整结构化预测结果
    """
    # 修复变量重名bug：使用cleaned_txt接收清洗后文本，不再覆盖函数名clean_text
    cleaned_txt = clean_text(raw_text)
    if len(cleaned_txt.strip()) == 0:
        return {"error": "文本内容为空，请输入有效评论"}

    # 根据模型类型执行推理
    if model_type == "bert":
        if "bert_model" not in models or "bert_tokenizer" not in models:
            return {"error": "BERT模型未成功加载，无法推理"}
        pred_res = predict_bert(cleaned_txt, models["bert_model"], models["bert_tokenizer"])
    elif model_type == "rf":
        if "rf_aspect_model" not in models:
            return {"error": "RF模型未成功加载，无法推理"}
        pred_res = predict_rf(cleaned_txt, models["rf_vectorizer"], models["rf_aspect_model"], models["rf_risk_model"])
        # RF无情感输出，填充默认正向
        pred_res["senti_label"] = "1,1,1,1,1,1,1,1"
        pred_res["confidence"] = 0.8
    elif model_type == "fasttext":
        if "fasttext_model" not in models:
            return {"error": "FastText模型未成功加载，无法推理"}
        pred_res = predict_fasttext(cleaned_txt, models["fasttext_model"])
        pred_res["senti_label"] = "1,1,1,1,1,1,1,1"
        pred_res["confidence"] = 0.75
    else:
        return {"error": f"不支持的模型类型：{model_type}，可选 bert/rf/fasttext"}

    # 规则后处理修正标签
    pred_res = post_process_labels(raw_text, pred_res)
    # 补充基础文本信息
    pred_res["raw_text"] = raw_text
    pred_res["clean_text"] = cleaned_txt
    pred_res["text_length"] = len(cleaned_txt)
    pred_res["risk_desc"] = RISK_RULES[pred_res["risk_label_id"]]
    return pred_res

# 测试入口
if __name__ == "__main__":
    print("==== 加载全部模型 ====")
    all_models = load_models()
    print("\n==== 开始单条推理测试 ====")
    test_comment = "这家店菜品太咸，服务员态度很差，上菜等待时间很久，不推荐！"
    output = inference_pipeline(test_comment, all_models, model_type="bert")
    # 格式化打印结果
    for k, v in output.items():
        print(f"{k}：{v}")