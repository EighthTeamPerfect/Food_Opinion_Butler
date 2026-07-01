# 对接大模型，基于分析结果生成可落地的运营处置建议
from openai import OpenAI
import os
from dotenv import load_dotenv
from data_preprocess_addpath import ASPECT_RULES, RISK_RULES

# 定位env文件
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
env_path = os.path.join(project_root, "uk.env")
print("env完整路径：", env_path)
print("文件存在？", os.path.exists(env_path))

# 显式返回加载结果，判断是否加载成功
load_success = load_dotenv(dotenv_path=env_path, encoding="utf-8")
print("dotenv加载是否成功：", load_success)

api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model_name = os.getenv("LLM_MODEL", "qwen-plus")

print("API_KEY是否为空：", api_key is None)
print("BASE_URL：", base_url)
print("模型名称：", model_name)

# 提前拦截空密钥，不用等到调用接口才报错
if not api_key:
    raise Exception("LLM_API_KEY读取失败，请检查uk.env文件路径、格式、编码")

# 初始化客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 风险等级映射
RISK_NAME_MAP = {
    0: "无风险好评",
    1: "普通轻度吐槽",
    2: "中度差评投诉",
    3: "高危食品安全舆情"
}

# 生成处置建议Prompt
def generate_suggestion_prompt(analysis_result):
    """基于分析结果生成大模型Prompt"""
    risk_id = analysis_result["risk_label_id"]
    risk_name = RISK_NAME_MAP[risk_id]
    aspect_list = analysis_result["aspect_label_id"].split(",")
    senti_list = analysis_result["senti_label"].split(",")
    # 解析负面维度
    negative_aspects = []
    for aid in aspect_list:
        if not aid:
            continue
        aid = int(aid)
        if senti_list[aid] == "0":
            negative_aspects.append(ASPECT_RULES[aid]["name"])
    # 构建Prompt
    prompt = f"""
    你是餐饮行业运营专家，现在有一条用户评论的舆情分析结果，请基于以下信息生成可落地的运营处置建议。
    【用户原始评论】：{analysis_result['raw_text']}
    【舆情风险等级】：{risk_name}（等级{risk_id}）
    【负面评价维度】：{','.join(negative_aspects) if negative_aspects else '无'}
    【置信度】：{analysis_result['confidence']:.2%}

    请按照以下要求生成建议：
    1. 先给出1句话的核心结论，明确该评论的核心问题和风险等级
    2. 分维度给出具体的改进建议，每条建议要可落地、可执行，不要空泛的套话
    3. 针对不同风险等级给出对应的处置优先级和响应要求
    4. 最终输出格式清晰，分点列出，适合餐饮门店运营人员直接使用
    """
    return prompt


# 生成处置建议
def generate_suggestion(analysis_result):
    """基于分析结果生成运营处置建议"""
    try:
        prompt = generate_suggestion_prompt(analysis_result)
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "qwen-plus"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        suggestion = response.choices[0].message.content.strip()
        return suggestion
    except Exception as e:
        print(f"大模型调用失败: {e}")
        return "生成建议失败，请检查大模型API配置"


# 主入口
if __name__ == "__main__":
    # 测试用例
    test_result = {
        "raw_text": "这家店的菜太咸了，服务员态度很差，等了一个小时才上菜，再也不来了！",
        "risk_label_id": 2,
        "aspect_label_id": "0,3,4",
        "senti_label": "0,1,1,0,0,1,1,1",
        "confidence": 0.92
    }
    suggestion = generate_suggestion(test_result)
    print("运营处置建议：", suggestion)