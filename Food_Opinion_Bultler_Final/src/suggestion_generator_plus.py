# 对接阿里云百炼大模型，基于舆情分析结果生成餐饮运营落地处置建议
import os
import traceback
from dotenv import load_dotenv
from openai import OpenAI

# ===================== 1. 自动加载环境配置（优先执行） =====================
# 自动拼接项目根目录uk.env绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
env_path = os.path.join(project_root, "uk.env")

# 打印调试信息，排查env读取问题
print(f"[ENV调试] env文件完整路径：{env_path}")
print(f"[ENV调试] 文件是否存在：{os.path.exists(env_path)}")
# 指定utf-8编码加载环境文件
load_success = load_dotenv(dotenv_path=env_path, encoding="utf-8")
print(f"[ENV调试] dotenv加载是否成功：{load_success}")

# 读取环境变量
api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
model_name = os.getenv("LLM_MODEL", "qwen-plus")

# 前置拦截空密钥，避免初始化客户端后模糊报错
if not api_key:
    raise Exception("配置错误：LLM_API_KEY为空，请检查项目根目录uk.env文件格式、编码、文件名！")

# 打印关键配置确认
print(f"[ENV调试] API_KEY是否为空：{api_key is None}")
print(f"[ENV调试] BASE_URL：{base_url}")
print(f"[ENV调试] 模型名称：{model_name}")

# 初始化阿里云百炼兼容OpenAI客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# ===================== 2. 业务模块导入（配置加载完成后导入） =====================
from data_preprocess_addpath import ASPECT_RULES, RISK_RULES

# 风险等级文字映射
RISK_NAME_MAP = {
    0: "无风险好评",
    1: "普通轻度吐槽",
    2: "中度差评投诉",
    3: "高危食品安全舆情"
}

# ===================== 3. Prompt生成工具函数 =====================
def generate_suggestion_prompt(analysis_result):
    """
    根据舆情结构化分析结果组装大模型提示词
    :param analysis_result: 推理流水线输出的结构化字典
    :return: 完整prompt字符串
    """
    risk_id = analysis_result["risk_label_id"]
    risk_name = RISK_NAME_MAP[risk_id]
    aspect_list = analysis_result["aspect_label_id"].split(",")
    senti_list = analysis_result["senti_label"].split(",")

    # 解析所有负面维度，增加空字符串、下标越界防护
    negative_aspects = []
    for aid_str in aspect_list:
        aid_str = aid_str.strip()
        # 过滤空字符串
        if not aid_str:
            continue
        aid = int(aid_str)
        # 防止情感列表下标越界
        if aid >= len(senti_list):
            continue
        # senti=0代表负面评价
        if senti_list[aid] == "0":
            negative_aspects.append(ASPECT_RULES[aid]["name"])

    # 标准化prompt，强制输出结构化运营建议
    prompt = f"""
你是专业餐饮门店运营舆情处理专家，请根据用户评论分析结果生成可落地的处置方案。
【用户原始评论】：{analysis_result['raw_text']}
【舆情风险等级】：{risk_name}（风险等级数字：{risk_id}）
【用户负面评价维度】：{','.join(negative_aspects) if negative_aspects else '无负面维度，为好评'}
【模型识别置信度】：{analysis_result['confidence']:.2%}

输出要求：
1. 第一段：一句话总结本条评论核心问题与风险等级；
2. 第二段：分点列出对应负面维度的门店整改落地建议，拒绝空泛套话；
3. 第三段：根据风险等级给出客户接待、回访、整改优先级规范；
4. 排版清晰，分段输出，适合门店运营人员直接使用。
"""
    return prompt

# ===================== 4. 对外调用主函数 =====================
def generate_suggestion(analysis_result):
    """
    对外统一入口：输入舆情分析结果，返回大模型生成的运营处置建议
    :param analysis_result: inference_pipeline输出的预测字典
    :return: str 处置建议 / 错误提示文本
    """
    try:
        prompt = generate_suggestion_prompt(analysis_result)
        # 调用阿里云百炼兼容接口
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600
        )
        # 提取返回文本并去除首尾空白
        suggestion = response.choices[0].message.content.strip()
        return suggestion
    except Exception as e:
        # 打印完整异常堆栈，精准定位鉴权/网络/模型错误
        print("=" * 60)
        print("大模型调用完整异常堆栈：")
        print(traceback.format_exc())
        print("=" * 60)
        err_msg = f"运营建议生成失败，错误详情：{str(e)}，请检查API密钥、模型权限、网络连接"
        print(f"大模型调用失败: {str(e)}")
        return err_msg

# ===================== 5. 本地测试入口（直接运行可调试） =====================
if __name__ == "__main__":
    # 模拟推理流水线输出的结构化结果
    test_analysis = {
        "raw_text": "这家店菜品太咸，服务员态度差，上菜等待一小时，体验很差！",
        "risk_label_id": 2,
        "aspect_label_id": "0,3,4",
        "senti_label": "0,1,1,0,0,1,1,1",
        "confidence": 0.93
    }
    print("\n===== 开始测试大模型建议生成 =====")
    res = generate_suggestion(test_analysis)
    print("\n【门店运营处置建议】\n")
    print(res)