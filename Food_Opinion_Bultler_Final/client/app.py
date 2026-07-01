# Streamlit 交互式界面，非技术人员可直接使用
import streamlit as st
import requests
import json

# 页面配置
st.set_page_config(page_title="餐饮评论舆情分析系统", page_icon="🍜", layout="wide")

# 服务端地址
API_URL = "http://localhost:8000/api/analyze"

# 页面标题
st.title("🍜 餐饮评论舆情分析与运营建议系统")
st.markdown("基于BERT多任务模型的餐饮评论智能分析，自动识别评价维度、情感极性、风险等级，生成可落地的运营处置建议")

# 侧边栏配置
with st.sidebar:
    st.header("模型配置")
    model_type = st.selectbox("选择分析模型", ["bert", "rf", "fasttext"], index=0)
    generate_suggestion = st.checkbox("生成运营处置建议", value=True)
    st.markdown("---")
    st.markdown("### 风险等级说明")
    st.markdown("""
    - 0️⃣ 无风险好评：正向夸赞，无任何吐槽
    - 1️⃣ 普通轻度吐槽：轻微不满，无负面维度
    - 2️⃣ 中度差评投诉：明显不满，存在负面维度
    - 3️⃣ 高危食品安全舆情：涉及食品安全问题
    """)

# 主界面
input_text = st.text_area("请输入用户评论内容", height=150,
                          placeholder="例如：这家店的菜太咸了，服务员态度很差，等了一个小时才上菜，再也不来了！")

# 分析按钮
if st.button("开始分析", type="primary", use_container_width=True):
    if not input_text.strip():
        st.error("请输入评论内容")
    else:
        with st.spinner("正在分析中..."):
            try:
                # 调用API
                response = requests.post(
                    API_URL,
                    json={
                        "text": input_text,
                        "model_type": model_type,
                        "generate_suggestion": generate_suggestion
                    }
                )
                response.raise_for_status()
                result = response.json()["data"]

                # 展示结果
                st.success("分析完成！")
                st.markdown("---")

                # 基础信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("文本长度", result["text_length"])
                with col2:
                    st.metric("置信度", f"{result['confidence']:.2%}")
                with col3:
                    risk_name = {0: "无风险好评", 1: "普通轻度吐槽", 2: "中度差评投诉", 3: "高危食品安全舆情"}[
                        result["risk_label_id"]]
                    st.metric("风险等级", risk_name)

                # 标签信息
                st.markdown("### 标签识别结果")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**评价维度标签**")
                    aspect_names = ["菜品口味", "食材新鲜度", "门店环境", "服务态度", "上菜速度", "性价比", "外卖包装",
                                    "食品安全卫生"]
                    aspect_list = result["aspect_label_id"].split(",")
                    for aid in aspect_list:
                        if aid:
                            st.markdown(f"- {aspect_names[int(aid)]}")
                with col_b:
                    st.markdown("**情感极性标签**")
                    senti_list = result["senti_label"].split(",")
                    for aid in aspect_list:
                        if aid:
                            senti = "正向" if senti_list[int(aid)] == "1" else "负面"
                            st.markdown(f"- {aspect_names[int(aid)]}：{senti}")

                # 运营建议
                if generate_suggestion and "suggestion" in result:
                    st.markdown("### 运营处置建议")
                    st.markdown(result["suggestion"])

            except Exception as e:
                st.error(f"分析失败：{str(e)}")

# 页脚
st.markdown("---")
st.markdown("© 2024 餐饮评论舆情分析系统 | 基于BERT多任务模型开发")