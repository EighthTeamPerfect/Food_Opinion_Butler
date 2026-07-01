import streamlit as st
import requests
import time

# 设置页面标题
st.title("食品舆情管家")

# 创建输入框
text_input = st.text_area("请输入要预测的文本：", "中华女子学院：本科层次仅1专业招男生")

# 创建预测按钮
if st.button("预测"):
    # 构造请求数据
    data = {'text': text_input}
    url = 'http://localhost:8008/predict'

    # 记录开始时间
    start_time = time.time()

    try:
        # 发送 POST 请求
        response = requests.post(url, json=data)
        # 计算耗时（毫秒）
        elapsed_time = (time.time() - start_time) * 1000

        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            st.success(f"预测结果: {result}")
            st.info(f"请求耗时: {elapsed_time:.2f} ms")
        else:
            st.error(f"请求失败: {response.json()['error']}")
    except Exception as e:
        st.error(f"请求出错: {str(e)}")

# 运行提示
st.write("请确保 Flask API 服务已在 localhost:8004 运行")
