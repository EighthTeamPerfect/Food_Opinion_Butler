# FastAPI 封装，提供标准化的 HTTP 接口，供前端 / 第三方系统调用
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# # 获取当前api文件夹路径
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 项目根目录（api上一级 day06）
# root_dir = os.path.dirname(current_dir)
# # src目录
# src_dir = os.path.join(root_dir, "src")
# # 把src加入系统路径
# sys.path.append(src_dir)
#
# from src.inference_pipeline_plus import load_models, inference_pipeline
# from src.suggestion_generator_plus import generate_suggestion

# 1. 计算路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
src_dir = os.path.join(root_dir, "src")

# 2. 把src加入系统搜索路径
sys.path.append(src_dir)

# 调试打印，确认路径生效
print("src路径：", src_dir)
print("文件夹存在：", os.path.exists(src_dir))

# 3. 导入不再带 src. 前缀
from inference_pipeline_plus import load_models, inference_pipeline
from suggestion_generator_plus import generate_suggestion



# 初始化FastAPI
app = FastAPI(title="餐饮评论舆情分析API", version="1.0.0")

# 全局加载模型
models = load_models()

# 请求体定义
class AnalyzeRequest(BaseModel):
    text: str
    model_type: str = "bert"  # 支持 bert/rf/fasttext
    generate_suggestion: bool = True

# 响应体定义
class AnalyzeResponse(BaseModel):
    code: int
    message: str
    data: dict

# 核心分析接口
@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_comment(request: AnalyzeRequest):
    """
    餐饮评论舆情分析接口
    - 输入：评论文本
    - 输出：维度标签、情感标签、风险等级、置信度、运营处置建议
    """
    try:
        print(f"请求参数：{request.json()}")
        # 执行推理
        result = inference_pipeline(request.text, models, model_type=request.model_type)
        print(f"推理结果：{result}")
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        print("222222")
        # 生成处置建议
        if request.generate_suggestion:
            print("3333")
            suggestion = generate_suggestion(result)
            print("444")
            result["suggestion"] = suggestion
            print("55")
        return AnalyzeResponse(code=200, message="分析成功", data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

# 健康检查接口
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}

# 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
