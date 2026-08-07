from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from utils.database import load_user_data, load_merchant_data, load_interaction_data, load_area_data
from utils.recommendation import create_user_merchant_matrix, collaborative_filtering, tag_based_recommendation
from utils.feedback import save_user_feedback
from fastapi import HTTPException
from routers.api import router as api_router
from agents import RecommendationAgent, MultiAgentSystem, get_available_tools_info
from agents.tools import SKILL_REGISTRY

# 导入配置
try:
    from config import LLM_PROVIDER, QWEN_CONFIG
except ImportError:
    # 配置文件不存在时使用默认值
    LLM_PROVIDER = "mock"
    QWEN_CONFIG = {"api_key": "", "model": "qwen-max"}

# 创建FastAPI应用
app = FastAPI()

# 注册API路由
app.include_router(api_router, prefix="/api")

# 配置模板
templates = Jinja2Templates(directory="../frontend/templates")

# 初始化Agent - 多智能体系统 (LangGraph)
print(f"[MultiAgent] Initializing LangGraph with provider: {LLM_PROVIDER}, model: {QWEN_CONFIG.get('model', 'qwen-max')}")
agent = MultiAgentSystem()
print(f"[MultiAgent] System ready: orchestrator → [recommendation, analysis, profile] → aggregator")

# 初始化数据
user_df = load_user_data()
merchant_df = load_merchant_data()
interaction_df = load_interaction_data()
area_df = load_area_data()

# 创建用户-商户交互矩阵
user_merchant_matrix = create_user_merchant_matrix(interaction_df)

# 根路径
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 数据可视化页面
@app.get("/visualization", response_class=HTMLResponse)
def visualization(request: Request):
    return templates.TemplateResponse("data_visualization.html", {"request": request})

# 检查用户是否存在
@app.post("/check_user", response_class=HTMLResponse)
def check_user(request: Request, user_id: str = Form(...)):
    # 检查用户是否存在
    user_exists = user_id in user_df['user_id'].values
    
    if user_exists:
        # 老用户，获取推荐
        recommendations = collaborative_filtering(user_id, user_merchant_matrix, interaction_df, merchant_df)
        recommendations = recommendations.to_dict('records')
        return templates.TemplateResponse("recommendations.html", {
            "request": request,
            "user_id": user_id,
            "recommendations": recommendations,
            "is_new_user": False
        })
    else:
        # 新用户，显示标签选择
        return templates.TemplateResponse("tag_selection.html", {
            "request": request,
            "user_id": user_id
        })

# 新用户标签选择
@app.post("/tag_selection", response_class=HTMLResponse)
def tag_selection(request: Request, user_id: str = Form(...), cuisine: str = Form(...), price_range: str = Form(...)):
    # 基于标签推荐
    recommendations = tag_based_recommendation(cuisine, price_range, merchant_df)
    recommendations = recommendations.to_dict('records')
    
    return templates.TemplateResponse("recommendations.html", {
        "request": request,
        "user_id": user_id,
        "recommendations": recommendations,
        "is_new_user": True,
        "cuisine": cuisine,
        "price_range": price_range
    })

# 提交反馈
@app.post("/submit_feedback")
def submit_feedback(user_id: str = Form(...), merchant_id: str = Form(...), is_satisfied: bool = Form(...), reason: str = Form(None)):
    try:
        save_user_feedback(user_id, merchant_id, is_satisfied, reason)
        return {"status": "success", "message": "反馈提交成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Agent对话端点 - 多智能体协作
@app.post("/chat")
def chat(message: str, thread_id: str = "default", user_id: str = None):
    try:
        result = agent.run(message, user_id=user_id, thread_id=thread_id)
        return {
            "response": result["response"],
            "trace": result["trace"],
            "iterations": result["iterations"],
            "tools_used": result["tools_used"],
            "intent": result.get("intent", ""),
            "routed_agents": result.get("routed_agents", []),
            "thread_id": result.get("thread_id", thread_id),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Agent流式对话端点 - SSE (Server-Sent Events)
@app.post("/chat/stream")
def chat_stream(message: str, thread_id: str = "default", user_id: str = None):
    """
    流式对话端点，返回 SSE 事件流。

    事件类型:
      - meta:  {intent, routed_agents, tools_used}
      - trace: [trace entries]
      - token: "内容片段"
      - done:  {iterations}
      - error: "错误信息"
    """
    import json as _json

    def event_generator():
        for event in agent.run_stream(message, user_id=user_id, thread_id=thread_id):
            event_type = event.get("event", "message")
            data = event.get("data", "")
            # SSE 格式: event: <type>\ndata: <json>\n\n
            if isinstance(data, (dict, list)):
                data_str = _json.dumps(data, ensure_ascii=False)
            else:
                data_str = str(data)
            yield f"event: {event_type}\ndata: {data_str}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 nginx 缓冲
        },
    )

# Agent对话页面
@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    tools_info = get_available_tools_info()
    agent_stats = agent.get_stats()
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "tools": tools_info,
        "agent_stats": agent_stats
    })

# Agent状态端点
@app.get("/agent/status")
def agent_status():
    return agent.get_stats()

# Agent可用工具列表
@app.get("/agent/tools")
def agent_tools():
    return get_available_tools_info()

# Skill 工程体系: 已注册 Skill 列表
@app.get("/agent/skills")
def agent_skills():
    return {
        "total_skills": SKILL_REGISTRY.skill_count,
        "skills": SKILL_REGISTRY.list_skills(),
        "by_category": SKILL_REGISTRY.list_by_category()
    }

# 单个 Skill 详情（含工具 schema）
@app.get("/agent/skills/{skill_name}")
def agent_skill_detail(skill_name: str):
    skill = SKILL_REGISTRY.get_skill(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")
    return {
        "metadata": skill,
        "tools": SKILL_REGISTRY.get_skill_tools(skill_name)
    }

# 重置Agent记忆
@app.post("/agent/reset")
def agent_reset(thread_id: str = None):
    agent.clear_memory(thread_id=thread_id)
    scope = f"thread '{thread_id}'" if thread_id else "all threads"
    return {"status": "success", "message": f"Agent memory cleared ({scope})"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()