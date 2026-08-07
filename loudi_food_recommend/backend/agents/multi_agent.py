"""
多智能体协作系统 - 基于 LangGraph StateGraph 实现

架构:
  Orchestrator (主控) → 路由分发 → [Recommendation Agent | Analysis Agent | Profile Agent] → Aggregator (聚合)

协作模式:
  1. 路由分发: 简单任务路由到单个Agent
  2. 并行协作: 复杂任务并行调用多个Agent
  3. 流水线:   依赖任务按顺序执行
"""
import json
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .shared_state import AgentState
from .llm_service import LLMService
from .tools import (
    CollaborativeFilteringTool, TagBasedTool, ContentBasedTool,
    BusinessOverviewTool, AreaAnalysisTool, TopMerchantsTool,
    UserPreferencesTool, CheckUserTool,
    get_tool_by_name, TOOLS
)
from .rag_tool import SemanticSearchTool
from .observability import observe
from config import LLM_PROVIDER, QWEN_CONFIG


# ============================================================
# LLM 实例 (所有Agent共享)
# ============================================================
_shared_llm = None

def get_shared_llm() -> LLMService:
    global _shared_llm
    if _shared_llm is None:
        _shared_llm = LLMService(
            provider=LLM_PROVIDER,
            api_key=QWEN_CONFIG["api_key"],
            model=QWEN_CONFIG["model"]
        )
    return _shared_llm


# ============================================================
# 1. Orchestrator Agent (主控协调者)
#    职责: 意图识别 → 任务拆分 → Agent路由 → 结果聚合
# ============================================================

def orchestrator_node(state: AgentState) -> Dict[str, Any]:
    """主控Agent: 分析用户意图，决定路由到哪些子Agent"""
    user_input = state["user_input"]
    trace = state.get("trace", [])

    llm = get_shared_llm()

    # 用LLM进行意图识别和任务拆分
    intent_prompt = f"""分析以下用户输入，返回JSON格式的意图分类和路由决策。

用户输入: "{user_input}"

请分析意图类别:
- "recommendation": 餐厅推荐相关（推荐餐厅、找吃的、用户个性化推荐等）
- "analysis": 数据分析相关（业务概览、区域分析、排行榜、统计数据等）
- "profile": 用户偏好分析（用户喜欢什么、偏好分布、用户画像等）
- "complex": 复杂任务，需要多个Agent协作（如"分析用户偏好并推荐"）
- "chat": 普通闲聊，无需调用Agent

只返回JSON，格式如下:
{{"intent": "recommendation", "agents": ["recommendation"]}}

对于complex意图，agents可以包含多个Agent名称。"""

    messages = [
        {"role": "system", "content": "你是任务路由器，只返回JSON格式。"},
        {"role": "user", "content": intent_prompt}
    ]

    try:
        response = llm.chat(messages, tools=None)
        content = response.get("content", "")
        # 解析LLM返回的JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        intent_data = json.loads(content.strip())
        intent = intent_data.get("intent", "chat")
        routed_agents = intent_data.get("agents", [])
    except Exception:
        # LLM解析失败时的降级策略: 基于关键词的路由
        intent, routed_agents = _fallback_router(user_input)

    # 记录追踪
    trace.append({
        "iteration": state.get("iterations", 0) + 1,
        "phase": "orchestrator",
        "description": f"意图识别: {intent}",
        "routed_agents": routed_agents
    })

    return {
        "intent": intent,
        "routed_agents": routed_agents,
        "trace": trace,
        "iterations": state.get("iterations", 0) + 1
    }


def _fallback_router(user_input: str) -> tuple:
    """LLM解析失败时的关键词降级路由"""
    input_lower = user_input.lower()

    recommend_keywords = ["推荐", "川菜", "粤菜", "湘菜", "鲁菜", "火锅", "烧烤", "预算",
                          "好吃的", "餐厅", "美食", "user", "用户"]
    analysis_keywords = ["业务", "区域", "排行", "统计", "数据", "概览", "分析", "销售"]
    profile_keywords = ["偏好", "喜欢什么", "画像", "分布", "用户偏好"]

    is_recommend = any(kw in input_lower for kw in recommend_keywords)
    is_analysis = any(kw in input_lower for kw in analysis_keywords)
    is_profile = any(kw in input_lower for kw in profile_keywords)

    if is_recommend and (is_analysis or is_profile):
        agents = []
        if is_recommend:
            agents.append("recommendation")
        if is_analysis:
            agents.append("analysis")
        if is_profile:
            agents.append("profile")
        return "complex", agents
    elif is_recommend:
        return "recommendation", ["recommendation"]
    elif is_analysis:
        return "analysis", ["analysis"]
    elif is_profile:
        return "profile", ["profile"]
    else:
        return "chat", []


def route_from_orchestrator(state: AgentState) -> List[str]:
    """条件路由: 根据意图决定执行哪些子Agent"""
    agents = state.get("routed_agents", [])
    intent = state.get("intent", "chat")

    if intent == "chat" or not agents:
        return ["aggregator"]

    return agents


# ============================================================
# 2. Recommendation Agent (推荐专家)
#    工具: 协同过滤, 标签推荐, 内容推荐
# ============================================================

def recommendation_node(state: AgentState) -> Dict[str, Any]:
    """推荐Agent: 处理餐厅推荐相关任务"""
    user_input = state["user_input"]
    # 每个子Agent使用自己的本地trace，只返回新增的条目（merge_lists会自动合并）
    local_trace = []
    local_tools_used = []

    llm = get_shared_llm()

    # 推荐Agent的工具集
    rec_tools = [
        CollaborativeFilteringTool.get_schema(),
        TagBasedTool.get_schema(),
        ContentBasedTool.get_schema(),
        CheckUserTool.get_schema(),
        SemanticSearchTool.get_schema(),
    ]

    # System Prompt定义推荐专家的角色
    system_prompt = """你是智能餐饮推荐专家，擅长根据用户需求推荐餐厅。

【重要规则】你必须在回复用户前调用至少一个工具获取数据。禁止在未调用任何工具的情况下直接推荐或询问用户更多信息。即使信息不完整，也要先调用最接近的工具获取数据，再基于结果回复。

你有以下工具:
1. collaborative_filtering - 基于用户历史行为的协同过滤推荐（需要user_id）
2. tag_based_recommendation - 基于菜系和价格的标签推荐（需要cuisine和price_range）
3. content_based_recommendation - 基于评分和销量的热门推荐（无需参数，适合默认推荐）
4. check_user - 检查用户是否存在
5. semantic_search_merchants - 基于语义相似度检索商户（适用于自然语言描述，如"想吃辣的暖胃的"）

选择策略（按优先级）:
- 用户提到菜系关键词（川菜/粤菜/湘菜/火锅/烧烤/日料/西餐等）→ tag_based_recommendation
  示例: "推荐川菜" → tag_based_recommendation(cuisine="川菜")
  示例: "想吃火锅" → tag_based_recommendation(cuisine="火锅")
- 用户用自然语言描述感觉/场景（"辣的""暖胃的""适合约会"）→ semantic_search_merchants
  示例: "想吃辣的暖胃的" → semantic_search_merchants(query="辣的暖胃的")
- 用户提供了 user_id（如 user001/user_001/我是xxx）→ check_user 先检查，再 collaborative_filtering
- 用户没有明确偏好，或说"推荐几家""有什么好吃的"→ content_based_recommendation
  示例: "推荐几家餐厅" → content_based_recommendation()

如果用户未指定价格区间，默认不传 price_range 参数（会返回所有价位）。
可以组合调用多个工具获取更全面的结果，例如同时调 tag_based 和 content_based。

调用工具后，基于工具返回的 items 列表，用自然语言总结推荐结果，包含商户名、评分、人均消费等关键信息。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    result = _run_agent_loop(llm, messages, rec_tools, "recommendation", local_trace, local_tools_used)

    return {
        "recommendation_result": result,
        "trace": local_trace,
        "tools_used": local_tools_used
    }


# ============================================================
# 3. Analysis Agent (数据分析师)
#    工具: 业务概览, 区域分析, 热门排行
# ============================================================

def analysis_node(state: AgentState) -> Dict[str, Any]:
    """数据分析Agent: 处理业务数据查询和分析任务"""
    user_input = state["user_input"]
    local_trace = []
    local_tools_used = []

    llm = get_shared_llm()

    analysis_tools = [
        BusinessOverviewTool.get_schema(),
        AreaAnalysisTool.get_schema(),
        TopMerchantsTool.get_schema(),
    ]

    system_prompt = """你是数据分析师，擅长查询和分析餐饮业务数据。

【重要规则】你必须在回复用户前调用至少一个工具获取数据。禁止在未调用任何工具的情况下直接回答。即使问题宽泛，也要先调用最接近的工具获取数据，再基于结果分析。

你有以下工具:
1. get_business_overview - 获取业务整体数据（用户数、商户数、订单数、交易额），无需参数
2. get_area_analysis - 获取各区域商业分析数据，无需参数
3. get_top_merchants - 获取热门商户排行榜，无需参数

选择策略:
- "业务情况""整体数据""概览" → get_business_overview()
- "区域分析""各区域""地区" → get_area_analysis()
- "排行""热门""排名""top" → get_top_merchants()
- 宽泛问题（"看看数据"）→ 同时调用多个工具获取全面数据

调用工具后，基于返回的数据用自然语言总结分析结果，给出关键数字和洞察。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    result = _run_agent_loop(llm, messages, analysis_tools, "analysis", local_trace, local_tools_used)

    return {
        "analysis_result": result,
        "trace": local_trace,
        "tools_used": local_tools_used
    }


# ============================================================
# 4. Profile Agent (用户画像师)
#    工具: 用户偏好分析, 用户检查
# ============================================================

def profile_node(state: AgentState) -> Dict[str, Any]:
    """用户画像Agent: 分析用户偏好和行为特征"""
    user_input = state["user_input"]
    local_trace = []
    local_tools_used = []

    llm = get_shared_llm()

    profile_tools = [
        UserPreferencesTool.get_schema(),
        CheckUserTool.get_schema(),
    ]

    system_prompt = """你是用户画像分析专家，擅长分析用户偏好和行为特征。

【重要规则】你必须在回复用户前调用至少一个工具获取数据。禁止在未调用工具的情况下直接回答。

你有以下工具:
1. get_user_preferences - 分析用户群体的菜系偏好和价格区间偏好分布，无需参数
2. check_user - 检查特定用户是否存在（需要 user_id 参数）

选择策略:
- "用户偏好""用户画像""喜欢什么" → get_user_preferences()
- "检查用户""用户xxx是否存在" → check_user(user_id="xxx")

调用工具后，基于返回的数据用自然语言总结用户画像分析结果。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    result = _run_agent_loop(llm, messages, profile_tools, "profile", local_trace, local_tools_used)

    return {
        "profile_result": result,
        "trace": local_trace,
        "tools_used": local_tools_used
    }


# ============================================================
# Agent 执行循环 (ReAct within sub-agent)
# ============================================================

def _run_agent_loop(
    llm: LLMService,
    messages: List[Dict],
    tools: List[Dict],
    agent_name: str,
    trace: List[Dict],
    tools_used: List[str],
    max_iterations: int = 3
) -> Dict[str, Any]:
    """子Agent内部的ReAct循环"""
    iteration = 0
    final_response = None
    tool_results = []

    while iteration < max_iterations:
        iteration += 1
        llm_response = llm.chat(messages, tools=tools)

        if "tool_calls" in llm_response and llm_response["tool_calls"]:
            # 执行工具调用
            assistant_msg = {
                "role": "assistant",
                "content": llm_response.get("content"),
                "tool_calls": llm_response["tool_calls"]
            }
            messages.append(assistant_msg)

            for tool_call in llm_response["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                try:
                    tool_args = json.loads(tool_call["function"]["arguments"])
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    trace.append({
                        "iteration": iteration,
                        "phase": "action",
                        "agent": agent_name,
                        "tool_name": tool_name,
                        "error": f"参数解析失败: {e}"
                    })
                    continue

                # 记录工具调用
                trace.append({
                    "iteration": iteration,
                    "phase": "action",
                    "agent": agent_name,
                    "tool_name": tool_name,
                    "tool_args": tool_args
                })
                if tool_name not in tools_used:
                    tools_used.append(tool_name)

                # 执行工具
                tool_cls = get_tool_by_name(tool_name)
                if tool_cls:
                    try:
                        result = tool_cls.execute(**tool_args)
                        tool_results.append(result)
                        trace.append({
                            "iteration": iteration,
                            "phase": "observation",
                            "agent": agent_name,
                            "tool_name": tool_name,
                            "result_summary": str(result)[:200]
                        })
                    except Exception as e:
                        result = {"status": "error", "message": str(e)}
                        tool_results.append(result)
                        trace.append({
                            "iteration": iteration,
                            "phase": "observation",
                            "agent": agent_name,
                            "tool_name": tool_name,
                            "error": str(e)
                        })

                    # 添加observation到messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, ensure_ascii=False)
                    })
        else:
            # LLM 没有调用工具，直接返回了文本
            if iteration == 1 and not tool_results:
                # 首轮就没调工具 → 追加强制指令再试一次（而非直接退出）
                trace.append({
                    "iteration": iteration,
                    "phase": "no_tool_warning",
                    "agent": agent_name,
                    "message": "LLM 未调用工具，追加强制指令重试"
                })
                messages.append({
                    "role": "user",
                    "content": "【系统提示】你刚才没有调用任何工具。请立即调用一个合适的工具获取数据，不要直接回复文本。请从可用工具中选择一个调用。"
                })
                continue
            # 非首次 或 已有工具结果 → 接受为最终回复
            final_response = llm_response.get("content", "")
            trace.append({
                "iteration": iteration,
                "phase": "final",
                "agent": agent_name,
                "response": final_response[:200]
            })
            break

    if final_response is None:
        # 达到最大迭代，用已有结果生成回复
        final_response = _generate_summary_from_results(tool_results, agent_name)

    return {
        "agent": agent_name,
        "response": final_response,
        "tool_results": tool_results,
        "iterations": iteration
    }


def _generate_summary_from_results(results: List[Dict], agent_name: str) -> str:
    """从工具结果生成总结"""
    if not results:
        return f"{agent_name} Agent 未能获取到结果"

    summary_parts = []
    for result in results:
        if isinstance(result, dict):
            if result.get("result_type") == "recommendation":
                items = result.get("items", [])
                summary_parts.append(f"找到 {len(items)} 家推荐餐厅")
            elif result.get("result_type") == "statistics":
                data = result.get("data", {})
                summary_parts.append(f"业务数据: {json.dumps(data, ensure_ascii=False)}")
            elif result.get("result_type") == "area_analysis":
                areas = result.get("areas", [])
                summary_parts.append(f"区域分析: {len(areas)} 个区域")
            elif result.get("result_type") == "ranking":
                ranking = result.get("ranking", [])
                summary_parts.append(f"排行榜: {len(ranking)} 家商户")
            elif result.get("result_type") == "preference_analysis":
                cuisines = result.get("cuisine_preferences", [])
                summary_parts.append(f"偏好分析: {len(cuisines)} 种菜系偏好")

    return "; ".join(summary_parts) if summary_parts else "分析完成"


# ============================================================
# 5. Aggregator Node (结果聚合器)
#    聚合多个子Agent的结果，生成最终回复
# ============================================================

def aggregator_node(state: AgentState) -> Dict[str, Any]:
    """聚合器: 汇总各Agent的结果，生成最终回复"""
    local_trace = []
    tools_used = state.get("tools_used", [])

    rec_result = state.get("recommendation_result")
    analysis_result = state.get("analysis_result")
    profile_result = state.get("profile_result")
    intent = state.get("intent", "chat")

    # 如果是闲聊，直接返回
    if intent == "chat":
        llm = get_shared_llm()
        messages = [
            {"role": "system", "content": "你是一个友好的智能餐饮助手。"},
            {"role": "user", "content": state["user_input"]}
        ]
        response = llm.chat(messages, tools=None)
        final_response = response.get("content", "您好，有什么可以帮您的？")
        local_trace.append({"phase": "aggregator", "description": "闲聊回复"})
        return {
            "final_response": final_response,
            "trace": local_trace,
            "tools_used": tools_used
        }

    # 收集各Agent的回复
    agent_responses = []
    if rec_result and rec_result.get("response"):
        agent_responses.append({"agent": "推荐专家", "response": rec_result["response"]})
    if analysis_result and analysis_result.get("response"):
        agent_responses.append({"agent": "数据分析师", "response": analysis_result["response"]})
    if profile_result and profile_result.get("response"):
        agent_responses.append({"agent": "用户画像师", "response": profile_result["response"]})

    # 如果只有一个Agent的回复，直接使用
    if len(agent_responses) == 1:
        final_response = agent_responses[0]["response"]
    elif len(agent_responses) > 1:
        # 多个Agent结果需要LLM聚合
        llm = get_shared_llm()
        combine_prompt = "你是协调者，请将以下多个专家的分析结果整合成一个连贯、自然的回复。\n\n"
        for resp in agent_responses:
            combine_prompt += f"【{resp['agent']}】:\n{resp['response']}\n\n"
        combine_prompt += f"用户原始问题: {state['user_input']}\n\n请综合以上信息给出完整回复:"

        messages = [
            {"role": "system", "content": "你是智能餐饮平台的协调者，负责汇总各专家的分析结果。"},
            {"role": "user", "content": combine_prompt}
        ]
        response = llm.chat(messages, tools=None)
        final_response = response.get("content", "抱歉，暂时无法处理您的请求。")
    else:
        final_response = "抱歉，我暂时无法处理您的请求，请尝试更具体的描述。"

    local_trace.append({
        "phase": "aggregator",
        "description": f"聚合 {len(agent_responses)} 个Agent结果",
        "agents_involved": [r["agent"] for r in agent_responses]
    })

    return {
        "final_response": final_response,
        "trace": local_trace,
        "tools_used": tools_used
    }


# ============================================================
# 6. 构建 LangGraph StateGraph
# ============================================================

def build_multi_agent_graph(checkpointer=None):
    """
    构建多智能体协作图

    流程:
      用户输入 → Orchestrator (意图识别)
                     │
           ┌─────────┼─────────┐
           ▼         ▼         ▼
     Recommendation Analysis  Profile   (并行/单选)
           │         │         │
           └─────────┼─────────┘
                     ▼
               Aggregator (结果聚合)
                     │
                     ▼
                   输出

    Args:
        checkpointer: LangGraph checkpointer (如 MemorySaver)，启用后支持多轮对话状态持久化
    """
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("recommendation", recommendation_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("profile", profile_node)
    graph.add_node("aggregator", aggregator_node)

    # 设置入口
    graph.set_entry_point("orchestrator")

    # 条件路由: 从orchestrator路由到子Agent
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "recommendation": "recommendation",
            "analysis": "analysis",
            "profile": "profile",
            "aggregator": "aggregator"
        }
    )

    # 子Agent完成后都路由到聚合器
    graph.add_edge("recommendation", "aggregator")
    graph.add_edge("analysis", "aggregator")
    graph.add_edge("profile", "aggregator")

    # 聚合器到结束
    graph.add_edge("aggregator", END)

    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    return graph.compile(**compile_kwargs)


# ============================================================
# 7. 多智能体系统入口
# ============================================================

class MultiAgentSystem:
    """多智能体协作系统

    支持多轮对话:
    - 使用 LangGraph MemorySaver 作为 checkpointer，按 thread_id 持久化图状态
    - 维护 per-thread 对话历史，作为上下文注入 orchestrator 的意图识别
    """

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = build_multi_agent_graph(checkpointer=self.checkpointer)
        # 对话历史按 thread_id 隔离: {thread_id: [{"role","content"}, ...]}
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}

    def _get_history(self, thread_id: str) -> List[Dict[str, str]]:
        return self.conversation_history.setdefault(thread_id, [])

    def _build_contextualized_input(self, user_input: str, thread_id: str) -> str:
        """把对话历史拼进 user_input，让 orchestrator 有多轮上下文"""
        history = self._get_history(thread_id)
        if not history:
            return user_input
        # 取最近 4 轮（8 条消息）避免 prompt 过长
        recent = history[-8:]
        context_lines = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            context_lines.append(f"{role}: {msg['content'][:150]}")
        context = "\n".join(context_lines)
        return f"[对话历史]\n{context}\n\n[当前问题]\n{user_input}"

    def run(self, user_input: str, user_id: Optional[str] = None,
            thread_id: str = "default") -> Dict[str, Any]:
        """执行多智能体协作

        Args:
            user_input: 用户输入
            user_id: 用户ID（用于 LangFuse trace）
            thread_id: 会话线程ID，相同 thread_id 的请求共享对话上下文
        """
        # 把对话历史拼进输入（多轮上下文）
        contextualized_input = self._build_contextualized_input(user_input, thread_id)

        # 初始化状态（每轮重置 trace/tools/iterations，对话上下文通过 user_input 注入）
        initial_state = {
            "user_input": contextualized_input,
            "intent": "",
            "sub_tasks": [],
            "routed_agents": [],
            "recommendation_result": None,
            "analysis_result": None,
            "profile_result": None,
            "trace": [],
            "iterations": 0,
            "final_response": "",
            "tools_used": []
        }

        # checkpointer 配置: 按 thread_id 持久化图状态
        config = {"configurable": {"thread_id": thread_id}}

        # 执行图（包裹在 LangFuse trace 中，无配置时自动 no-op）
        with observe(name="multi_agent_run", user_id=user_id,
                     metadata={"input_preview": user_input[:200], "thread_id": thread_id}) as lf_trace:
            final_state = self.graph.invoke(initial_state, config=config)
            try:
                lf_trace.update(metadata={
                    "intent": final_state.get("intent", ""),
                    "routed_agents": final_state.get("routed_agents", []),
                    "iterations": final_state.get("iterations", 0),
                    "tools_used": list(dict.fromkeys(final_state.get("tools_used", []))),
                })
            except Exception:
                pass

        # 更新对话历史（存原始 user_input，不含历史拼接）
        history = self._get_history(thread_id)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": final_state["final_response"]})

        return {
            "response": final_state["final_response"],
            "trace": final_state.get("trace", []),
            "iterations": final_state.get("iterations", 0),
            "tools_used": list(dict.fromkeys(final_state.get("tools_used", []))),  # 去重
            "intent": final_state.get("intent", ""),
            "routed_agents": final_state.get("routed_agents", []),
            "thread_id": thread_id,
        }

    def clear_memory(self, thread_id: Optional[str] = None):
        """清除对话记忆。不传 thread_id 则清除所有线程。"""
        if thread_id is not None:
            self.conversation_history.pop(thread_id, None)
        else:
            self.conversation_history.clear()

    def run_stream(self, user_input: str, user_id: Optional[str] = None,
                   thread_id: str = "default"):
        """
        流式执行多智能体协作。

        流程:
          1. 正常运行 graph 获取 intent / trace / tools_used / 各 agent 结果
          2. 先 yield 元数据事件（intent / routed_agents / tools_used）
          3. 用 LLMService.stream_chat 流式重新生成最终聚合回复，逐 token yield

        Yields:
            dict: SSE 事件
              {"event": "meta", "data": {intent, routed_agents, tools_used}}
              {"event": "trace", "data": [...]}
              {"event": "token", "data": "..."}
              {"event": "done", "data": {iterations}}
              {"event": "error", "data": "..."}
        """
        try:
            result = self.run(user_input, user_id=user_id, thread_id=thread_id)
        except Exception as e:
            yield {"event": "error", "data": str(e)}
            return

        # 1. 元数据
        yield {
            "event": "meta",
            "data": {
                "intent": result.get("intent", ""),
                "routed_agents": result.get("routed_agents", []),
                "tools_used": result.get("tools_used", []),
            }
        }

        # 2. trace
        yield {"event": "trace", "data": result.get("trace", [])}

        # 3. 流式输出最终回复
        final_response = result.get("response", "")
        llm = get_shared_llm()

        # 构造聚合回复的 messages（与 aggregator 逻辑一致）
        # 直接用 stream_chat 重新生成，让用户看到逐 token 输出
        messages = [
            {"role": "system", "content": "你是智能餐饮推荐助手。请基于以下分析结果，用自然语言给用户一个连贯、完整的回复。"},
            {"role": "user", "content": f"用户问题: {user_input}\n\n分析结果:\n{final_response}\n\n请基于以上结果给用户一个完整回复:"}
        ]

        try:
            for token in llm.stream_chat(messages, tools=None):
                yield {"event": "token", "data": token}
        except Exception as e:
            # 流式失败 → 降级直接输出完整回复
            yield {"event": "token", "data": final_response}

        # 4. 完成
        yield {"event": "done", "data": {"iterations": result.get("iterations", 0)}}

    def get_stats(self) -> Dict[str, Any]:
        return {
            "available_tools": len(TOOLS),
            "active_threads": len(self.conversation_history),
            "total_messages": sum(len(h) for h in self.conversation_history.values()),
            "agents": ["orchestrator", "recommendation", "analysis", "profile", "aggregator"],
            "architecture": "LangGraph StateGraph + MemorySaver",
            "checkpointer": "MemorySaver (in-memory, per thread_id)"
        }
