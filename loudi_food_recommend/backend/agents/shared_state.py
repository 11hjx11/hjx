"""
多智能体共享状态定义
基于 LangGraph TypedDict 定义 Agent 间共享的数据结构
"""
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add


def merge_lists(left: List, right: List) -> List:
    """合并两个列表（用于并行节点的状态合并）"""
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right


def merge_dicts(left: Dict, right: Dict) -> Dict:
    """合并两个字典"""
    if left is None:
        left = {}
    if right is None:
        right = {}
    return {**left, **right}


class AgentState(TypedDict):
    """多智能体协作的共享状态"""
    # 用户输入
    user_input: str

    # 主控Agent分析结果
    intent: str
    routed_agents: List[str]

    # 各子Agent的执行结果（每个Agent只写自己的字段）
    recommendation_result: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]
    profile_result: Optional[Dict[str, Any]]

    # 并行节点共享的字段 - 使用 Annotated 合并器
    trace: Annotated[List[Dict[str, Any]], merge_lists]
    tools_used: Annotated[List[str], merge_lists]

    # 最终输出
    final_response: str
    iterations: int
