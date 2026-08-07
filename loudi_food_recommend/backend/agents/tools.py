from typing import Dict, Any, List
from utils.database import load_user_data, load_merchant_data, load_interaction_data, load_area_data, load_business_overview
from utils.recommendation import create_user_merchant_matrix, collaborative_filtering, tag_based_recommendation, content_based_recommendation, ensure_numeric

# ============================================================
# Tool Definitions for LLM Function Calling
# Each tool follows OpenAI Function Calling format
# ============================================================

class BaseTool:
    """Base class for all tools"""
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {}
    
    @classmethod
    def execute(cls, **kwargs) -> Any:
        raise NotImplementedError
    
    @classmethod
    def get_schema(cls) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": cls.parameters
            }
        }

class CollaborativeFilteringTool(BaseTool):
    """
    基于用户历史行为的协同过滤推荐
    当用户提供用户ID，且该用户有历史交互记录时使用
    """
    name = "collaborative_filtering"
    description = "基于用户历史行为和相似用户偏好推荐餐厅。适用于已有历史记录的老用户。需要提供用户ID。"
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "用户ID，系统中的唯一用户标识"
            }
        },
        "required": ["user_id"]
    }

    @classmethod
    def execute(cls, user_id: str) -> Dict[str, Any]:
        interaction_df = load_interaction_data()
        merchant_df = load_merchant_data()
        user_merchant_matrix = create_user_merchant_matrix(interaction_df)
        
        recommendations = collaborative_filtering(user_id, user_merchant_matrix, interaction_df, merchant_df)
        result_list = recommendations.to_dict('records')
        
        # Format result with readable information
        formatted_result = []
        for item in result_list:
            formatted_result.append({
                "merchant_name": item.get("merchant_name", "未知餐厅"),
                "score": item.get("score", 0),
                "total_orders": item.get("total_orders", 0),
                "avg_order_amount": item.get("avg_order_amount", 0),
                "area": item.get("area", ""),
                "recommendation_reason": "基于与您口味相似的用户偏好推荐"
            })
        
        return {
            "tool": cls.name,
            "result_type": "recommendation",
            "total_found": len(formatted_result),
            "items": formatted_result
        }

class TagBasedTool(BaseTool):
    """
    基于用户偏好标签的推荐
    当用户明确表达菜系偏好或价格预算时使用
    """
    name = "tag_based_recommendation"
    description = "根据用户指定的菜系和价格区间推荐餐厅。适用于新用户或有明确偏好的用户。"
    parameters = {
        "type": "object",
        "properties": {
            "cuisine": {
                "type": "string",
                "description": "菜系类型，如：湘菜、川菜、粤菜、鲁菜、火锅、烧烤、西餐、日料、韩料等"
            },
            "price_range": {
                "type": "string",
                "description": "价格区间：经济型（人均50以下）、中档（人均50-100）、高档（人均100以上）",
                "enum": ["经济型", "中档", "高档"]
            }
        },
        "required": ["cuisine", "price_range"]
    }

    @classmethod
    def execute(cls, cuisine: str, price_range: str) -> Dict[str, Any]:
        merchant_df = load_merchant_data()
        recommendations = tag_based_recommendation(cuisine, price_range, merchant_df)
        result_list = recommendations.to_dict('records')
        
        formatted_result = []
        for item in result_list:
            formatted_result.append({
                "merchant_name": item.get("merchant_name", "未知餐厅"),
                "score": item.get("score", 0),
                "total_orders": item.get("total_orders", 0),
                "avg_order_amount": item.get("avg_order_amount", 0),
                "area": item.get("area", ""),
                "recommendation_reason": f"符合您对{cuisine}菜系、{price_range}价位的偏好"
            })
        
        return {
            "tool": cls.name,
            "result_type": "recommendation",
            "total_found": len(formatted_result),
            "items": formatted_result,
            "filters_applied": {
                "cuisine": cuisine,
                "price_range": price_range
            }
        }

class ContentBasedTool(BaseTool):
    """
    基于内容的推荐（热门推荐）
    当用户没有明确偏好，或其他推荐方式无法生成结果时使用
    """
    name = "content_based_recommendation"
    description = "基于餐厅评分和销量的热门推荐。适用于用户没有明确偏好或不知道选什么时的默认推荐。"
    parameters = {
        "type": "object",
        "properties": {
            "n_recommendations": {
                "type": "integer",
                "description": "推荐数量，默认5条",
                "default": 5
            }
        }
    }

    @classmethod
    def execute(cls, n_recommendations: int = 5) -> Dict[str, Any]:
        merchant_df = load_merchant_data()
        recommendations = content_based_recommendation(merchant_df, n_recommendations)
        result_list = recommendations.to_dict('records')
        
        formatted_result = []
        for item in result_list:
            formatted_result.append({
                "merchant_name": item.get("merchant_name", "未知餐厅"),
                "score": item.get("score", 0),
                "total_orders": item.get("total_orders", 0),
                "avg_order_amount": item.get("avg_order_amount", 0),
                "area": item.get("area", ""),
                "recommendation_reason": "该餐厅评分高、人气旺，是热门选择"
            })
        
        return {
            "tool": cls.name,
            "result_type": "recommendation",
            "total_found": len(formatted_result),
            "items": formatted_result,
            "sort_by": "score_and_popularity"
        }

class BusinessOverviewTool(BaseTool):
    """
    业务概览数据查询
    """
    name = "get_business_overview"
    description = "获取业务整体数据概览，包括总用户数、总商户数、总订单数、总交易额等统计信息。"
    parameters = {
        "type": "object",
        "properties": {}
    }

    @classmethod
    def execute(cls) -> Dict[str, Any]:
        business_df = load_business_overview()
        data = business_df.to_dict('records')[0]
        
        return {
            "tool": cls.name,
            "result_type": "statistics",
            "data": {
                "total_users": data.get("total_users", 0),
                "total_merchants": data.get("total_merchants", 0),
                "total_orders": data.get("total_orders", 0),
                "total_amount": data.get("total_amount", 0),
                "avg_order_amount": data.get("avg_order_amount", 0),
                "avg_merchant_score": data.get("avg_merchant_score", 0)
            }
        }

class AreaAnalysisTool(BaseTool):
    """
    区域商业分析
    """
    name = "get_area_analysis"
    description = "获取各区域的商业分析数据，包括每个区域的商户数、订单数、销售额等。"
    parameters = {
        "type": "object",
        "properties": {}
    }

    @classmethod
    def execute(cls) -> Dict[str, Any]:
        area_df = load_area_data()
        areas = area_df.to_dict('records')
        
        formatted_areas = []
        for area in areas:
            formatted_areas.append({
                "area": area.get("area", "未知区域"),
                "total_merchants": area.get("total_merchants", 0),
                "total_orders": area.get("total_orders", 0),
                "total_amount": area.get("total_amount", 0),
                "avg_merchant_score": area.get("avg_merchant_score", 0)
            })
        
        return {
            "tool": cls.name,
            "result_type": "area_analysis",
            "total_areas": len(formatted_areas),
            "areas": formatted_areas
        }

class TopMerchantsTool(BaseTool):
    """
    热门商户排行
    """
    name = "get_top_merchants"
    description = "获取评分最高和销量最好的热门商户排行榜。"
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "返回数量，默认10家",
                "default": 10
            }
        }
    }

    @classmethod
    def execute(cls, limit: int = 10) -> Dict[str, Any]:
        merchant_df = load_merchant_data()
        merchant_df = ensure_numeric(merchant_df, ['score', 'total_orders'])
        top_merchants = merchant_df.sort_values(['score', 'total_orders'], ascending=False).head(limit)
        result_list = top_merchants.to_dict('records')
        
        formatted_result = []
        for rank, item in enumerate(result_list, 1):
            formatted_result.append({
                "rank": rank,
                "merchant_name": item.get("merchant_name", "未知餐厅"),
                "score": item.get("score", 0),
                "total_orders": item.get("total_orders", 0),
                "avg_order_amount": item.get("avg_order_amount", 0),
                "area": item.get("area", "")
            })
        
        return {
            "tool": cls.name,
            "result_type": "ranking",
            "total_results": len(formatted_result),
            "ranking": formatted_result
        }

class UserPreferencesTool(BaseTool):
    """
    用户偏好分析
    """
    name = "get_user_preferences"
    description = "分析用户群体的偏好分布，包括菜系偏好和价格区间偏好。"
    parameters = {
        "type": "object",
        "properties": {}
    }

    @classmethod
    def execute(cls) -> Dict[str, Any]:
        user_df = load_user_data()
        cuisine_dist = user_df['preferred_cuisine'].value_counts().to_dict()
        price_dist = user_df['preferred_price_range'].value_counts().to_dict()
        
        # Format cuisine distribution
        cuisine_preferences = [
            {"cuisine": k, "user_count": v}
            for k, v in cuisine_dist.items()
        ]
        
        # Format price distribution
        price_preferences = [
            {"price_range": k, "user_count": v}
            for k, v in price_dist.items()
        ]
        
        return {
            "tool": cls.name,
            "result_type": "preference_analysis",
            "cuisine_preferences": cuisine_preferences,
            "price_preferences": price_preferences
        }

class CheckUserTool(BaseTool):
    """
    用户身份检查
    """
    name = "check_user"
    description = "检查指定的用户ID是否存在于系统中，用于判断是新用户还是老用户。"
    parameters = {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "要检查的用户ID"
            }
        },
        "required": ["user_id"]
    }

    @classmethod
    def execute(cls, user_id: str) -> Dict[str, Any]:
        user_df = load_user_data()
        user_exists = user_id in user_df['user_id'].values
        
        if user_exists:
            user_data = user_df[user_df['user_id'] == user_id].iloc[0]
            return {
                "tool": cls.name,
                "user_found": True,
                "user_info": {
                    "user_id": user_id,
                    "preferred_cuisine": user_data.get("preferred_cuisine", ""),
                    "preferred_price_range": user_data.get("preferred_price_range", ""),
                    "age": user_data.get("age", 0),
                    "gender": user_data.get("gender", "")
                }
            }
        else:
            return {
                "tool": cls.name,
                "user_found": False,
                "message": f"用户 {user_id} 不存在，可能是新用户"
            }

class RecommendationSummaryTool(BaseTool):
    """
    推荐结果总结
    """
    name = "summarize_recommendations"
    description = "将多个推荐结果进行整合和总结，生成最终的推荐列表。"
    parameters = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object"
                },
                "description": "需要整合的推荐结果列表"
            }
        },
        "required": ["results"]
    }

    @classmethod
    def execute(cls, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Deduplicate and merge recommendations
        all_items = []
        seen_merchants = set()

        for result in results:
            items = result.get("items", [])
            for item in items:
                merchant_name = item.get("merchant_name", "")
                if merchant_name not in seen_merchants:
                    seen_merchants.add(merchant_name)
                    all_items.append(item)

        # Sort by score
        all_items.sort(key=lambda x: x.get("score", 0), reverse=True)

        return {
            "tool": cls.name,
            "result_type": "summary",
            "total_unique_recommendations": len(all_items),
            "merged_recommendations": all_items[:10]  # Return top 10
        }


# 延迟导入 SemanticSearchTool 避免循环依赖（rag_tool 依赖本模块的 BaseTool）
def _get_semantic_search_tool():
    from .rag_tool import SemanticSearchTool
    return SemanticSearchTool


# ============================================================
# Tool Registry
# ============================================================

def _build_tool_registry():
    """构建工具注册表（包含可选的 RAG 工具）"""
    base_tools = [
        CollaborativeFilteringTool,
        TagBasedTool,
        ContentBasedTool,
        BusinessOverviewTool,
        AreaAnalysisTool,
        TopMerchantsTool,
        UserPreferencesTool,
        CheckUserTool,
        RecommendationSummaryTool,
    ]
    # RAG 工具依赖 chromadb + sentence-transformers，缺失时仍能注册（执行时降级）
    try:
        base_tools.append(_get_semantic_search_tool())
    except Exception as e:
        print(f"[Tools] SemanticSearchTool 注册失败: {e}")
    return base_tools


TOOLS = _build_tool_registry()

# Tool name to class mapping
TOOL_MAP = {tool.name: tool for tool in TOOLS}

def get_tool_by_name(name: str) -> BaseTool:
    """Get tool class by name"""
    return TOOL_MAP.get(name)

def get_tools_schema() -> List[Dict[str, Any]]:
    """Get all tool schemas for LLM function calling"""
    return [tool.get_schema() for tool in TOOLS]

def get_available_tools_info() -> List[Dict[str, str]]:
    """Get basic information about all available tools"""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        }
        for tool in TOOLS
    ]


# ============================================================
# Skill Registry - Skill 工程化: 工具注册/查询/组合
# ============================================================

class SkillRegistry:
    """Skill 注册中心: 支持工具动态注册、查询、组合为复合 Skill

    设计模式:
    - 每个 Skill 封装一组相关工具，对外暴露统一接口
    - 支持条件加载: 依赖缺失的工具自动跳过注册
    - 支持 Skill 组合: 多个工具可打包为业务语义的复合 Skill
    """

    def __init__(self):
        self._skills: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        tools: List[str],
        category: str = "general",
        description: str = "",
        version: str = "1.0.0"
    ) -> None:
        """注册 Skill: 将一组工具打包为可复用的业务 Skill"""
        self._skills[name] = {
            "name": name,
            "tools": tools,
            "category": category,
            "description": description,
            "version": version
        }

    def get_skill(self, name: str) -> Dict[str, Any]:
        """获取指定 Skill 的元数据"""
        return self._skills.get(name)

    def get_skill_tools(self, name: str) -> List[Dict[str, Any]]:
        """获取 Skill 包含的所有工具 schema（用于 LLM Function Calling）"""
        skill = self._skills.get(name)
        if not skill:
            return []
        tools = []
        for tool_name in skill["tools"]:
            tool_cls = get_tool_by_name(tool_name)
            if tool_cls:
                tools.append(tool_cls.get_schema())
        return tools

    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有已注册的 Skill"""
        return [
            {
                "name": s["name"],
                "category": s["category"],
                "tools_count": len(s["tools"]),
                "description": s["description"]
            }
            for s in self._skills.values()
        ]

    def list_by_category(self) -> Dict[str, List[str]]:
        """按类别分组列出 Skill"""
        categories: Dict[str, List[str]] = {}
        for name, s in self._skills.items():
            categories.setdefault(s["category"], []).append(name)
        return categories

    @property
    def skill_count(self) -> int:
        return len(self._skills)


# ============================================================
# 预设 Skill 定义（基于现有工具组合）
# ============================================================

def _build_default_skills(registry: SkillRegistry) -> None:
    """构建默认 Skill 注册"""
    registry.register(
        name="recommendation",
        tools=[
            "collaborative_filtering",
            "tag_based_recommendation",
            "content_based_recommendation",
            "semantic_search_merchants",
        ],
        category="recommendation",
        description="餐厅推荐 Skill: 协同过滤 + 标签 + 内容 + 语义检索四路混合推荐",
        version="1.0.0"
    )
    registry.register(
        name="analysis",
        tools=[
            "get_business_overview",
            "get_area_analysis",
            "get_top_merchants",
        ],
        category="analysis",
        description="业务分析 Skill: 数据概览 + 区域分析 + 热门排行",
        version="1.0.0"
    )
    registry.register(
        name="profile",
        tools=[
            "get_user_preferences",
            "check_user",
        ],
        category="profile",
        description="用户画像 Skill: 偏好分析 + 身份检查",
        version="1.0.0"
    )
    registry.register(
        name="aggregation",
        tools=["summarize_recommendations"],
        category="aggregation",
        description="结果聚合 Skill: 多推荐结果去重合并",
        version="1.0.0"
    )
    registry.register(
        name="full_service",
        tools=[
            "collaborative_filtering",
            "tag_based_recommendation",
            "content_based_recommendation",
            "semantic_search_merchants",
            "get_business_overview",
            "get_area_analysis",
            "get_top_merchants",
            "get_user_preferences",
            "check_user",
            "summarize_recommendations",
        ],
        category="all",
        description="全功能 Skill: 所有 10 个工具的完整组合",
        version="1.0.0"
    )


# 全局 Skill 注册中心
SKILL_REGISTRY = SkillRegistry()
_build_default_skills(SKILL_REGISTRY)