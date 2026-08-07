"""
RAG 语义检索工具 - SemanticSearchTool

把自然语言查询映射到商户语义检索（基于 bge-small-zh embedding + ChromaDB）。
与传统的基于菜系/价格标签的 tag_based_recommendation 互补:
  - tag_based: 严格匹配菜系和价格区间
  - semantic_search: 语义相似度匹配，处理模糊/自然语言描述（如"想吃辣的暖胃的"）

设计:
- 首次调用时自动把全量商户数据索引进向量库
- chromadb / sentence-transformers 不可用时返回明确的降级提示
"""
from typing import Dict, Any, List

from .tools import BaseTool
from .vector_store import get_vector_store
from utils.database import load_merchant_data


class SemanticSearchTool(BaseTool):
    """
    语义检索商户

    适用场景:
    - 用户用自然语言描述需求（如"想吃辣的""适合约会""暖胃的"）
    - 模糊查询，无法明确归类到某个菜系
    - 跨菜系的综合推荐
    """
    name = "semantic_search_merchants"
    description = (
        "基于语义相似度检索商户。适用于用户用自然语言描述需求的场景，"
        "如'想吃辣的暖胃的'、'适合约会的餐厅'。返回与查询语义最匹配的商户列表。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "自然语言查询，如'辣的暖胃的餐厅'、'适合约会的高档餐厅'"
            },
            "top_k": {
                "type": "integer",
                "description": "返回数量，默认5条",
                "default": 5
            }
        },
        "required": ["query"]
    }

    # 已索引标记（避免每次调用都重新索引）
    _indexed = False

    @classmethod
    def _ensure_indexed(cls):
        """首次调用时把商户数据索引进向量库"""
        if cls._indexed:
            return
        vs = get_vector_store()
        if vs is None:
            return
        try:
            merchant_df = load_merchant_data()
            rows = merchant_df.to_dict("records")
            added = vs.index_merchants(rows)
            if added > 0:
                print(f"[RAG] 已索引 {added} 家商户到向量库")
            cls._indexed = True
        except Exception as e:
            print(f"[RAG] 商户索引失败: {e}")

    @classmethod
    def execute(cls, query: str, top_k: int = 5) -> Dict[str, Any]:
        cls._ensure_indexed()
        vs = get_vector_store()

        # 向量库不可用 → 降级返回提示
        if vs is None or not vs._ensure_initialized():
            return {
                "tool": cls.name,
                "result_type": "semantic_search",
                "enabled": False,
                "message": "语义检索未启用（chromadb / sentence-transformers 未安装），请使用 tag_based_recommendation 或 content_based_recommendation",
                "items": [],
                "total_found": 0,
            }

        results = vs.search(query, top_k=top_k)

        formatted = []
        for item in results:
            formatted.append({
                "merchant_name": item.get("merchant_name", ""),
                "score": item.get("score", 0),
                "total_orders": item.get("total_orders", 0),
                "avg_order_amount": item.get("avg_order_amount", 0),
                "area": item.get("area", ""),
                "similarity": item.get("similarity"),
                "recommendation_reason": f"语义匹配查询「{query}」"
            })

        return {
            "tool": cls.name,
            "result_type": "semantic_search",
            "enabled": True,
            "query": query,
            "total_found": len(formatted),
            "items": formatted,
        }
