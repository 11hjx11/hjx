"""
向量存储模块 - 基于 ChromaDB + sentence-transformers (bge-small-zh)

设计目标:
1. 懒加载: 首次查询时才初始化 embedding 模型和索引商户数据
2. 优雅降级: chromadb / sentence-transformers 未安装时返回 None，调用方自行回退
3. 持久化: ChromaDB 数据存到本地 .chroma_db 目录，避免每次重启都重新 embedding

商户文档构造:
    "{merchant_name} | 菜系:{cuisine} | 区域:{area} | 评分:{score} | 人均:{avg_order_amount}元 | 销量:{total_orders}单"
"""
import os
from typing import Any, Dict, List, Optional

# ChromaDB 持久化路径（相对 backend 目录）
_DB_DIR = os.path.join(os.path.dirname(__file__), "..", ".chroma_db")
_COLLECTION_NAME = "merchants"
_EMBED_MODEL = "BAAI/bge-small-zh"

# 懒加载单例
_vector_store_instance = None


def _build_merchant_doc(row: Dict) -> str:
    """把商户行构造成可检索的文本"""
    name = row.get("merchant_name", "")
    area = row.get("area", "")
    score = row.get("score", "")
    orders = row.get("total_orders", "")
    avg = row.get("avg_order_amount", "")
    # 菜系信息不在商户表里，但商户名常含菜系关键词（如"川味居"→川菜）
    return f"{name} | 区域:{area} | 评分:{score} | 人均:{avg}元 | 销量:{orders}单"


class VectorStore:
    """ChromaDB 向量存储封装"""

    def __init__(self):
        self._initialized = False
        self._collection = None
        self._embed_fn = None
        self._indexed_count = 0

    def _ensure_initialized(self) -> bool:
        """惰性初始化。成功返回 True，依赖缺失/失败返回 False。"""
        if self._initialized:
            return self._collection is not None
        self._initialized = True

        try:
            import chromadb
        except ImportError:
            print("[VectorStore] chromadb 未安装，RAG 语义检索已禁用")
            return False

        try:
            from chromadb.config import Settings
            client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=os.path.abspath(_DB_DIR),
            ))
            self._collection = client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            # chromadb 新版本 API 变动或其他初始化失败
            print(f"[VectorStore] ChromaDB 初始化失败: {e}")
            try:
                # 尝试新版 API（无 chroma_db_impl 参数）
                import chromadb
                client = chromadb.PersistentClient(path=os.path.abspath(_DB_DIR))
                self._collection = client.get_or_create_collection(
                    name=_COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e2:
                print(f"[VectorStore] ChromaDB 新版 API 也失败: {e2}")
                self._collection = None
                return False

        try:
            from sentence_transformers import SentenceTransformer
            self._embed_fn = SentenceTransformer(_EMBED_MODEL)
        except Exception as e:
            print(f"[VectorStore] sentence-transformers 加载失败 ({_EMBED_MODEL}): {e}")
            print("[VectorStore] 尝试用 ChromaDB 内置默认 embedding 作为回退")
            self._embed_fn = None  # 让 ChromaDB 用默认 embedding

        return self._collection is not None

    def index_merchants(self, merchant_rows: List[Dict[str, Any]]) -> int:
        """
        把商户数据索引进 ChromaDB。
        已存在的 document 不会重复索引（按 merchant_id 去重）。

        Args:
            merchant_rows: [{"merchant_id": ..., "merchant_name": ..., ...}, ...]
        Returns:
            实际新增的文档数
        """
        if not self._ensure_initialized() or not merchant_rows:
            return 0

        # 查询已有 id，避免重复 embedding
        existing_ids = set()
        try:
            existing = self._collection.get()
            existing_ids = set(existing.get("ids", []))
        except Exception:
            pass

        new_docs, new_ids, new_metadatas = [], [], []
        for row in merchant_rows:
            mid = str(row.get("merchant_id", ""))
            if not mid or mid in existing_ids:
                continue
            doc = _build_merchant_doc(row)
            new_docs.append(doc)
            new_ids.append(mid)
            new_metadatas.append({
                "merchant_id": mid,
                "merchant_name": str(row.get("merchant_name", "")),
                "area": str(row.get("area", "")),
                "score": float(row.get("score", 0) or 0),
                "total_orders": int(row.get("total_orders", 0) or 0),
                "avg_order_amount": float(row.get("avg_order_amount", 0) or 0),
            })

        if not new_docs:
            return 0

        # 计算 embeddings（若有 sentence-transformers）
        embeddings = None
        if self._embed_fn is not None:
            try:
                embeddings = self._embed_fn.encode(new_docs).tolist()
            except Exception as e:
                print(f"[VectorStore] embedding 编码失败，回退到 ChromaDB 默认: {e}")
                embeddings = None

        try:
            if embeddings is not None:
                self._collection.add(
                    documents=new_docs,
                    ids=new_ids,
                    metadatas=new_metadatas,
                    embeddings=embeddings,
                )
            else:
                # ChromaDB 默认 embedding 模型（需联网下载，可能失败）
                self._collection.add(
                    documents=new_docs,
                    ids=new_ids,
                    metadatas=new_metadatas,
                )
        except Exception as e:
            print(f"[VectorStore] 索引写入失败: {e}")
            return 0

        self._indexed_count += len(new_docs)
        return len(new_docs)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索 top_k 最相关商户。

        Returns:
            [{"merchant_id", "merchant_name", "area", "score",
              "total_orders", "avg_order_amount", "distance"}]
            依赖不可用时返回空列表。
        """
        if not self._ensure_initialized():
            return []

        try:
            if self._embed_fn is not None:
                query_embedding = self._embed_fn.encode([query]).tolist()
                results = self._collection.query(
                    query_embeddings=query_embedding,
                    n_results=top_k,
                )
            else:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=top_k,
                )
        except Exception as e:
            print(f"[VectorStore] 查询失败: {e}")
            return []

        return self._format_results(results)

    @staticmethod
    def _format_results(results: Dict) -> List[Dict[str, Any]]:
        """把 ChromaDB 返回结构展平为商户列表"""
        if not results or not results.get("ids"):
            return []

        ids = results["ids"][0] if results["ids"] else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        out = []
        for i, mid in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else None
            out.append({
                "merchant_id": mid,
                "merchant_name": meta.get("merchant_name", ""),
                "area": meta.get("area", ""),
                "score": meta.get("score", 0),
                "total_orders": meta.get("total_orders", 0),
                "avg_order_amount": meta.get("avg_order_amount", 0),
                # distance 越小越相似（cosine 距离）
                "similarity": round(1 - dist, 4) if dist is not None else None,
            })
        return out

    def get_stats(self) -> Dict[str, Any]:
        """返回当前索引状态"""
        if not self._ensure_initialized():
            return {"enabled": False, "indexed_count": 0}
        count = 0
        try:
            count = self._collection.count()
        except Exception:
            count = self._indexed_count
        return {
            "enabled": True,
            "indexed_count": count,
            "embed_model": _EMBED_MODEL if self._embed_fn is not None else "chromadb-default",
            "persist_dir": os.path.abspath(_DB_DIR),
        }


def get_vector_store() -> Optional[VectorStore]:
    """获取全局 VectorStore 单例"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
