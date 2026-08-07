"""
LangFuse 可观测性封装

设计目标: 无配置时优雅降级（不抛错、不打扰主流程），有配置时自动上报 trace。
使用方式:
    from agents.observability import observe
    with observe(name="multi_agent_run", user_id=uid) as trace:
        ...  # 业务逻辑
        trace.generation(name="llm_call", input=..., output=..., model=...)

依赖:
    pip install langfuse
环境变量 (可选，未配置则自动禁用):
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    LANGFUSE_HOST  (默认 https://cloud.langfuse.com)
"""
import os
import contextvars
from contextlib import contextmanager
from typing import Any, Dict, Optional

try:
    from config import LLM_PROVIDER  # noqa: F401  仅用于触发 dotenv 加载
except ImportError:
    pass

# 当前生效的 trace 对象（线程/协程安全）
_current_trace: contextvars.ContextVar = contextvars.ContextVar("_langfuse_trace", default=None)

# ============================================================
# 延迟初始化 LangFuse 客户端
# ============================================================
_client = None
_client_initialized = False


def _get_client():
    """惰性初始化 LangFuse 客户端。无 key 时返回 None。"""
    global _client, _client_initialized
    if _client_initialized:
        return _client
    _client_initialized = True

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        # 未配置 → 静默禁用
        return None

    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception as e:  # 导入失败或初始化失败均不影响主流程
        print(f"[LangFuse] 初始化失败，已禁用可观测性: {e}")
        _client = None
    return _client


def is_enabled() -> bool:
    """是否启用 LangFuse 上报"""
    return _get_client() is not None


# ============================================================
# 兼容层: 提供 no-op 的占位 trace 对象
# ============================================================
class _NoopTrace:
    """未启用 LangFuse 时的空 trace，避免业务代码写 if 判断"""

    def generation(self, name: str, input: Any = None, output: Any = None,
                   model: Optional[str] = None, metadata: Optional[Dict] = None,
                   usage: Optional[Dict] = None):
        return _NoopSpan()

    def span(self, name: str, metadata: Optional[Dict] = None):
        return _NoopSpan()

    def update(self, **kwargs):
        pass

    def end(self):
        pass

    # 支持 with 语句
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
        return False


class _NoopSpan:
    def update(self, **kwargs):
        pass

    def end(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
        return False


# ============================================================
# 公共 API
# ============================================================
@contextmanager
def observe(name: str, user_id: Optional[str] = None,
            session_id: Optional[str] = None, metadata: Optional[Dict] = None):
    """
    开启一个 trace 上下文。

    用法:
        with observe("multi_agent_run", user_id="u1") as trace:
            trace.generation(name="llm", input=msg, output=resp, model="qwen-max")

    无 LangFuse 配置时返回 _NoopTrace，所有方法均为 no-op。
    """
    client = _get_client()
    if client is None:
        noop = _NoopTrace()
        yield noop
        return

    try:
        trace = client.trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )
    except Exception as e:
        # 上报失败不应影响业务
        print(f"[LangFuse] 创建 trace 失败，降级为 no-op: {e}")
        yield _NoopTrace()
        return

    # 设置当前 trace 上下文，供 LLMService 等内部组件上报 generation
    token = _current_trace.set(trace)
    try:
        yield trace
    finally:
        _current_trace.reset(token)
        try:
            trace.update(metadata={"status": "completed"})
        except Exception:
            pass


def get_current_trace():
    """获取当前上下文中生效的 trace（未启用时返回 None）"""
    return _current_trace.get()


def record_generation(name: str, input: Any = None, output: Any = None,
                      model: Optional[str] = None, metadata: Optional[Dict] = None,
                      usage: Optional[Dict] = None):
    """向当前 trace 上报一次 LLM generation（无 trace 时静默跳过）"""
    trace = _current_trace.get()
    if trace is None:
        return
    try:
        trace.generation(
            name=name,
            input=input,
            output=output,
            model=model,
            metadata=metadata or {},
            usage=usage,
        )
    except Exception as e:
        # 上报失败绝不影响业务
        print(f"[LangFuse] generation 上报失败: {e}")


def flush():
    """主动 flush 上报队列（进程退出前调用更稳妥）"""
    client = _get_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        pass
