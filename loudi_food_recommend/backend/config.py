# ============================================================
# LLM Configuration - 大模型配置文件
# 支持从 .env 文件或环境变量读取敏感信息
# ============================================================
import os

# 尝试从 .env 文件加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass  # python-dotenv 未安装时使用系统环境变量

# LLM Provider: "mock" 使用模拟LLM, "qwen" 使用通义千问API
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")

# 通义千问 (Qwen) API配置
QWEN_CONFIG = {
    "api_key": os.getenv("QWEN_API_KEY", ""),
    "model": os.getenv("QWEN_MODEL", "qwen-max"),
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
}

# OpenAI API配置 (可选)
OPENAI_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "model": os.getenv("OPENAI_MODEL", "gpt-4"),
    "api_url": "https://api.openai.com/v1/chat/completions"
}

# 数据库配置 (可选)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "192.168.88.128"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "loudi_food"),
}

# ============================================================
# 使用说明
# ============================================================
# 1. 复制 .env.example 为 .env:  cp .env.example .env
# 2. 在 .env 中填写真实的 API Key
# 3. 设置 LLM_PROVIDER=qwen 使用通义千问，或 LLM_PROVIDER=mock 使用模拟
# 4. model可选值: qwen-max, qwen-plus, qwen-turbo, qwen3-max
