# ============================================================
# 多智能体餐饮推荐系统 - Docker 镜像
# ============================================================
FROM python:3.11-slim

# 设置时区和基础工具
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 先装依赖（利用 Docker layer 缓存）
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码（backend + frontend）
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# 工作目录切到 backend（app.py 依赖 ../frontend/templates 相对路径）
WORKDIR /app/backend

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/agent/status', timeout=3)" || exit 1

CMD ["python", "app.py"]
