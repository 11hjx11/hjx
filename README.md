# 智能生活服务推荐系统

基于 Agent 架构的智能餐饮推荐平台，通过自然语言对话为用户推荐符合口味与消费能力的餐厅。采用 LangGraph 多智能体协作 + 通义千问（Qwen）Function Calling，支持意图识别、工具编排与多轮对话。

## 功能特性

- **自然语言对话推荐**：用户用一句话描述需求，Agent 自动识别意图并选择推荐策略
- **多智能体协作**：Orchestrator 路由分发到「推荐专家 / 数据分析师 / 用户画像师」并行处理，Aggregator 聚合结果
- **多种推荐算法**：协同过滤（基于用户历史）、标签推荐（菜系+价格）、内容推荐（评分+销量）、**RAG 语义检索**（bge-small-zh + ChromaDB）
- **SSE 流式输出**：`/chat/stream` 端点逐 token 返回回复，前端可实时渲染打字效果
- **多轮对话记忆**：LangGraph MemorySaver 按 `thread_id` 持久化图状态 + 对话历史上下文注入
- **可观测性**：LangFuse 集成，自动上报 trace / generation / token usage（无配置时静默禁用）
- **评测框架**：`eval/` 目录含 25 条用例 + `run_eval.py` 脚本，输出准确率 / 工具覆盖率 / P95 延迟
- **数据可视化**：业务概览、区域分析、热门商户排行、用户偏好分布
- **反馈收集**：用户对推荐结果反馈，持续优化
- **Docker 一键部署**：`docker compose up` 启动 app + mysql 两个服务
- **容错设计**：数据库/LLM/向量库/LangFuse 不可用时自动回退，服务始终保持可运行

## 技术栈

- **后端**：Python 3.11、FastAPI、SQLAlchemy、Pandas、scikit-learn
- **Agent**：LangGraph StateGraph + MemorySaver、ReAct 循环、Function Calling
- **LLM**：通义千问 Qwen（OpenAI 兼容模式），支持流式 SSE + mock 模式
- **RAG**：ChromaDB 向量库 + sentence-transformers（bge-small-zh embedding）
- **可观测性**：LangFuse（trace / generation / token 用量上报）
- **数据库**：MySQL（存储用户/商户/交互数据）
- **前端**：Jinja2 模板 + HTML/CSS
- **部署**：Docker + docker-compose（app + mysql）
- **数据工程**：Hive（ODS/DWD/DWS/ADS 分层）、Sqoop（Hive↔MySQL 同步）

## 项目结构

```
loudi_food_recommend/
├── backend/
│   ├── app.py                # FastAPI 入口（含 /chat/stream SSE 端点）
│   ├── config.py             # 配置（从 .env 读取）
│   ├── routers/api.py        # 数据 API
│   ├── utils/
│   │   ├── database.py       # 数据库加载（带 Mock 回退）
│   │   ├── recommendation.py # 推荐算法（协同过滤/标签/内容）
│   │   └── feedback.py       # 反馈存储
│   ├── agents/
│   │   ├── multi_agent.py    # LangGraph 多智能体协作 + MemorySaver
│   │   ├── agent.py          # 单 Agent ReAct 引擎
│   │   ├── llm_service.py    # LLM 服务（Qwen + Mock + 流式 stream_chat）
│   │   ├── tools.py          # 10 个工具注册表（含 RAG）
│   │   ├── shared_state.py   # 多智能体共享状态（TypedDict）
│   │   ├── rag_tool.py       # SemanticSearchTool（语义检索工具）
│   │   ├── vector_store.py   # ChromaDB + bge-small-zh 向量库封装
│   │   └── observability.py  # LangFuse 可观测性封装（无配置时 no-op）
│   ├── eval/
│   │   ├── eval_set.json     # 25 条评测用例
│   │   └── run_eval.py       # 评测脚本（accuracy/latency/coverage）
│   └── tests/                # pytest 单元测试
├── frontend/templates/       # HTML 页面（首页/对话/可视化/推荐/标签选择）
├── data/                     # 原始与清洗后数据
├── scripts/                  # 数据处理 / Hive / MySQL 脚本
├── Dockerfile                # 应用镜像
├── docker-compose.yml        # app + mysql 编排
└── requirements.txt
```

## 快速开始

### 方式一：Docker 一键启动（推荐）

```bash
cd loudi_food_recommend
# 在 .env 中配置 QWEN_API_KEY（必须），其余可选项见 .env.example
docker compose up -d
```

访问 http://localhost:8000

### 方式二：本地运行

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r loudi_food_recommend/requirements.txt
```

复制 `loudi_food_recommend/backend/.env.example` 为 `.env`，填写：

```env
LLM_PROVIDER=qwen               # 或 mock（无需 API Key）
QWEN_API_KEY=sk-xxx
QWEN_MODEL=qwen-max
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your-password
DB_NAME=loudi_food
# 可选：LangFuse 可观测性
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

```bash
cd loudi_food_recommend/backend
python app.py
```

访问 http://localhost:8000

> 未配置数据库时自动使用 Mock 数据；未配置 API Key 时 LLM 回退到 Mock；未配置 LangFuse 时可观测性自动禁用；未安装 chromadb 时 RAG 工具返回降级提示。

## 核心接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 首页（输入用户 ID） |
| POST | `/check_user` | 检查用户存在性：老用户返回推荐列表，新用户跳转标签选择 |
| POST | `/tag_selection` | 新用户提交菜系与价格偏好，返回标签推荐列表 |
| POST | `/submit_feedback` | 提交对推荐结果的满意度反馈 |
| GET | `/chat` | Agent 对话页面 |
| POST | `/chat` | 自然语言对话推荐（参数 `message` / `thread_id` / `user_id`） |
| POST | `/chat/stream` | **SSE 流式对话**，逐 token 返回（事件: meta / trace / token / done） |
| GET | `/visualization` | 数据可视化页面 |
| GET | `/agent/status` | 多智能体系统状态（工具数、活跃线程数、架构） |
| GET | `/agent/tools` | 列出 10 个可用工具的 schema |
| POST | `/agent/reset` | 清空对话记忆（可选参数 `thread_id` 指定线程） |
| GET | `/api/data/overview` | 业务概览数据 |
| GET | `/api/data/areas` | 区域分析数据 |
| GET | `/api/data/top_merchants` | 热门商户 Top 10 |
| GET | `/api/data/user_preferences` | 用户偏好分布 |

## 约束说明

- **运行目录**：本地运行时必须 `cd loudi_food_recommend/backend` 后执行 `python app.py`（Docker 部署无此限制）。
- **LLM 模型名**：通义千问有效模型名为 `qwen-max` / `qwen-plus` / `qwen-turbo`。默认 `qwen-max`。
- **数据库日期分区**：`utils/database.py` 中所有 ADS 层查询硬编码 `dt = '2026-02-14'`，切换数据日期需修改源码。
- **DB / LLM / RAG / LangFuse 容错**：所有外部依赖均有降级机制——数据库失败用 Mock 数据，LLM 失败用 Mock LLM，chromadb 未装时 RAG 工具返回提示，LangFuse 未配置时 trace 静默跳过。

## Agent 架构

```
用户输入 → Orchestrator（意图识别 + 对话历史上下文）
              ├─→ Recommendation Agent（协同过滤 / 标签 / 内容 / RAG 语义检索）
              ├─→ Analysis Agent（业务概览 / 区域 / 排行）
              └─→ Profile Agent（用户偏好分析）
                    ↓
              Aggregator（结果聚合）→ 最终回复（可流式输出）
```

- **MemorySaver**：按 `thread_id` 持久化图状态，支持多轮对话（最近 4 轮历史注入 orchestrator 上下文）
- **LangFuse trace**：整个 `run()` 包裹在 trace 中，每次 LLM 调用上报 generation + token usage
- **ReAct 循环**：每个子 Agent 内部 Thought → Action → Observation，通过 Function Calling 调用 10 个工具

### 10 个工具

| 工具 | 类型 | 说明 |
|------|------|------|
| collaborative_filtering | 推荐 | 基于用户历史的协同过滤 |
| tag_based_recommendation | 推荐 | 菜系 + 价格标签匹配 |
| content_based_recommendation | 推荐 | 评分 + 销量热门推荐 |
| semantic_search_merchants | 推荐 | **RAG 语义检索**（bge-small-zh + ChromaDB） |
| get_business_overview | 分析 | 业务整体数据 |
| get_area_analysis | 分析 | 区域商业分析 |
| get_top_merchants | 分析 | 热门商户排行 |
| get_user_preferences | 画像 | 用户偏好分布 |
| check_user | 画像 | 用户身份检查 |
| summarize_recommendations | 聚合 | 推荐结果去重合并 |

## 评测

```bash
cd loudi_food_recommend/backend
python eval/run_eval.py                 # 跑全部 25 条用例
python eval/run_eval.py --limit 5       # 只跑前 5 条
python eval/run_eval.py --category recommendation_cuisine
python eval/run_eval.py --output result.json
```

输出指标：意图准确率、工具覆盖率、关键词覆盖率、平均/P50/P95 延迟、按类别准确率。

## 数据仓库分层

`scripts/hive/` 下含 ODS → DWD → DWS → ADS 四层建表脚本，`scripts/mysql/create_ads_tables.sql` 为 MySQL ADS 层建表，Sqoop 用于 Hive 与 MySQL 之间的数据同步。
