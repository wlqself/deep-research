# Deep Research Agent

一个面向学习和本地使用的 Deep Research Agent。项目以 FastAPI 为后端，使用
LangChain、LangGraph 和 deepagents 组织 Main/Researcher 多 Agent 工作流，并集成
网页研究、可验证引用、本地 RAG、短期记忆和长期记忆。

## 功能

- Main Agent 负责用户交互、研究编排、最终回答和报告保存。
- Researcher Subagent 负责网页搜索、页面阅读、本地知识库检索和证据整理。
- 最多两个独立 Researcher 任务并行执行，并提供任务状态和取消能力。
- 使用 `source_id` 统一管理网页与本地 RAG 引用。
- 使用结构化 findings 保存线程内研究证据。
- 使用 SQLite Checkpointer 保存线程、消息、任务、文件和短期记忆。
- 使用 SummarizationMiddleware 压缩长对话上下文。
- 使用 Qdrant 和 Embedding 检索 PDF、Markdown、TXT 文档切块。
- 使用 LangGraph AsyncSqliteStore 保存跨线程长期记忆。
- 长期记忆分为 user、reference、project、feedback，并支持查看、修改和遗忘。
- 支持在线生成、保存和下载 Markdown 报告。

## 架构

```text
Browser
  -> FastAPI
      -> Main Agent / Supervisor
          -> Researcher Subagent
              -> Web Search / Read Page
              -> Qdrant RAG
              -> Findings
      -> SQLite Checkpointer       (线程与短期记忆)
      -> AsyncSqliteStore          (长期记忆)
      -> Qdrant                    (文档 Chunk 向量)
      -> Document Registry SQLite  (知识库文档状态)
```

主要目录：

```text
deep_research/
├── agent/        Agent 构造、服务和流式事件
├── artifacts/    Markdown 报告元数据
├── citations/    引用校验与来源格式化
├── context/      单轮预算和并发上下文
├── findings/     结构化研究证据
├── handlers/     FastAPI 路由
├── memory/       长期记忆、召回和自动归纳
├── middleware/   并发控制和 Main 记忆召回
├── persistence/  Checkpointer 与 Memory Store 生命周期
├── prompts/      Main、Researcher、Todo、摘要和记忆提示词
├── rag/          文档注册、解析、切块、索引和检索
├── state/        LangGraph 线程状态
├── static/       原生 Web 前端
├── tests/        自动测试
└── tools/        Agent 工具
```

## RAG 生产检索边界

生产检索固定使用以下链路：原始 query 进入 Dense 子 Chunk 检索；查询改写词只进入
BM25；Chunk Embedding 使用 metadata 加权文本；Dense 与 BM25 的 ID 排名使用 RRF
融合；融合后的完整候选先按 `parent_id` 折叠，再返回最多 `top_k` 个不同父块，并按
`chunk_index` 恢复连续正文。`parent_size=4`、`candidate_k=32`、`rrf_k=60` 和
`metadata_weight=2` 是当前默认值。

Qdrant 是主检索数据源，BM25 是从 Qdrant 中已索引 Chunk 派生出的词法索引。BM25
故障只会退化为 Dense 结果，查询改写失败则退回原始 lexical query。`evaluation/`
只负责离线指标、评测 Runner 和 reranker 实验，生产代码不依赖它；reranker 尚未进入
生产链路。

修改 `parent_size`、Chunk 结构、metadata embedding 策略、Embedding 模型或维度后，
必须重新索引现有知识库。本项目不在本次改造中自动迁移或维护索引版本。

## 环境要求

- Python 3.11+
- 可用的 OpenAI 兼容 Chat API
- 可用的 Embedding API
- Tavily API Key（使用网页搜索时需要）

Qdrant 使用本地磁盘模式，不需要单独启动 Qdrant Server。

## 安装

```powershell
git clone https://github.com/wlqself/deep-research.git
cd deep-research

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## 配置

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

至少配置：

```dotenv
MODEL_API_KEY=
MODEL_NAME=
MODEL_BASE_URL=

EMBEDDING_MODEL_NAME=
EMBEDDING_DIMENSIONS=
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=

TAVILY_API_KEY=
```

`EMBEDDING_DIMENSIONS` 必须与 Embedding 模型输出维度及现有 Qdrant Collection
一致。更换模型或维度后需要重新索引知识库。

AI 生图默认使用硅基流动的 `Qwen/Qwen-Image`。它复用 `MODEL_API_KEY`；如需使用
单独的 Key，可填写 `IMAGE_GENERATION_API_KEY`。用户要求“生成并发布”时，生成结果
会自动保存到共享图片库并作为本次发布配图，不再重复弹出配图选择卡片，但仍需经过最终发布审批。

完整配置及默认值参见 `.env.example`。

## 启动

```powershell
uvicorn deep_research.main:app --reload
```

访问：

```text
http://127.0.0.1:8000
```

健康检查：

```text
http://127.0.0.1:8000/health
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s deep_research/tests -v
```

测试使用 Fake Model、Fake Embeddings、临时 SQLite 和临时 Qdrant 数据，避免调用
真实模型、Tavily 或生产知识库。

## 本地数据与隐私

以下内容只保存在本地，已通过 `.gitignore` 排除：

- `.env` 和 API Key
- SQLite 会话及长期记忆数据库
- Qdrant 向量数据
- 用户上传的 PDF、Markdown 和 TXT
- 长期记忆 Markdown 投影
- 日志与诊断文件

不要将运行时数据库、用户文档或真实 `.env` 提交到公开仓库。

## 当前边界

- 当前主要面向本地单用户环境，没有身份认证和多租户隔离。
- Researcher 不直接访问长期记忆；长期记忆只属于 Main Agent。
- 同步 Researcher 任务只在当前请求内运行，不是可跨重启恢复的后台任务。
- PDF 第一版只提取文本，不包含 OCR。
- Qdrant 当前使用本地模式；生产部署需要重新评估并发、鉴权和持久化方案。
