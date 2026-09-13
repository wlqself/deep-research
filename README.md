# Deep Research Agent

一个面向本地使用的多 Agent Deep Research 系统，支持研究、知识库、记忆、图片和多平台内容发布。

## 功能概览

- Main Agent / Researcher 多 Agent 研究工作流
- 流式输出、取消、错误反馈和线程恢复
- SQLite 短期状态、长期记忆和 Qdrant 混合检索
- 图片上传、图片库、视觉模型分析、OCR/图片上下文和图片附件绑定
- `generate_image` AI 生图工具，默认使用硅基流动的 Qwen/Qwen-Image
- 小红书、抖音、微信公众号和本地静态站点发布
- 文章草稿、平台格式化、配图绑定、发布审批和状态追踪
- 持久化 `interrupt_id`、幂等决策、过期时间、审计记录和 HITL 重启恢复

## 架构

```text
Browser -> FastAPI -> Main Agent / Supervisor -> Researcher Subagent
                         |-> Qdrant RAG / Findings / Memory
                         |-> Publishing Service
                              |-> WeChat API / Drafts
                              |-> Xiaohongshu MCP
                              |-> Douyin Playwright Service
                              |-> Local Static Site
```

主 Agent 不直接持有第三方浏览器进程。小红书和抖音由独立本地服务负责浏览器登录与页面自动化；主服务负责文章、附件、审批、状态轮询、错误反馈和恢复。

## 项目目录

```text
.
├── .env.example                 环境变量模板
├── deep_research/
│   ├── agent/                   Agent、模型角色、流式输出、事件和恢复
│   ├── artifacts/               Markdown 报告元数据
│   ├── citations/               引用校验与来源格式化
│   ├── context/                 单轮预算和并发上下文
│   ├── evaluation/              RAG 离线评测、指标和实验 Runner
│   ├── findings/                结构化研究证据
│   ├── handlers/                FastAPI 路由
│   ├── hitl/                    人工确认模型、决策服务、仓库和恢复
│   ├── log/                     结构化日志、审计日志和脱敏
│   ├── memory/                  长期记忆、召回和自动归纳
│   ├── middleware/              并发控制、请求上下文、召回和摘要
│   ├── persistence/             Checkpointer 与 Memory Store 生命周期
│   ├── prompts/                 Main、Researcher、Todo、摘要和记忆提示词
│   ├── publishing/              文章、附件、图片分析/生图和多平台发布
│   │   ├── publishers/          微信、小红书、抖音和本地站点适配器
│   │   └── renderers/           Markdown、微信公众号等平台渲染器
│   ├── rag/                    文档解析、切块、索引、混合检索和查询改写
│   ├── sources/                 网页和本地来源类型
│   ├── state/                  LangGraph 线程状态和状态访问
│   ├── static/                 原生 Web 前端和各功能页面
│   ├── tests/                  单元、HTTP、重启恢复和发布链路测试
│   └── tools/                  Agent 工具
└── data/                       本地运行数据，不提交到 Git
```

## 自动化工作流

```text
用户发送问题/图片
  -> 保存消息和图片附件
  -> 按需调用视觉分析或 OCR
  -> Main Agent 整理研究结果或文章草稿
  -> 绑定用户选择的图片，或使用 AI 生图结果
  -> 持久化发布审批 interrupt
  -> 用户在会话卡片中确认
  -> 恢复 Agent，调用对应平台发布器
  -> 轮询发布状态并记录回执、失败或结果未知
```

图片上传到图片库本身不会触发对话或图片分析；用户发送消息后，图片才会作为本轮请求附件传给 Agent。用户明确要求查看图片时，Main Agent 才会调用视觉分析工具。

调用 `generate_image` 时，提示词由 LLM 根据用户意图生成，并受到 `IMAGE_GENERATION_PROMPT_MAX_CHARS` 限制。生成结果自动保存为附件；如果用户要求使用生成图片发布，这些图片直接作为本次配图，不重复弹出配图选择卡片，但仍需最终发布审批。

发布审批采用严格串行流程：一个 `interrupt` 完成后 Agent 才继续并生成下一个 `interrupt`。前端启动或刷新时会从服务端恢复未完成的 HITL 卡片。决策使用 `interrupt_id`、幂等键和过期时间，发布中心和会话中的发布状态共用同一数据源。

## RAG 生产检索边界

生产检索使用 Dense 子 Chunk 检索、BM25、metadata 加权 Embedding、RRF 融合和 parent 折叠，并按 `chunk_index` 恢复连续正文。默认值为 `parent_size=4`、`candidate_k=32`、`rrf_k=60`、`metadata_weight=2`。

Qdrant 是主检索数据源，BM25 是从已索引 Chunk 派生出的词法索引。BM25 故障会退化为 Dense 结果，查询改写失败则退回原始 query。`evaluation/` 只负责离线指标和实验，生产代码不依赖它。

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

```powershell
Copy-Item .env.example .env
```

至少配置 `MODEL_API_KEY`、`MODEL_NAME`、`MODEL_BASE_URL`、`EMBEDDING_MODEL_NAME`、`EMBEDDING_DIMENSIONS`、`EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL` 和 `TAVILY_API_KEY`。

AI 生图默认使用硅基流动的 `Qwen/Qwen-Image`，复用 `MODEL_API_KEY`；如需单独的 Key，可填写 `IMAGE_GENERATION_API_KEY`。完整配置及默认值参见 `.env.example`。

## 启动主服务

```powershell
.\.venv\Scripts\python.exe -m uvicorn deep_research.main:app --reload
```

访问 `http://127.0.0.1:8000`，健康检查为 `http://127.0.0.1:8000/health`。

## 启动小红书自动化服务

小红书服务需要单独下载到以下目录。可执行文件和登录态不会提交到 Git：

```powershell
$serviceDir = "E:\my_agent\old coding\deep_research\publishing\publishers\xiaohongshu\xiaohongshu-mcp"
Set-Location $serviceDir

# 首次登录或登录态失效时执行
.\xiaohongshu-login-windows-amd64.exe

# 保持该窗口运行
.\xiaohongshu-mcp-windows-amd64.exe -headless=false -port ":18060"
```

主项目 `.env`：

```dotenv
XIAOHONGSHU_BASE_URL=http://127.0.0.1:18060
XIAOHONGSHU_TIMEOUT_SECONDS=120
XIAOHONGSHU_MAX_IMAGES=9
XIAOHONGSHU_TITLE_MAX_LENGTH=20
XIAOHONGSHU_CONTENT_MAX_LENGTH=1000
```

若提示 `18060` 端口已占用，通常表示服务已启动，不要重复启动。服务健康检查和登录状态接口分别为 `/health`、`/api/v1/login/status`。

## 启动抖音自动化服务

安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\deep_research\publishing\publishers\douyin\douyin-publisher\requirements.txt
```

启动服务：

```powershell
.\.venv\Scripts\python.exe .\deep_research\publishing\publishers\douyin\douyin-publisher\app.py
```

服务默认监听 `http://127.0.0.1:18070`。第一次发布会打开持久化 Playwright 浏览器，登录抖音创作者中心后，后续发布复用同一个登录态。

主项目 `.env`：

```dotenv
DOUYIN_BASE_URL=http://127.0.0.1:18070
DOUYIN_TIMEOUT_SECONDS=30
DOUYIN_MAX_IMAGES=30
DOUYIN_TITLE_MAX_LENGTH=30
```

可在抖音服务环境中设置 `DOUYIN_DRY_RUN=true`，先验证登录、上传和内容填充流程；确认无误后改为 `false` 执行真实发布。`POST /api/v1/publish` 返回 `202 Accepted` 只代表任务入队，主项目会继续轮询任务状态。

## 平台发布行为

- 微信公众号：`WECHAT_PUBLISH_MODE=draft` 为安全的草稿模式，改为 `publish` 才提交发布。
- 小红书：通过独立 MCP/浏览器服务发布，主项目轮询并保存 `note_id`。
- 抖音：通过独立 Playwright 服务异步发布，主项目轮询 `queued/running/published/failed`。
- 本地静态站点：用于开发验证文章和配图，不依赖第三方账号。

外部服务不可用、超时或没有返回可验证回执时，系统会显示明确失败或结果未知状态，保留可恢复记录，避免在无法确认时盲目重复发布。

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s deep_research/tests -v
```

测试使用 Fake Model、Fake Embeddings、临时 SQLite 和临时 Qdrant 数据，避免调用真实模型、Tavily 或生产知识库。

## 本地数据与隐私

以下内容只保存在本地，已通过 `.gitignore` 排除：`.env`、API Key、SQLite 数据库、Qdrant 向量数据、用户文件、日志、诊断文件、小红书 `cookies.json` 和下载的 `.exe`。

不要将运行时数据库、用户文档、真实登录态或真实 `.env` 提交到公开仓库。

## 当前边界

- 当前主要面向本地单用户环境，没有身份认证和多租户隔离。
- Researcher 不直接访问长期记忆；长期记忆只属于 Main Agent。
- 同步 Researcher 任务只在当前请求内运行，不是可跨重启恢复的后台任务。
- PDF 第一版只提取文本，不包含 OCR。
- Qdrant 当前使用本地模式；生产部署需要重新评估并发、鉴权和持久化方案。
