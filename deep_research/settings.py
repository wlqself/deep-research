from pydantic import Field, NonNegativeInt, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    log_level: str = "INFO"
    log_format: str = "json"
    log_dir: str = Field(default="logs", min_length=1)
    log_file: str = Field(
        default="deep-research.log",
        min_length=1,
    )
    log_max_bytes: PositiveInt = 10_485_760
    log_backup_count: int = Field(default=5, ge=0)
    log_audit_file: str = Field(
        default="audit.log",
        min_length=1,
    )
    log_include_traceback: bool = False


    model_api_key: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_base_url: str | None = None
    main_agent_enable_thinking: bool = False
    researcher_enable_thinking: bool = True
    image_analysis_enabled: bool = True
    image_analysis_model_name: str | None = Field(default=None, min_length=1)
    image_analysis_api_key: str | None = Field(default=None, min_length=1)
    image_analysis_base_url: str | None = Field(default=None, min_length=1)
    image_analysis_timeout_seconds: PositiveInt = 60
    image_generation_enabled: bool = True
    image_generation_model_name: str = Field(
        default="Qwen/Qwen-Image",
        min_length=1,
    )
    image_generation_api_key: str | None = Field(default=None, min_length=1)
    image_generation_base_url: str = Field(
        default="https://api.siliconflow.cn/v1",
        min_length=1,
    )
    image_generation_timeout_seconds: PositiveInt = 180
    image_generation_prompt_max_chars: PositiveInt = 800
    image_generation_max_images: PositiveInt = 4
    main_stream_max_chars: PositiveInt = 32_000
    main_stream_repetition_min_chars: PositiveInt = 160
    main_stream_repetition_count: PositiveInt = 3
    heartbeat_interval_seconds: PositiveInt = 15

    checkpoint_db_path: str = Field(
        default="deep_research.sqlite",
        min_length=1,
    )
    publishing_db_path: str = Field(
        default="deep_research_publishing.sqlite",
        min_length=1,
    )
    hitl_db_path: str = Field(
        default="deep_research_hitl.sqlite",
        min_length=1,
    )
    hitl_expiration_seconds: PositiveInt = 86_400
    published_site_dir: str = Field(
        default="data/published-site",
        min_length=1,
    )
    public_site_base_url: str | None = None

    wechat_app_id: str | None = Field(default=None, min_length=1)
    wechat_app_secret: str | None = Field(default=None, min_length=1)
    wechat_author: str | None = Field(default=None, min_length=1)
    wechat_thumb_media_id: str | None = Field(default=None, min_length=1)
    wechat_publish_mode: str = Field(
        default="draft",
        pattern="^(draft|publish)$",
    )
    publishing_upload_dir: str = Field(
        default="data/publishing-uploads",
        min_length=1,
    )
    publishing_upload_max_bytes: PositiveInt = 10_485_760
    wechat_theme: str = Field(
        default="minimal",
        pattern="^(minimal|tech|warm)$",
    )
    wechat_recovery_interval_seconds: PositiveInt = 30
    wechat_recovery_max_attempts: PositiveInt = 12
    publication_recovery_interval_seconds: PositiveInt = 30
    publication_recovery_max_attempts: PositiveInt = 12

    # Xiaohongshu uses a separately managed local Playwright/CDP service.
    xiaohongshu_base_url: str | None = Field(default=None, min_length=1)
    xiaohongshu_timeout_seconds: PositiveInt = 30
    xiaohongshu_max_images: PositiveInt = 9
    xiaohongshu_title_max_length: PositiveInt = 20
    xiaohongshu_content_max_length: PositiveInt = 1000

    # Douyin uses a separately managed local Playwright/CDP service.
    douyin_base_url: str | None = Field(default=None, min_length=1)
    douyin_timeout_seconds: PositiveInt = 30
    douyin_max_images: PositiveInt = 30
    douyin_title_max_length: PositiveInt = 30
    
    tavily_api_key: str | None = None

    agent_recursion_limit: PositiveInt = 32 #最多想几步
    max_search_calls: PositiveInt = 4 #最多搜几次
    max_page_reads: PositiveInt = 6 #最多读几页
    max_page_chars: PositiveInt = 12_000 #每页最多看多少字
    max_parallel_research_tasks: PositiveInt = 2
    memory_trigger_tokens: PositiveInt = 128_000
    memory_keep_messages: PositiveInt = 18
    memory_trim_tokens: PositiveInt = 4000

    rag_chunk_size: PositiveInt = 1200
    rag_chunk_overlap: int = Field(default=200, ge=0)
    rag_parent_size: PositiveInt = 4
    rag_top_k: PositiveInt = 8
    rag_candidate_k: PositiveInt = 32
    rag_rrf_k: int = Field(default=60, ge=0)
    rag_query_rewrite_enabled: bool = True
    rag_metadata_weight: NonNegativeInt = 2
    embedding_model_name: str = Field(min_length=1)
    embedding_dimensions: PositiveInt
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_batch_size: PositiveInt = 64
    rag_max_file_bytes: PositiveInt = 10_000_000
    rag_max_context_chars: PositiveInt = 24_000
    rag_qdrant_path: str = Field(
        default="data/qdrant",
        min_length=1,
    )
    rag_collection_name: str = Field(
        default="deep_research_documents",
        min_length=1,
    )
    rag_registry_db_path: str = Field(
        default="deep_research_rag.sqlite",
        min_length=1,
    )
    rag_documents_path: str = Field(
        default="data/rag/documents",
        min_length=1,
    )
    rag_bm25_db_path: str = Field(
        default="data/rag/bm25.sqlite",
        min_length=1,
    )
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        str_strip_whitespace=True,
    )

    # 未来创建独立 AsyncSqliteStore
    memory_db_path: str = Field(
        default="deep_research_memory.sqlite",
        min_length=1,
    )
    # 当前用户的唯一标识。多用户场景下每个用户有独立的记忆空间
    memory_user_id: str = Field(
        default="local-user",
        min_length=1,
    )

    memory_user_max_entries: PositiveInt = 30 # # 每个用户最多保留 30 条记忆。超过时，旧的/不重要的会被淘汰或合并
    memory_user_max_chars: PositiveInt = 3000 # 每条记忆的最大字符数。超过会被截断，防止单条记忆占用过多 token
    memory_recall_limit: PositiveInt = 3 # 每次对话时最多召回 3 条相关记忆注入到 prompt 里。太少会漏掉重要上下文，太多会浪费 token
    memory_page_size: PositiveInt = 20 # 前端列出记忆时的分页大小——每页展示 20 条
    memory_review_interval_turns: PositiveInt = 10 # 每 10 轮对话触发一次记忆回顾。回顾时会让 LLM 整理、合并、淘汰记忆
    memory_max_candidates: PositiveInt = 5 # 记忆提取/回顾时，最多给 LLM 5 条候选记忆让它决定保留哪些、合并哪些、丢弃哪些
    memory_projection_path: str = Field(
        default="data/memory_projection",
        min_length=1,
    )
    memory_recall_max_chars: PositiveInt = 6000

    memory_extraction_max_messages: PositiveInt = 30 # 一批最多提交多少条 ReviewMessage
    memory_extraction_max_chars: PositiveInt = 16_000 # 一批消息内容最多多少字符
    memory_extraction_max_artifacts: PositiveInt = 20 # 一批最多附带多少条 artifact metadata
    memory_review_max_batches_per_turn: PositiveInt = 2 # 一次成功轮次最多处理多少个 ReviewBatch
    memory_extraction_max_summary_chars: PositiveInt = 4_000
settings = Settings()
