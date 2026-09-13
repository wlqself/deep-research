import asyncio
import logging
from ..log.logging_utils import log_event
from ..config import settings
from .projection import (
    _ordered_entries,
    ensure_projection_directories,
    projection_root,
    render_index_markdown,
    render_page_markdown,
    render_summary_archive_markdown,
    render_user_markdown,
    summary_archive_path,
    write_text_atomic,
)
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from ..log.log_audit import audit_event
from langgraph.store.base import PutOp
from langgraph.store.sqlite import AsyncSqliteStore
from langgraph.store.base import PutOp
from .decision import MemoryDecision
from .type import (
    MemoryCandidate,
    MemoryEntry,
    MemoryEntryKind,
    MemoryScope,
    SummaryArchive,
    MemorySuppression,
)
logger = logging.getLogger(__name__)

_ENTRY_KINDS: tuple[MemoryEntryKind, ...] = (
    "user",
    "reference",
    "project",
    "feedback",
)

class MemoryService:
    def __init__(
        self,
        store: AsyncSqliteStore,
        *,
        user_id: str,
    ) -> None:
        normalized_user_id = user_id.strip()

        if not normalized_user_id:
            raise ValueError("user_id must not be empty")

        self._store = store
        self._user_id = normalized_user_id
        self._recall_revision = 0

    @property
    def recall_revision(self) -> int:
        return self._recall_revision


    def _mark_store_changed(self) -> None:
        self._recall_revision += 1
    # 用来把不同 kind 的记忆隔离存储
    def _entry_namespace(
        self,
        kind: MemoryEntryKind,
    ) -> tuple[str, ...]:
        return (
            "memories",
            self._user_id,
            kind,
        )

    def _entry_prefix(self) -> tuple[str, ...]:
        return (
            "memories",
            self._user_id,
        )

    # 这是唯一的写入口。所有记忆（无论 reference 还是 feedback）都走这一个方法，保证写入逻辑统一。
    async def put_entry(
        self,
        entry: MemoryEntry,
    ) -> None:
        await self._store.aput(
            self._entry_namespace(entry.kind),
            entry.id, # 唯一键
            # 把 Pydantic 模型转成纯 Python 字典，格式兼容 JSON 序列化
            entry.model_dump(
                mode="json",
                exclude_none=False,
            ),
            # 这三个字段需要被索引，用于后续的语义/关键词检索。
            index=[
                "title",
                "summary",
                "keywords",
            ],
        )
        self._mark_store_changed()
        await self._rebuild_after_store_write()
    # 精准获取记忆
    async def get_entry(
        self,
        kind: MemoryEntryKind,
        entry_id: str,
    ) -> MemoryEntry | None:
        item = await self._store.aget(
            self._entry_namespace(kind),
            entry_id,
        )

        if item is None:
            return None

        return MemoryEntry.model_validate(item.value)

    # 给定记忆类型、业务键、作用域，直接过滤出所有匹配的 active 记忆条目（通常预期 0 或 1 条，但防御性地返回列表）。
    # 它用于「按 key 查重 / 定位旧记忆」，而不是语义搜索。
    async def find_active_by_key(
        self,
        kind: MemoryEntryKind,
        memory_key: str,
        scope: MemoryScope,
    ) -> list[MemoryEntry]:
        normalized_key = memory_key.strip()

        if not normalized_key:
            raise ValueError(
                "memory_key must not be empty"
            )

        items = await self._store.asearch(
            self._entry_namespace(kind), # namespace = ("memories", user_id, kind)
            query=None,  # ← 关键：None 表示不走向量搜索
            filter={
                "memory_key": normalized_key,
                "scope": scope,
                "status": "active",
            },
            limit=2,
        )

        return [
            MemoryEntry.model_validate(item.value)
            for item in items
        ]
    # 在指定类型的记忆命名空间内，用自然语言查询做语义相似度搜索，只返回状态为 "active" 的记忆，并附带每条记忆的匹配分数。
    async def find_similar_entries(
        self,
        kind: MemoryEntryKind,
        query: str,
        *,
        limit: int,
    ) -> list[tuple[MemoryEntry, float | None]]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must not be empty")

        if limit < 1:
            raise ValueError("limit must be positive")

        items = await self._store.asearch(
            self._entry_namespace(kind),
            query=normalized_query,
            filter={"status": "active"},
            limit=limit,
        )

        return [
            (
                MemoryEntry.model_validate(item.value),
                item.score,
            )
            for item in items
        ]

    async def find_review_memories(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[MemoryEntry]:
        normalized_query = query.strip()

        if not normalized_query:
            return []

        if limit < 1:
            raise ValueError("limit must be positive")

        items = await self._store.asearch(
            self._entry_prefix(),
            query=normalized_query,
            filter={"status": "active"},
            limit=limit,
        )
        matches = [
            (
                MemoryEntry.model_validate(item.value),
                item.score,
            )
            for item in items
        ]
        # 全局排序，True（有分数）排在 False（无分数）前面​ → 保证「真正检索到的」优先于「没分数的兜底项」，在有分数的那些里，按分数从高到低排（越相似越靠前）
        matches.sort(
            key=lambda item: (
                item[1] is not None,
                item[1] if item[1] is not None else 0.0,
            ),
            reverse=True,
        )
        # 去重
        selected: list[MemoryEntry] = []
        seen: set[tuple[str, str]] = set()

        for entry, _score in matches:
            identity = (entry.kind, entry.id)

            if identity in seen:
                continue

            seen.add(identity)
            selected.append(entry)

            if len(selected) >= limit:
                break

        return selected
    # 分页列出某类记忆
    async def list_entries(
        self,
        kind: MemoryEntryKind,
        *,
        limit: int,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        items = await self._store.asearch(
            self._entry_namespace(kind),
            query=None,
            limit=limit,
            offset=offset,
        )

        return [
            MemoryEntry.model_validate(item.value)
            for item in items
        ]
    # 按 ID 删除
    async def delete_entry(
        self,
        kind: MemoryEntryKind,
        entry_id: str,
    ) -> None:
        await self._store.adelete(
            self._entry_namespace(kind),
            entry_id,
        )
        self._mark_store_changed()
        await self._rebuild_after_store_write()

    async def find_similar_candidates(
        self,
        candidate: MemoryCandidate,
        limit: int,
    ) -> list[tuple[MemoryEntry, float | None]]:
        """
        面向 MemoryCandidate 的只读相似搜索。

        规则：
        - ignore：不搜索、不写入，直接返回空列表
        - 其他 kind：只搜索相同 kind、同一 user_id namespace、status=active
        - 查询输入：title + summary + keywords 组合
        - 保留原始 score，不做阈值判断、不决定重复/更新
        - Store / Embedding 失败直接抛异常，不伪装成空列表
        """
        # ── ignore 直接跳过 ──────────────────────────────────────
        if candidate.kind == "ignore":
            return []

        # ── 构造查询文本 ──────────────────────────────────────────
        query_parts: list[str] = []

        if getattr(candidate, "title", None):
            query_parts.append(candidate.title)
        if getattr(candidate, "summary", None):
            query_parts.append(candidate.summary)
        if getattr(candidate, "keywords", None):
            query_parts.append(" ".join(candidate.keywords))

        query_text = " ".join(query_parts).strip()

        # 候选本身没有任何可搜索内容，无法查询
        # 这不是 Store 失败，返回空列表是合理的
        if not query_text:
            return []

        # ── 复用现有搜索逻辑 ──────────────────────────────────────
        # find_similar_entries 内部已经保证：
        #   - namespace = ("memories", self._user_id, kind)  → 不跨用户、不跨 kind
        #   - filter={"status": "active"}                    → 不搜索 superseded/deleted
        #   - limit                                          → 控制返回数量
        #   - query 非空校验                                  → 上面已保证
        #
        # 如果 self._store.asearch 抛异常，这里不 catch，
        # 异常自然向上传播，调用方可以正确判断本轮失败。
        return await self.find_similar_entries(
            candidate.kind,
            query=query_text,
            limit=limit,
        )
    # 转成 JSON 兼容的字典
    def _entry_value(
        self,
        entry: MemoryEntry,
    ) -> dict[str, object]:
        return entry.model_dump(
            mode="json",
            exclude_none=False,
        )

    # 构造操作描述对象
    def _entry_put_op(
        self,
        entry: MemoryEntry,
    ) -> PutOp:
        return PutOp(
            self._entry_namespace(entry.kind),
            entry.id,
            self._entry_value(entry),
            index=[
                "title",
                "summary",
                "keywords",
            ],
        )


    def _entry_from_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        now: datetime,
    ) -> MemoryEntry:
        return MemoryEntry(
            id=uuid4().hex, # 生成新 ID
            created_at=now,
            updated_at=now,
            status="active",
            **candidate.model_dump(mode="python"), # 返回 Python 原生对象
        )

    async def apply_decision(
        self,
        candidate: MemoryCandidate,
        decision: MemoryDecision,
        *,
        existing: MemoryEntry | None = None,
        clear_suppression: bool = False,
    ) -> MemoryEntry | None:

        if candidate.kind == "ignore":
            if decision.action != "ignore":
                raise ValueError(
                    "ignore candidate must produce ignore decision"
                )

            log_event(
                logger,
                logging.INFO,
                "memory.suppressed",
                status="suppressed",
            )

            return None

        if decision.action == "ignore":
            raise ValueError(
                "non-ignore candidate cannot produce ignore decision"
            )

        suppression = None

        if clear_suppression:
            suppression = await self.find_suppression(
                candidate.kind,
                candidate.memory_key,
                candidate.scope,
            )

        if decision.action == "no-op":
            if (
                existing is not None
                and decision.existing_id != existing.id
            ):
                raise ValueError(
                    "decision existing_id does not match existing entry"
                )

            if suppression is not None:
                await self._store.abatch(
                    [
                        PutOp(
                            self._suppression_namespace(),
                            suppression.suppression_id,
                            None,
                        )
                    ]
                )
                self._mark_store_changed()
            log_event(
                logger,
                logging.INFO,
                "memory.noop",
                memory_id=(
                    existing.id
                    if existing is not None
                    else None
                ),
                status="no_op",
            )
            return existing

        now = datetime.now(timezone.utc)
        new_entry = self._entry_from_candidate(
            candidate,
            now=now,
        )

        if decision.action == "create":
            ops: list[PutOp] = [
                self._entry_put_op(new_entry)
            ]

            if suppression is not None:
                ops.append(
                    PutOp(
                        self._suppression_namespace(),
                        suppression.suppression_id,
                        None,
                    )
                )

            await self._store.abatch(ops)
            self._mark_store_changed()
            await self._rebuild_after_store_write()

            audit_event(
                "memory.create",
                "completed",
                thread_id=getattr(
                    candidate,
                    "source_thread_id",
                    None,
                ),
                memory_id=new_entry.id,
            )
            log_event(
                logger,
                logging.INFO,
                "memory.created",
                memory_id=new_entry.id,
                status="completed",
            )

            return new_entry

        if decision.action == "update":
            if existing is None:
                raise ValueError(
                    "update requires existing entry"
                )

            if decision.existing_id != existing.id:
                raise ValueError(
                    "decision existing_id does not match existing entry"
                )

            if existing.status != "active":
                raise ValueError(
                    "only active entry can be updated"
                )

            if (
                existing.kind != candidate.kind
                or existing.memory_key != candidate.memory_key
                or existing.scope != candidate.scope
            ):
                raise ValueError(
                    "existing entry does not match candidate identity"
                )
            # 基于现有模型实例，创建一个修改后的副本。
            superseded = existing.model_copy(
                update={
                    "status": "superseded",
                    "updated_at": now,
                }
            )
            # 原子批量写入，先修改状态再改内容
            """
            正常写入
            -> 旧记录 superseded
            -> 新记录 active

            写入失败
            -> 删除可能已经写入的新记录
            -> 恢复旧记录原始内容
            -> 重新抛出原始异常
            """
            ops: list[PutOp] = [
                self._entry_put_op(superseded),
                self._entry_put_op(new_entry),
            ]

            if suppression is not None:
                ops.append(
                    PutOp(
                        self._suppression_namespace(),
                        suppression.suppression_id,
                        None,
                    )
                )

            try:
                audit_event(
                    "memory.update",
                    "completed",
                    thread_id=getattr(
                        candidate,
                        "source_thread_id",
                        None,
                    ),
                    memory_id=new_entry.id,
                )
                await self._store.abatch(ops)
            except Exception:
                try:
                    await self._store.adelete(
                        self._entry_namespace(new_entry.kind),
                        new_entry.id,
                    )

                    await self._store.aput(
                        self._entry_namespace(existing.kind),
                        existing.id,
                        self._entry_value(existing),
                        index=[
                            "title",
                            "summary",
                            "keywords",
                        ],
                    )
                except Exception as compensation_error:
                    raise RuntimeError(
                        "memory update failed and compensation failed"
                    ) from compensation_error

                raise
            self._mark_store_changed()
            await self._rebuild_after_store_write()
            log_event(
                logger,
                logging.INFO,
                "memory.updated",
                memory_id=new_entry.id,
                status="completed",
            )

            return new_entry

        raise ValueError(
            f"unsupported decision action: {decision.action}"
        )

    # 输出类型下的所有​ MemoryEntry 列表
    async def _load_all_entries(
        self,
        kind: MemoryEntryKind,
        *,
        batch_size: int,
    ) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        offset = 0

        while True:
            batch = await self.list_entries(
                kind,
                limit=batch_size,
                offset=offset,
            )

            if not batch:
                break

            entries.extend(batch)
            offset += len(batch)

            if len(batch) < batch_size:
                break

        return entries


    async def _load_all_summary_archives(
        self,
        *,
        batch_size: int,
    ) -> list[SummaryArchive]:
        archives: list[SummaryArchive] = []
        offset = 0

        while True:
            batch = await self.list_summary_archives(
                limit=batch_size,
                offset=offset,
            )

            if not batch:
                break

            archives.extend(batch)
            offset += len(batch)

            if len(batch) < batch_size:
                break

        return archives

    # suppression 全量读取 helper
    async def _load_all_suppressions(
        self,
        *,
        batch_size: int,
    ) -> list[MemorySuppression]:
        suppressions: list[MemorySuppression] = []
        offset = 0

        while True:
            items = await self._store.asearch(
                self._suppression_namespace(),
                query=None,
                limit=batch_size,
                offset=offset,
            )

            if not items:
                break

            suppressions.extend(
                MemorySuppression.model_validate(item.value)
                for item in items
            )

            offset += len(items)

            if len(items) < batch_size:
                break

        return suppressions

    # 把 SQLite 里的所有记忆数据重新渲染成 Markdown 文件。
    async def rebuild_projections(self) -> None:
        page_size = settings.memory_page_size
        # 创建 data/memory_projection/memories/、reference/、feedback/ 等目录
        await asyncio.to_thread(
            ensure_projection_directories
        )

        entries_by_kind: dict[
            MemoryEntryKind,
            list[MemoryEntry],
        ] = {}
        # 加载所有模块
        for kind in (
            "user",
            "reference",
            "project",
            "feedback",
        ): entries_by_kind[kind] = await self._load_all_entries(
            kind,
            batch_size=page_size,
        )

        root = projection_root()
        # 从 Store 全量加载所有归档
        archives = await self._load_all_summary_archives(
            batch_size=settings.memory_page_size,
        )

        archive_directory = (
            root
            / "memory-internal"
            / "summary-archive"
        )
        # 遍历，渲染 + 原子写入，同时收集"应该存在的文件名"
        archive_names: set[str] = set()

        for archive in archives:
            archive_path = summary_archive_path(
                archive
            )
            archive_names.add(archive_path.name)

            archive_markdown = render_summary_archive_markdown(
                archive
            )

            await asyncio.to_thread(
                write_text_atomic,
                archive_path,
                archive_markdown,
            )

        await asyncio.to_thread(
            self._remove_stale_summary_archives,
            archive_directory,
            archive_names,
        )
        # 加载user makrdown
        user_markdown = render_user_markdown(
            entries_by_kind["user"],
            max_entries=settings.memory_user_max_entries,
            max_chars=settings.memory_user_max_chars,
        )
        # 原子写入
        await asyncio.to_thread(
            write_text_atomic,
            root / "memories" / "USER.md",
            user_markdown,
        )

        for kind in (
            "reference",
            "project",
            "feedback",
        ):
            kind_entries = entries_by_kind[kind]
            ordered = _ordered_entries(kind_entries) # 排序，排在最前面（时间）
            kind_directory = root / "memories" / kind

            page_names: set[str] = set()
            # 相同的分页逻辑 每页二十条
            for start in range(0, len(ordered), page_size):
                page_number = start // page_size + 1 # 包含了应该存在的页码
                page_name = f"page-{page_number:03d}.md"
                page_names.add(page_name)

                page_markdown = render_page_markdown(
                    ordered[start:start + page_size]
                )

                await asyncio.to_thread(
                    write_text_atomic,
                    kind_directory / page_name,
                    page_markdown,
                )

            index_markdown = render_index_markdown(
                kind_entries,
                page_size=page_size,
            )
            # 生成 index.md
            await asyncio.to_thread(
                write_text_atomic,
                kind_directory / "index.md",
                index_markdown,
            )
            # 清理过期页面
            await asyncio.to_thread(
                self._remove_stale_pages,
                kind_directory,
                page_names,
            )
    # 清理过期页面
    @staticmethod
    def _remove_stale_pages(
        directory: Path,
        keep: set[str],
    ) -> None:
        for page_path in directory.glob("page-*.md"):
            if page_path.name not in keep:
                page_path.unlink()

    @staticmethod
    def _remove_stale_summary_archives(
        directory: Path,
        keep: set[str],
    ) -> None:
        for archive_path in directory.glob(
            "summary-*.md"
        ):
            if archive_path.name not in keep:
                archive_path.unlink()

    async def _rebuild_after_store_write(self) -> None:
        try:
            await self.rebuild_projections()
        except Exception as error:
            log_event(
                logger,
                logging.ERROR,
                "memory.projection.rebuild.failed",
                status="failed",
                error_code="memory_projection_rebuild_failed",
                exception_type=type(error).__name__,
                exc_info=True,
            )
    """
    kind=None
    -> 四个业务 namespace

    keyword=None
    -> 普通 active 列表

    keyword 有值
    -> Store 语义搜索

    多类结果
    -> 按 updated_at 倒序合并
    -> 全局分页
    """
    async def list_active_memories(
        self,
        *,
        kind: MemoryEntryKind | None = None,
        keyword: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> list[MemoryEntry]:
        if page < 1:
            raise ValueError("page must be positive")

        if page_size < 1:
            raise ValueError("page_size must be positive")

        if kind is not None and kind not in _ENTRY_KINDS:
            raise ValueError("invalid memory kind")

        normalized_keyword = (
            keyword.strip()
            if isinstance(keyword, str)
            else ""
        )

        selected_kinds = (
            (kind,)
            if kind is not None
            else _ENTRY_KINDS
        )
        # 每个 kind 各拉 page * page_size 条，然后全局排序，最后 matches[start:end] 切片。
        fetch_limit = page * page_size
        matches: list[MemoryEntry] = []

        for selected_kind in selected_kinds:
            items = await self._store.asearch(
                self._entry_namespace(selected_kind),
                query=normalized_keyword or None,
                filter={
                    "status": "active",
                },
                limit=fetch_limit,
            )

            matches.extend(
                MemoryEntry.model_validate(item.value)
                for item in items
            )

        matches.sort(
            key=lambda entry: (
                entry.updated_at,
                entry.id,
            ),
            reverse=True,
        )

        start = (page - 1) * page_size
        end = start + page_size

        return matches[start:end]

    def _summary_archive_namespace(
        self,
        thread_id: str,
    ) -> tuple[str, ...]:
        normalized_thread_id = thread_id.strip()

        if not normalized_thread_id:
            raise ValueError(
                "thread_id must not be empty"
            )

        if "." in normalized_thread_id:
            raise ValueError(
                "thread_id must not contain periods"
            )

        return (
            *self._summary_archive_prefix(),
            normalized_thread_id,
        )
    # 幂等写入
    async def put_summary_archive(
        self,
        archive: SummaryArchive,
    ) -> bool:
        namespace = self._summary_archive_namespace(
            archive.thread_id
        )

        existing = await self._store.aget(
            namespace,
            archive.summary_hash,
        )

        if existing is not None:
            return False

        await self._store.aput(
            namespace,
            archive.summary_hash, # ← 用 hash 当 key
            archive.model_dump(
                mode="json",
            ),
            index=[
                "retrieval_summary",
                "topics",
            ],
        )
        self._mark_store_changed()
        await self._rebuild_after_store_write()

        return True

    # 精确读取
    async def get_summary_archive(
        self,
        thread_id: str,
        summary_hash: str,
    ) -> SummaryArchive | None:
        item = await self._store.aget(
            self._summary_archive_namespace(thread_id),
            summary_hash,
        )

        if item is None:
            return None

        return SummaryArchive.model_validate(
            item.value
        )

    def _summary_archive_prefix(
        self,
    ) -> tuple[str, ...]:
        return (
            "memory-internal",
            self._user_id,
            "summary-archive",
        )
    """
    list_summary_archives(thread_id="thread-1")
    -> 只读一个 thread 的归档

    list_summary_archives()
    -> 读取当前用户全部 thread 的归档，支持分页
    """
    async def list_summary_archives(
        self,
        *,
        thread_id: str | None = None,
        limit: int,
        offset: int = 0,
    ) -> list[SummaryArchive]:
        if limit < 1:
            raise ValueError("limit must be positive")

        if offset < 0:
            raise ValueError("offset must not be negative")

        namespace = (
            self._summary_archive_namespace(thread_id)
            if thread_id is not None
            else self._summary_archive_prefix()
        )

        items = await self._store.asearch(
            namespace,
            query=None,
            limit=limit,
            offset=offset,
        )

        return [
            SummaryArchive.model_validate(item.value)
            for item in items
        ]

    async def list_recallable_summary_archives(
        self,
        *,
        limit: int,
    ) -> list[SummaryArchive]:
        """List a bounded local index without performing semantic search."""

        if limit < 1:
            raise ValueError("limit must be positive")

        archives = await self.list_summary_archives(
            limit=max(limit * 4, limit),
        )
        recallable: list[SummaryArchive] = []
        for archive in archives:
            if await self._archive_is_recallable(archive):
                recallable.append(archive)
            if len(recallable) >= limit:
                break
        return recallable

    # 选择性向量检索： search_summary_archives(query)
    #                 -> 当前用户全部 thread 的归档中做向量检索
    async def search_summary_archives(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[SummaryArchive]:
        normalized_query = query.strip()

        if not normalized_query:
            return []

        if limit < 1:
            raise ValueError("limit must be positive")

        items = await self._store.asearch(
            self._summary_archive_prefix(),
            query=normalized_query,
            limit=limit,
        )

        archives = [
            SummaryArchive.model_validate(item.value)
            for item in items
        ]

        recallable_archives: list[SummaryArchive] = []

        for archive in archives:
            if await self._archive_is_recallable(archive):
                recallable_archives.append(archive)

            if len(recallable_archives) >= limit:
                break

        return recallable_archives

    def _suppression_namespace(self) -> tuple[str, ...]:
        return (
            "memory-internal",
            self._user_id,
            "suppressions",
        )

    # 生成稳定的 identity key
    # 这个 ID 的目的，是保证同一个 (kind, memory_key, scope) 不会因为重复 forget 产生多条 suppression

    def _suppression_id(
        self,
        kind: MemoryEntryKind,
        memory_key: str,
        scope: MemoryScope,
    ) -> str:
        normalized_key = memory_key.strip()

        if not normalized_key:
            raise ValueError("memory_key must not be empty")

        return f"{kind}:{scope}:{normalized_key}"

    async def find_suppression(
        self,
        kind: MemoryEntryKind,
        memory_key: str,
        scope: MemoryScope,
    ) -> MemorySuppression | None:
        suppression_id = self._suppression_id(
            kind,
            memory_key,
            scope,
        )

        item = await self._store.aget(
            self._suppression_namespace(),
            suppression_id,
        )

        if item is None:
            return None

        return MemorySuppression.model_validate(
            item.value
        )

    async def put_suppression(
        self,
        suppression: MemorySuppression,
    ) -> None:
        # 重新根据业务身份计算正确 Store key。不要直接信任传入的 suppression_id
        expected_id = self._suppression_id(
            suppression.kind,
            suppression.memory_key,
            suppression.scope,
        )
        # 内部一致性保护
        if suppression.suppression_id != expected_id:
            raise ValueError(
                "suppression_id does not match suppression identity"
            )

        await self._store.aput(
            self._suppression_namespace(),
            suppression.suppression_id,
            suppression.model_dump(
                mode="json",
                exclude_none=False,
            ),
            index=[],
        )

        self._mark_store_changed()
        log_event(
            logger,
            logging.INFO,
            "memory.suppression.created",
            memory_id=suppression.forgotten_memory_id,
            status="completed",
        )

    async def delete_suppression(
        self,
        kind: MemoryEntryKind,
        memory_key: str,
        scope: MemoryScope,
    ) -> bool:

        suppression = await self.find_suppression(
            kind,
            memory_key,
            scope,
        )

        if suppression is None:
            return False

        await self._store.adelete(
            self._suppression_namespace(),
            suppression.suppression_id,
        )

        self._mark_store_changed()
        log_event(
            logger,
            logging.INFO,
            "memory.suppression.deleted",
            status="completed",
        )
        return True

    async def forget_entry(
        self,
        kind: MemoryEntryKind,
        entry_id: str,
        *,
        source_thread_id: str | None = None,
        reason: str = "explicit_user_forget",
    ) -> MemoryEntry:
        # 读取并验证目标
        entry = await self.get_entry(
            kind,
            entry_id,
        )

        if entry is None or entry.status != "active":
            raise ValueError("active memory not found")

        # 构造 suppression。
        now = datetime.now(timezone.utc)

        suppression_id = self._suppression_id(
            entry.kind,
            entry.memory_key,
            entry.scope,
        )
        # 这里使用 entry 中的身份字段，不使用调用者再次传入的 key 或 scope
        suppression = MemorySuppression(
            suppression_id=suppression_id,
            kind=entry.kind,
            memory_key=entry.memory_key,
            scope=entry.scope,
            forgotten_memory_id=entry.id,
            source_thread_id=source_thread_id,
            created_at=now,
            reason=reason,
        )

        # 构造一个批次
        await self._store.abatch(
            [
                PutOp(
                    self._suppression_namespace(),
                    suppression.suppression_id,
                    suppression.model_dump(
                        mode="json",
                        exclude_none=False,
                    ),
                    index=[],
                ),
                # value = None 表示删除
                PutOp(
                    self._entry_namespace(entry.kind),
                    entry.id,
                    None,
                ),
            ]
        )
        # Store 成功后的状态更新。
        self._mark_store_changed()
        """
        Store 成功 + projection 成功
            -> 返回 entry
            -> 上层报告成功

        Store 成功 + projection 失败
            -> revision 已增加
            -> 异常继续抛出
            -> 上层不得报告成功
        """
        await self.rebuild_projections()
        audit_event(
            "memory.forget",
            "completed",
            thread_id=source_thread_id,
            memory_id=entry.id,
        )
        log_event(
            logger,
            logging.INFO,
            "memory.forgotten",
            memory_id=entry.id,
            status="completed",
        )
        return entry

    """
    四类业务 namespace
    -> 删除 active
    -> 删除 superseded

    summary-archive namespace
    -> 删除全部 Archive

    suppressions namespace
    -> 删除全部 suppression
    """
    async def clear_all_memories(self) -> int:
        ops: list[PutOp] = []
        deleted_memory_count = 0

        for kind in _ENTRY_KINDS:
            entries = await self._load_all_entries( # 不只加载 active
                kind,
                batch_size=settings.memory_page_size,
            )

            for entry in entries:
                ops.append(
                    PutOp(
                        self._entry_namespace(entry.kind),
                        entry.id,
                        None,
                    )
                )
                deleted_memory_count += 1

        archives = await self._load_all_summary_archives(
            batch_size=settings.memory_page_size,
        )

        for archive in archives:
            ops.append(
                PutOp(
                    self._summary_archive_namespace(
                        archive.thread_id
                    ),
                    archive.summary_hash,
                    None,
                )
            )

        suppressions = await self._load_all_suppressions(
            batch_size=settings.memory_page_size,
        )

        for suppression in suppressions:
            ops.append(
                PutOp(
                    self._suppression_namespace(),
                    suppression.suppression_id,
                    None,
                )
            )

        if ops:
            await self._store.abatch(ops)
            self._mark_store_changed()

        await self.rebuild_projections()

        audit_event(
            "memory.clear",
            "completed",
        )
        log_event(
            logger,
            logging.INFO,
            "memory.cleared",
            status="completed",
        )
        return deleted_memory_count
    async def _archive_is_recallable(
        self,
        archive: SummaryArchive,
    ) -> bool:
        if not archive.recallable:
            return False

        if not archive.retrieval_summary.strip():
            return False

        if not archive.memory_keys:
            return False

        suppression_namespace = self._suppression_namespace()
        # 这里的 archive.memory_keys 实际保存的是稳定 identity：user:global:user.phone
        for memory_key in archive.memory_keys:
            item = await self._store.aget(
                suppression_namespace,
                memory_key,
            )

            if item is not None:
                return False

        return True
