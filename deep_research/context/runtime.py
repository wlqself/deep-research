from dataclasses import dataclass, field
from threading import BoundedSemaphore, Lock

from pydantic import SkipValidation

from ..config import settings
from ..sources import PersistentSource


@dataclass(slots=True) # 用 __slots__ 节省内存，防止动态加属性
class ResearchContext: # 运行时的上下文
    max_search_calls: int # 最多搜几次
    max_page_reads: int # 最多读几页
    max_page_chars: int
    max_parallel_research_tasks: int = 2 # 最多几个子agent干活
    search_count: int = 0
    page_read_count: int = 0

    next_source_number: int = 1
    new_sources: dict[str, PersistentSource] = field(default_factory=dict)
    # 保护 next_source_number 自增 + new_sources 写入（防止两个线程拿到同一个 S 编号）
    _source_lock: SkipValidation[Lock] = field(
        default_factory=Lock,
        repr=False,
    )
    # 保护 search_count / page_read_count 的读写（防止超预算）
    _budget_lock: SkipValidation[Lock] = field(
        default_factory=Lock,
        repr=False,
    )
    # 限制同时运行的 research 任务数
    _task_semaphore: SkipValidation[BoundedSemaphore] = field(
        init=False,
        repr=False,
    )
    # 从全局配置创建实例，避免到处传 settings。
    @classmethod
    def from_settings(
        cls,
        *,
        next_source_number: int = 1,
    ) -> "ResearchContext":

        return cls(
            max_search_calls=settings.max_search_calls,
            max_page_reads=settings.max_page_reads,
            max_page_chars=settings.max_page_chars,
            next_source_number=max(next_source_number, 1),
            max_parallel_research_tasks=(
                settings.max_parallel_research_tasks
            ),
        )

    def register_source(
        self,
        title: str,
        url: str,
        snippet: str,
        *,
        source_type: str | None = None,
        document_id: str | None = None,
        chunk_id: str | None = None,
        page_number: int | None = None,
        section_title: str | None = None,
    ) -> PersistentSource:
        with self._source_lock:
            source_id = f"S{self.next_source_number}"
            self.next_source_number += 1

            source: PersistentSource = {
                "source_id": source_id,
                "title": title,
                "url": url,
                "snippet": snippet,
            }

            if source_type is not None:
                source["source_type"] = source_type

            if document_id is not None:
                source["document_id"] = document_id

            if chunk_id is not None:
                source["chunk_id"] = chunk_id

            if page_number is not None:
                source["page_number"] = page_number

            if section_title is not None:
                source["section_title"] = section_title

            self.new_sources[source_id] = source
            return source

    def try_acquire_search_slot(self) -> bool:
        with self._budget_lock:
            if self.search_count >= self.max_search_calls:
                return False

            self.search_count += 1
            return True


    def try_acquire_page_read_slot(self) -> bool:
        with self._budget_lock:
            if self.page_read_count >= self.max_page_reads:
                return False

            self.page_read_count += 1
            return True
    def __post_init__(self) -> None:
        self._task_semaphore = BoundedSemaphore(
            self.max_parallel_research_tasks
        )

    def try_acquire_research_task_slot(self) -> bool:
        return self._task_semaphore.acquire(blocking=False)

    def release_research_task_slot(self) -> None:
        self._task_semaphore.release()
