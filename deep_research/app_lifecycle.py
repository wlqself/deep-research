import logging
import inspect
import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from .hitl.repository import HITLRepository
from .hitl.decision_service import HITLDecisionService
from .hitl.service import HITLService
from .log.logging_utils import log_event
from .publishing.artifacts import ArtifactService
from .publishing.publishers.local_static import LocalStaticPublisher
from .publishing.publishers.wechat.publisher import WeChatOfficialAccountPublisher
from .publishing.publishers.wechat_api import (
    WeChatAccessTokenProvider,
    WeChatCredentials,
    WeChatOfficialAccountClient,
)
from .publishing.publishers.xiaohongshu import (
    XiaohongshuClient,
    XiaohongshuPublisher,
)
from .publishing.publishers.douyin import (
    DouyinClient,
    DouyinPublisher,
)
from .publishing.repository import PublishingRepository
from .publishing.service import PublishingService
from .publishing.service_support import PublicationValidationError
from .publishing.models import PublicationChannel
from .publishing.wechat_covers import WeChatCoverService
from .publishing.attachments import ImageAttachmentService
from .publishing.image_analysis import ImageAnalysisService
from .publishing.image_generation import SiliconFlowImageGenerationService


async def _publication_recovery_loop(
    service: PublishingService,
    publishers: dict[PublicationChannel, Any],
    *,
    interval_seconds: float = 30.0,
    max_attempts: int = 12,
) -> None:
    """Continuously poll durable in-flight publications for every channel."""

    recovery_logger = logging.getLogger("deep_research.publishing.recovery")
    while True:
        try:
            for channel, publisher in publishers.items():
                if publisher is None or not callable(
                    getattr(publisher, "get_publication_status", None)
                ):
                    continue

                list_recoverable = getattr(
                    service,
                    "list_recoverable_publications",
                    None,
                )
                if callable(list_recoverable):
                    publications = list_recoverable(channel=channel)
                else:
                    publications = service.repository.list_incomplete_publications(
                        channel=channel,
                    )

                for publication in publications:
                    if publication.attempt_count >= max_attempts:
                        recovery_logger.warning(
                            "publication recovery attempt limit reached",
                            extra={
                                "publication_id": publication.publication_id,
                                "channel": channel.value,
                                "attempt_count": publication.attempt_count,
                            },
                        )
                        continue
                    try:
                        result = await asyncio.to_thread(
                            service.resume_publication_for_article,
                            publication.article_id,
                            channel=channel,
                            publisher=publisher,
                        )
                        if inspect.isawaitable(result):
                            await result
                    except Exception:
                        # Recovery is best effort; the durable record remains
                        # publishing and will be considered on the next pass.
                        recovery_logger.warning(
                            "publication recovery attempt failed",
                            extra={
                                "publication_id": publication.publication_id,
                                "channel": channel.value,
                            },
                            exc_info=False,
                        )
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            recovery_logger.warning(
                "publication recovery loop iteration failed",
                exc_info=False,
            )
            await asyncio.sleep(interval_seconds)


# Kept as a compatibility wrapper for callers that imported the old helper.
async def _wechat_recovery_loop(
    service: PublishingService,
    publisher: Any,
    *,
    interval_seconds: float = 30.0,
    max_attempts: int = 12,
) -> None:
    await _publication_recovery_loop(
        service,
        {PublicationChannel.WECHAT_OFFICIAL_ACCOUNT: publisher},
        interval_seconds=interval_seconds,
        max_attempts=max_attempts,
    )


class _LiveAgentStateReader:
    """Resolve the current application Agent when an Artifact is read."""

    def __init__(self, agent_module: Any) -> None:
        self._agent_module = agent_module

    async def aget_state(self, config: dict[str, object]) -> Any:
        return await self._agent_module.agent.aget_state(config)


def _build_wechat_publisher(
    settings: Any,
    *,
    cover_media_id_resolver: Callable[[], str | None] | None = None,
    attachment_resolver: Callable[[Any, tuple[str, ...]], list[tuple[str, str]]] | None = None,
    cover_available: bool | None = None,
) -> WeChatOfficialAccountPublisher | None:
    """Build the optional WeChat publisher without making a network request."""

    app_id = getattr(settings, "wechat_app_id", None)
    app_secret = getattr(settings, "wechat_app_secret", None)
    thumb_media_id = getattr(settings, "wechat_thumb_media_id", None)
    credentials_ready = all(
        isinstance(value, str) and value.strip()
        for value in (app_id, app_secret)
    )
    if cover_available is None:
        cover_available = True
    if not credentials_ready or (
        not cover_available and cover_media_id_resolver is None
    ):
        return None

    credentials = WeChatCredentials(
        app_id=app_id,
        app_secret=app_secret,
    )
    token_provider = WeChatAccessTokenProvider(credentials)
    client = WeChatOfficialAccountClient(token_provider)
    return WeChatOfficialAccountPublisher(
        client,
        thumb_media_id=(
            thumb_media_id.strip()
            if isinstance(thumb_media_id, str) and thumb_media_id.strip()
            else None
        ),
        cover_media_id_resolver=cover_media_id_resolver,
        attachment_resolver=attachment_resolver,
        author=getattr(settings, "wechat_author", None),
        publish_mode=getattr(settings, "wechat_publish_mode", "draft"),
        theme=getattr(settings, "wechat_theme", "minimal"),
    )


def _build_xiaohongshu_publisher(
    settings: Any,
    repository: PublishingRepository,
) -> XiaohongshuPublisher | None:
    """Build the optional local Xiaohongshu adapter without network I/O."""

    base_url = getattr(settings, "xiaohongshu_base_url", None)
    if not isinstance(base_url, str) or not base_url.strip():
        return None
    client = XiaohongshuClient(
        base_url,
        timeout_seconds=getattr(settings, "xiaohongshu_timeout_seconds", 30),
    )
    def resolve_image_paths(article: Any, attachment_ids: tuple[str, ...]) -> list[str]:
        paths: list[str] = []
        for attachment_id in attachment_ids:
            try:
                attachment = repository.get_image_attachment(attachment_id)
            except Exception as error:
                raise PublicationValidationError(
                    "xiaohongshu_image_not_found",
                    "Xiaohongshu image attachment does not exist",
                ) from error
            paths.append(attachment.storage_path)
        return paths

    return XiaohongshuPublisher(
        client,
        image_paths_resolver=resolve_image_paths,
        max_images=getattr(settings, "xiaohongshu_max_images", 9),
        title_max_length=getattr(settings, "xiaohongshu_title_max_length", 20),
        content_max_length=getattr(settings, "xiaohongshu_content_max_length", 1000),
    )


def _build_douyin_publisher(
    settings: Any,
    repository: PublishingRepository,
) -> DouyinPublisher | None:
    """Build the optional local Douyin adapter without network I/O."""

    base_url = getattr(settings, "douyin_base_url", None)
    if not isinstance(base_url, str) or not base_url.strip():
        return None
    client = DouyinClient(
        base_url,
        timeout_seconds=getattr(settings, "douyin_timeout_seconds", 30),
    )

    def resolve_image_paths(article: Any, attachment_ids: tuple[str, ...]) -> list[str]:
        paths: list[str] = []
        for attachment_id in attachment_ids:
            try:
                attachment = repository.get_image_attachment(attachment_id)
            except Exception as error:
                raise PublicationValidationError(
                    "douyin_image_not_found",
                    "Douyin image attachment does not exist",
                ) from error
            paths.append(attachment.storage_path)
        return paths

    return DouyinPublisher(
        client,
        image_paths_resolver=resolve_image_paths,
        max_images=getattr(settings, "douyin_max_images", 30),
        title_max_length=getattr(settings, "douyin_title_max_length", 30),
    )


def _build_agent_with_optional_publishing(
    *,
    agent_module: Any,
    checkpointer: Any,
    rag_service: Any,
    memory_service: Any,
    publishing_service: PublishingService,
    hitl_service: Any = None,
    image_attachment_service: Any = None,
    image_analysis_service: Any = None,
    wechat_cover_service: Any = None,
    image_generation_service: Any = None,
) -> Any:
    """Build the Agent while keeping older injected factories compatible."""

    build_agent = agent_module.build_agent
    try:
        parameters = inspect.signature(build_agent).parameters
    except (TypeError, ValueError):
        parameters = {}

    optional_arguments = {}
    if "publishing_service" in parameters:
        optional_arguments["publishing_service"] = publishing_service
    if "hitl_service" in parameters:
        optional_arguments["hitl_service"] = hitl_service
    if "image_attachment_service" in parameters:
        optional_arguments["image_attachment_service"] = image_attachment_service
    if "image_analysis_service" in parameters:
        optional_arguments["image_analysis_service"] = image_analysis_service
    if "wechat_cover_service" in parameters:
        optional_arguments["wechat_cover_service"] = wechat_cover_service
    if "image_generation_service" in parameters:
        optional_arguments["image_generation_service"] = image_generation_service

    if optional_arguments:
        return build_agent(
            checkpointer,
            rag_service,
            memory_service,
            **optional_arguments,
        )

    return build_agent(
        checkpointer,
        rag_service,
        memory_service,
    )


def build_lifespan(
    *,
    settings: Any,
    agent_module: Any,
    document_registry_factory: Callable[[str], Any],
    rag_service_factory: Callable[[Any], Any],
    checkpointer_factory: Callable[[], Any],
    memory_store_factory: Callable[[], Any],
    memory_service_factory: Callable[..., Any],
    memory_extractor_factory: Callable[[Any], Any],
    model: Any,
):
    logger = logging.getLogger("deep_research.app")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        previous_agent = agent_module.agent

        #  初始化文档注册表
        registry = document_registry_factory(
            settings.rag_registry_db_path
        )
        registry.initialize()

        #  创建 RAG 服务
        rag_service = rag_service_factory(settings)

        # 记录启动日志（Qdrant 存储）
        log_event(
            logger,
            logging.INFO,
            "store.qdrant.started",
            status="started",
        )

        publishing_repository: PublishingRepository | None = None
        hitl_repository: HITLRepository | None = None
        wechat_publisher: WeChatOfficialAccountPublisher | None = None
        xiaohongshu_publisher: XiaohongshuPublisher | None = None
        douyin_publisher: DouyinPublisher | None = None
        wechat_cover_service: WeChatCoverService | None = None
        image_attachment_service: ImageAttachmentService | None = None
        image_analysis_service: ImageAnalysisService | None = None
        image_generation_service: SiliconFlowImageGenerationService | None = None
        recovery_task: asyncio.Task[None] | None = None

        # 挂到 app.state 上
        app.state.document_registry = registry
        app.state.rag_service = rag_service

        # 嵌套异步上下文管理器（checkpointer + memory_store）
        try:
            publishing_repository = PublishingRepository(
                settings.publishing_db_path
            )
            publishing_repository.initialize()
            image_attachment_service = ImageAttachmentService(
                publishing_repository,
                getattr(settings, "publishing_upload_dir", "data/publishing-uploads"),
                max_bytes=getattr(settings, "publishing_upload_max_bytes", 10_485_760),
            )
            image_analysis_service = ImageAnalysisService(
                publishing_repository,
                model_name=(
                    getattr(settings, "image_analysis_model_name", None)
                    or getattr(settings, "model_name", None)
                ),
                api_key=(
                    getattr(settings, "image_analysis_api_key", None)
                    or getattr(settings, "model_api_key", None)
                ),
                base_url=(
                    getattr(settings, "image_analysis_base_url", None)
                    or getattr(settings, "model_base_url", None)
                ),
                enabled=getattr(settings, "image_analysis_enabled", True),
                timeout_seconds=getattr(
                    settings,
                    "image_analysis_timeout_seconds",
                    60,
                ),
            )
            image_generation_service = SiliconFlowImageGenerationService(
                image_attachment_service,
                model_name=getattr(settings, "image_generation_model_name", None),
                api_key=(
                    getattr(settings, "image_generation_api_key", None)
                    or getattr(settings, "model_api_key", None)
                ),
                base_url=getattr(settings, "image_generation_base_url", None),
                enabled=getattr(settings, "image_generation_enabled", True),
                timeout_seconds=getattr(settings, "image_generation_timeout_seconds", 180),
                prompt_max_chars=getattr(settings, "image_generation_prompt_max_chars", 800),
                max_images=getattr(settings, "image_generation_max_images", 4),
            )
            for existing_attachment in publishing_repository.list_image_attachments():
                # A missing/pending record means the user uploaded an image
                # but has not sent it to the Agent yet. Only resume work that
                # was already actively analyzing when the service stopped.
                existing_analysis = publishing_repository.get_image_attachment_analysis(
                    existing_attachment.attachment_id,
                )
                if existing_analysis is not None and existing_analysis.status == "analyzing":
                    image_analysis_service.schedule(existing_attachment.attachment_id)
            hitl_repository = HITLRepository(settings.hitl_db_path)
            hitl_repository.initialize()
            log_event(
                logger,
                logging.INFO,
                "store.hitl.started",
                status="started",
            )
            local_static_publisher = LocalStaticPublisher(
                settings.published_site_dir,
                attachment_resolver=(
                    lambda article, attachment_ids: [
                        (
                            attachment.storage_path,
                            attachment.content_type,
                        )
                        for attachment in (
                            publishing_repository.get_image_attachment(attachment_id)
                            for attachment_id in attachment_ids
                        )
                    ]
                ),
            )
            configured_cover_id = getattr(settings, "wechat_thumb_media_id", None)
            active_cover = publishing_repository.get_active_wechat_cover_asset()

            def resolve_cover_media_id() -> str | None:
                asset = publishing_repository.get_active_wechat_cover_asset()
                if asset is not None:
                    return asset.remote_media_id
                return (
                    configured_cover_id.strip()
                    if isinstance(configured_cover_id, str)
                    and configured_cover_id.strip()
                    else None
                )

            def resolve_wechat_attachments(
                article: Any,
                attachment_ids: tuple[str, ...],
            ) -> list[tuple[str, str]]:
                resolved: list[tuple[str, str]] = []
                for attachment_id in attachment_ids:
                    try:
                        attachment = publishing_repository.get_image_attachment(
                            attachment_id
                        )
                    except Exception as error:
                        raise PublicationValidationError(
                            "wechat_image_not_found",
                            "WeChat image attachment does not exist",
                        ) from error
                    resolved.append((attachment.storage_path, attachment.content_type))
                return resolved

            wechat_publisher = _build_wechat_publisher(
                settings,
                cover_media_id_resolver=resolve_cover_media_id,
                attachment_resolver=resolve_wechat_attachments,
                cover_available=(
                    active_cover is not None
                    or (
                        isinstance(configured_cover_id, str)
                        and bool(configured_cover_id.strip())
                    )
                ),
            )
            xiaohongshu_publisher = _build_xiaohongshu_publisher(
                settings,
                publishing_repository,
            )
            douyin_publisher = _build_douyin_publisher(
                settings,
                publishing_repository,
            )
            if wechat_publisher is not None:
                wechat_cover_service = WeChatCoverService(
                    publishing_repository,
                    wechat_publisher.client,
                )
            if wechat_publisher is None:
                log_event(
                    logger,
                    logging.INFO,
                    "publishing.wechat.disabled",
                    status="disabled",
                    error_code="wechat_configuration_incomplete",
                )
            else:
                log_event(
                    logger,
                    logging.INFO,
                    "publishing.wechat.started",
                    status="started",
                )
            if xiaohongshu_publisher is None:
                log_event(
                    logger,
                    logging.INFO,
                    "publishing.xiaohongshu.disabled",
                    status="disabled",
                    error_code="xiaohongshu_configuration_incomplete",
                )
            else:
                log_event(
                    logger,
                    logging.INFO,
                    "publishing.xiaohongshu.started",
                    status="started",
                )
            if douyin_publisher is None:
                log_event(
                    logger,
                    logging.INFO,
                    "publishing.douyin.disabled",
                    status="disabled",
                    error_code="douyin_configuration_incomplete",
                )
            else:
                log_event(
                    logger,
                    logging.INFO,
                    "publishing.douyin.started",
                    status="started",
                )

            async with checkpointer_factory() as checkpointer:
                async with memory_store_factory() as memory_store:
                    # 创建记忆服务和提取器
                    memory_service = memory_service_factory(
                        memory_store,
                        user_id=settings.memory_user_id,
                    )
                    memory_extractor = memory_extractor_factory(model)

                    app.state.memory_store = memory_store
                    app.state.memory_service = memory_service
                    app.state.memory_extractor = memory_extractor

                    app.state.artifact_service = ArtifactService(
                        _LiveAgentStateReader(agent_module)
                    )
                    app.state.publishing_repository = publishing_repository
                    app.state.hitl_repository = hitl_repository
                    app.state.hitl_service = HITLService(
                        hitl_repository,
                        expiration_seconds=getattr(
                            settings,
                            "hitl_expiration_seconds",
                            86_400,
                        ),
                    )
                    app.state.publishing_service = PublishingService(
                        publishing_repository,
                        app.state.artifact_service,
                    )
                    app.state.hitl_decision_service = HITLDecisionService(
                        app.state.hitl_service,
                        app.state.publishing_service,
                    )
                    app.state.local_static_publisher = local_static_publisher
                    app.state.wechat_publisher = wechat_publisher
                    app.state.xiaohongshu_publisher = xiaohongshu_publisher
                    app.state.douyin_publisher = douyin_publisher
                    app.state.wechat_cover_service = wechat_cover_service
                    app.state.image_attachment_service = image_attachment_service
                    app.state.image_analysis_service = image_analysis_service
                    app.state.image_generation_service = image_generation_service

                    recovery_publishers = {
                        channel: publisher
                        for channel, publisher in (
                            (
                                PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
                                wechat_publisher,
                            ),
                            (PublicationChannel.XIAOHONGSHU, xiaohongshu_publisher),
                            (PublicationChannel.DOUYIN, douyin_publisher),
                        )
                        if publisher is not None
                        and callable(getattr(publisher, "get_publication_status", None))
                    }
                    if recovery_publishers:
                        recovery_task = asyncio.create_task(
                            _publication_recovery_loop(
                                app.state.publishing_service,
                                recovery_publishers,
                                interval_seconds=getattr(
                                    settings,
                                    "publication_recovery_interval_seconds",
                                    getattr(
                                        settings,
                                        "wechat_recovery_interval_seconds",
                                        30,
                                    ),
                                ),
                                max_attempts=getattr(
                                    settings,
                                    "publication_recovery_max_attempts",
                                    getattr(
                                        settings,
                                        "wechat_recovery_max_attempts",
                                        12,
                                    ),
                                ),
                            ),
                            name="publication-recovery",
                        )
                        app.state.publication_recovery_task = recovery_task
                        # Keep the old state attribute for compatibility with
                        # existing diagnostics and tests.
                        app.state.wechat_recovery_task = recovery_task

                    agent_module.agent = _build_agent_with_optional_publishing(
                        agent_module=agent_module,
                        checkpointer=checkpointer,
                        rag_service=rag_service,
                        memory_service=memory_service,
                        publishing_service=app.state.publishing_service,
                        hitl_service=app.state.hitl_service,
                        image_attachment_service=image_attachment_service,
                        image_analysis_service=image_analysis_service,
                        wechat_cover_service=wechat_cover_service,
                        image_generation_service=image_generation_service,
                    )
                    app.state.agent = agent_module.agent

                    log_event(
                        logger,
                        logging.INFO,
                        "app.started",
                        status="started",
                    )

                    yield

        finally:
            if recovery_task is not None:
                recovery_task.cancel()
                try:
                    await recovery_task
                except asyncio.CancelledError:
                    pass
            if image_analysis_service is not None:
                await image_analysis_service.close()
            app.state.wechat_recovery_task = None
            app.state.publication_recovery_task = None
            agent_module.agent = previous_agent

            app.state.document_registry = None
            app.state.rag_service = None
            app.state.memory_store = None
            app.state.memory_service = None
            app.state.memory_extractor = None
            app.state.artifact_service = None
            app.state.agent = None
            app.state.publishing_service = None
            app.state.publishing_repository = None
            app.state.hitl_repository = None
            app.state.hitl_service = None
            app.state.hitl_decision_service = None
            app.state.local_static_publisher = None
            app.state.wechat_publisher = None
            app.state.xiaohongshu_publisher = None
            app.state.douyin_publisher = None
            app.state.wechat_cover_service = None
            app.state.image_attachment_service = None
            app.state.image_analysis_service = None
            app.state.image_generation_service = None

            if hitl_repository is not None:
                hitl_repository.close()

            if publishing_repository is not None:
                publishing_repository.close()

            if wechat_publisher is not None:
                log_event(
                    logger,
                    logging.INFO,
                    "publishing.wechat.stopped",
                    status="completed",
                )

            if xiaohongshu_publisher is not None:
                log_event(
                    logger,
                    logging.INFO,
                    "publishing.xiaohongshu.stopped",
                    status="completed",
                )

            log_event(
                logger,
                logging.INFO,
                "store.hitl.stopped",
                status="completed",
            )

            rag_service.close()

            log_event(
                logger,
                logging.INFO,
                "store.qdrant.stopped",
                status="completed",
            )

            log_event(
                logger,
                logging.INFO,
                "app.stopped",
                status="completed",
            )

    return lifespan
