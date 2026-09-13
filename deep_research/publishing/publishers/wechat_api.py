"""Small, server-side primitives for the WeChat Official Account API."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import httpx

# 异常与凭证数据类
class WeChatTokenError(RuntimeError):
    """Raised when a WeChat access token cannot be obtained safely."""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        provider_code: int | str | None = None,
        provider_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.provider_code = provider_code
        self.provider_message = provider_message


class WeChatApiError(RuntimeError):
    """Raised when a WeChat API request fails with a safe error code."""

    def __init__(
        self,
        error_code: str,
        message: str,
        *,
        provider_code: int | str | None = None,
        provider_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.provider_code = provider_code
        self.provider_message = provider_message


class WeChatDeliveryUnknownError(WeChatApiError):
    """The provider request may have been accepted but no receipt arrived."""


def _raise_request_error(error: Exception, *, error_code: str, message: str) -> None:
    """Raise a safe deterministic error without copying provider text."""
    if isinstance(error, httpx.HTTPStatusError):
        raise WeChatApiError(error_code, message) from error
    if isinstance(error, httpx.TransportError):
        raise WeChatDeliveryUnknownError(error_code.replace("request_failed", "delivery_unknown"), message) from error
    raise WeChatApiError(error_code, message) from error

# 不可变的凭证数据类，存放 app_id 和 app_secret
@dataclass(frozen=True)
class WeChatCredentials:
    """Credentials kept in application configuration, never in publishing DB."""

    app_id: str
    app_secret: str = field(repr=False) # field(repr=False)：secret 不会出现在 repr() 输出中，防止日志泄露
    # 构造时校验两者非空。
    def __post_init__(self) -> None:
        if not isinstance(self.app_id, str) or not self.app_id.strip():
            raise ValueError("app_id must not be empty")
        if not isinstance(self.app_secret, str) or not self.app_secret.strip():
            raise ValueError("app_secret must not be empty")

# 不可变的 token 数据类，包含 token 字符串和绝对过期时间
@dataclass(frozen=True)
class WeChatAccessToken:
    """An access token and its absolute expiry time."""

    value: str = field(repr=False)
    expires_at: datetime


@dataclass(frozen=True)
class WeChatPublishStatus:
    """Safe subset of the asynchronous WeChat publish status response."""

    publish_id: str
    publish_status: int
    article_id: str | None = None
    article_url: str | None = None
    fail_indexes: tuple[int, ...] = ()

# 定义 HTTP 客户端的协议接口（结构化类型）
class _TokenHttpClient(Protocol):
    # 方便依赖注入和单元测试
    def post(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        json: dict[str, object] | None = None,
        files: Any = None,
        timeout: float,
    ) -> httpx.Response:
        ...

# 负责获取和缓存微信稳定版 access_token
class WeChatAccessTokenProvider:
    """Fetch and cache the stable WeChat service access token in memory.

    No request is made during construction.  The optional HTTP client exists
    for deterministic tests and later application-level dependency injection.
    The default client is created only when ``get_access_token`` is called.
    """

    TOKEN_URL = "https://api.weixin.qq.com/cgi-bin/stable_token"
    # 构造时接收凭证、可选 HTTP 客户端、token URL、超时时间和刷新偏移量
    # 刷新偏移量： 在 token 真正过期前 5 分钟就主动刷新，避免"临界区"使用到已过期的 token
    def __init__(
        self,
        credentials: WeChatCredentials,
        *,
        client: _TokenHttpClient | None = None,
        token_url: str = TOKEN_URL,
        timeout_seconds: float = 10.0,
        refresh_skew_seconds: int = 300,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if refresh_skew_seconds < 0:
            raise ValueError("refresh_skew_seconds must not be negative")
        # token 只缓存在内存中，不持久化到数据库（安全考量：token 是短期凭证，不需要持久化）
        self._credentials = credentials
        self._client = client
        self._token_url = token_url
        self._timeout_seconds = timeout_seconds
        self._refresh_skew = timedelta(seconds=refresh_skew_seconds)
        self._cached_token: WeChatAccessToken | None = None
        # 使用 threading.RLock() 保证多线程环境下的线程安全
        self._lock = threading.RLock()
    # 核心获取逻辑
    def get_access_token(
        self,
        *,
        force_refresh: bool = False,
        now: datetime | None = None,
    ) -> str:
        """Return a cached token or fetch a new one before the safety window."""

        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        # 加锁保证线程安全
        with self._lock:
            # 如果不需要强制刷新且缓存的 token 仍然可用（没过期且在偏移量窗口内），直接返回缓存值
            if not force_refresh and self._is_usable(current_time):
                assert self._cached_token is not None
                return self._cached_token.value
            # 调用 _request_token() 从微信服务器获取新 token
            token = self._request_token()
            self._cached_token = WeChatAccessToken(
                value=token["access_token"],
                # 计算绝对过期时间（当前时间 + expires_in 秒），缓存并返回
                expires_at=current_time + timedelta(
                    seconds=token["expires_in"]
                ),
            )
            return self._cached_token.value
    # 清除内存中的缓存 token（不影响微信服务器端的 token），用于测试或强制重新认证
    def clear(self) -> None:
        """Forget the in-memory token without touching any external system."""

        with self._lock:
            self._cached_token = None
    # 判断缓存是否可用：token 存在 且​ 当前时间早于"过期时间 - 偏移量"
    def _is_usable(self, now: datetime) -> bool:
        return (
            self._cached_token is not None
            and now < self._cached_token.expires_at - self._refresh_skew
        )
    # 构造微信 API 要求的请求体
    def _request_token(self) -> dict[str, Any]:
        payload = {
            "grant_type": "client_credential",
            "appid": self._credentials.app_id,
            "secret": self._credentials.app_secret,
            "force_refresh": False,
        }
        # 如果注入了自定义客户端就用注入的，否则创建临时 httpx.Client
        try:
            if self._client is not None:
                response = self._client.post(
                    self._token_url,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.post(self._token_url, json=payload)
            response.raise_for_status()
            result = response.json()
        # 捕获网络/解析错误，包装为 WeChatTokenError
        except WeChatTokenError:
            raise
        except (httpx.HTTPError, ValueError, TypeError) as error:
            raise WeChatTokenError(
                "wechat_token_request_failed",
                "WeChat access token request failed",
            ) from error

        if not isinstance(result, dict):
            raise WeChatTokenError(
                "wechat_token_response_invalid",
                "WeChat access token response was invalid",
            )

        if result.get("errcode") not in (None, 0):
            raise WeChatTokenError(
                "wechat_token_rejected",
                "WeChat rejected the access token request",
                provider_code=(
                    result.get("errcode")
                    if isinstance(result.get("errcode"), (int, str))
                    else None
                ),
                provider_message=(
                    result.get("errmsg")
                    if isinstance(result.get("errmsg"), str)
                    else None
                ),
            )
        # 校验响应字段的类型和值，防止畸形响应导致后续逻辑出错。
        access_token = result.get("access_token")
        expires_in = result.get("expires_in")
        if (
            not isinstance(access_token, str)
            or not access_token.strip()
            or not isinstance(expires_in, int)
            or expires_in <= 0
        ):
            raise WeChatTokenError(
                "wechat_token_response_invalid",
                "WeChat access token response was incomplete",
            )

        return {
            "access_token": access_token,
            "expires_in": expires_in,
        }


class WeChatOfficialAccountClient:
    """Call narrowly scoped Official Account endpoints through a token provider."""

    DRAFT_ADD_PATH = "/cgi-bin/draft/add"
    PERMANENT_MATERIAL_PATH = "/cgi-bin/material/add_material"
    INLINE_IMAGE_PATH = "/cgi-bin/media/uploadimg"
    FREE_PUBLISH_SUBMIT_PATH = "/cgi-bin/freepublish/submit"
    FREE_PUBLISH_STATUS_PATH = "/cgi-bin/freepublish/get"
    MAX_INLINE_IMAGE_BYTES = 1_000_000

    def __init__(
        self,
        token_provider: WeChatAccessTokenProvider,
        *,
        client: _TokenHttpClient | None = None,
        api_base_url: str = "https://api.weixin.qq.com",
        timeout_seconds: float = 10.0,
    ) -> None:
        if not api_base_url.startswith("https://"):
            raise ValueError("api_base_url must use HTTPS")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self._token_provider = token_provider
        self._client = client
        self._api_base_url = api_base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def add_draft(self, payload: dict[str, Any]) -> str:
        """Create a WeChat draft and return its opaque media ID."""

        access_token = self._token_provider.get_access_token()
        url = f"{self._api_base_url}{self.DRAFT_ADD_PATH}"

        try:
            if self._client is not None:
                response = self._client.post(
                    url,
                    params={"access_token": access_token},
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.post(
                        url,
                        params={"access_token": access_token},
                        json=payload,
                    )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPStatusError, httpx.TransportError) as error:
            _raise_request_error(
                error,
                error_code="wechat_draft_request_failed",
                message="WeChat draft request failed",
            )
        except (ValueError, TypeError) as error:
            raise WeChatApiError(
                "wechat_draft_request_failed",
                "WeChat draft request failed",
            ) from error

        if not isinstance(result, dict):
            raise WeChatApiError(
                "wechat_draft_response_invalid",
                "WeChat draft response was invalid",
            )

        if result.get("errcode") not in (None, 0):
            raise WeChatApiError(
                "wechat_draft_rejected",
                "WeChat rejected the draft request",
                provider_code=(
                    result.get("errcode")
                    if isinstance(result.get("errcode"), (int, str))
                    else None
                ),
            )

        media_id = result.get("media_id")
        if not isinstance(media_id, str) or not media_id.strip():
            raise WeChatApiError(
                "wechat_draft_response_invalid",
                "WeChat draft response did not include a media id",
            )

        return media_id.strip()

    def upload_cover(
        self,
        content: bytes,
        *,
        filename: str,
        content_type: str = "image/jpeg",
    ) -> str:
        """Upload controlled cover bytes and return the permanent media ID."""

        if not isinstance(content, bytes) or not content:
            raise WeChatApiError(
                "wechat_cover_invalid",
                "WeChat cover content is invalid",
            )
        if (
            not isinstance(filename, str)
            or not filename.strip()
            or "/" in filename
            or "\\" in filename
        ):
            raise WeChatApiError(
                "wechat_cover_filename_invalid",
                "WeChat cover filename is invalid",
            )
        if content_type not in {"image/jpeg", "image/png", "image/gif"}:
            raise WeChatApiError(
                "wechat_cover_type_invalid",
                "WeChat cover type is unsupported",
            )

        access_token = self._token_provider.get_access_token()
        url = f"{self._api_base_url}{self.PERMANENT_MATERIAL_PATH}"
        files = {
            "media": (
                filename.strip(),
                content,
                content_type,
            )
        }

        try:
            if self._client is not None:
                response = self._client.post(
                    url,
                    params={
                        "access_token": access_token,
                        "type": "thumb",
                    },
                    files=files,
                    timeout=self._timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.post(
                        url,
                        params={
                            "access_token": access_token,
                            "type": "thumb",
                        },
                        files=files,
                    )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPStatusError, httpx.TransportError) as error:
            # Cover upload is an operator action; it is not an article
            # publication side effect and remains an ordinary API failure.
            raise WeChatApiError(
                "wechat_cover_request_failed",
                "WeChat cover upload failed",
            ) from error
        except (ValueError, TypeError) as error:
            raise WeChatApiError(
                "wechat_cover_request_failed",
                "WeChat cover upload failed",
            ) from error

        if not isinstance(result, dict):
            raise WeChatApiError(
                "wechat_cover_response_invalid",
                "WeChat cover response was invalid",
            )

        if result.get("errcode") not in (None, 0):
            raise WeChatApiError(
                "wechat_cover_rejected",
                "WeChat rejected the cover upload",
                provider_code=(
                    result.get("errcode")
                    if isinstance(result.get("errcode"), (int, str))
                    else None
                ),
                provider_message=(
                    result.get("errmsg")
                    if isinstance(result.get("errmsg"), str)
                    else None
                ),
            )

        media_id = result.get("media_id")
        if not isinstance(media_id, str) or not media_id.strip():
            raise WeChatApiError(
                "wechat_cover_response_invalid",
                "WeChat cover response did not include a media id",
            )

        return media_id.strip()

    def upload_inline_image(
        self,
        content: bytes,
        *,
        filename: str,
        content_type: str = "image/jpeg",
    ) -> str:
        """Upload controlled article-image bytes and return its WeChat URL."""

        if (
            not isinstance(content, bytes)
            or not content
            or len(content) > self.MAX_INLINE_IMAGE_BYTES
        ):
            raise WeChatApiError(
                "wechat_inline_image_invalid",
                "WeChat inline image content is invalid",
            )
        if (
            not isinstance(filename, str)
            or not filename.strip()
            or "/" in filename
            or "\\" in filename
        ):
            raise WeChatApiError(
                "wechat_inline_image_filename_invalid",
                "WeChat inline image filename is invalid",
            )
        if content_type not in {"image/jpeg", "image/png"}:
            raise WeChatApiError(
                "wechat_inline_image_type_invalid",
                "WeChat inline image type is unsupported",
            )

        access_token = self._token_provider.get_access_token()
        url = f"{self._api_base_url}{self.INLINE_IMAGE_PATH}"
        files = {
            "media": (
                filename.strip(),
                content,
                content_type,
            )
        }

        try:
            if self._client is not None:
                response = self._client.post(
                    url,
                    params={"access_token": access_token},
                    files=files,
                    timeout=self._timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.post(
                        url,
                        params={"access_token": access_token},
                        files=files,
                    )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPStatusError, httpx.TransportError) as error:
            raise WeChatApiError(
                "wechat_inline_image_request_failed",
                "WeChat inline image upload failed",
            ) from error
        except (ValueError, TypeError) as error:
            raise WeChatApiError(
                "wechat_inline_image_request_failed",
                "WeChat inline image upload failed",
            ) from error

        if not isinstance(result, dict):
            raise WeChatApiError(
                "wechat_inline_image_response_invalid",
                "WeChat inline image response was invalid",
            )

        if result.get("errcode") not in (None, 0):
            raise WeChatApiError(
                "wechat_inline_image_rejected",
                "WeChat rejected the inline image upload",
                provider_code=(
                    result.get("errcode")
                    if isinstance(result.get("errcode"), (int, str))
                    else None
                ),
            )

        image_url = result.get("url")
        if not isinstance(image_url, str) or not image_url.strip():
            raise WeChatApiError(
                "wechat_inline_image_response_invalid",
                "WeChat inline image response did not include a URL",
            )

        return image_url.strip()

    def submit_draft(self, media_id: str) -> str:
        """Submit one existing WeChat draft and return its publish ID."""

        if not isinstance(media_id, str) or not media_id.strip():
            raise WeChatApiError(
                "wechat_draft_media_id_invalid",
                "WeChat draft media id is invalid",
            )

        access_token = self._token_provider.get_access_token()
        url = f"{self._api_base_url}{self.FREE_PUBLISH_SUBMIT_PATH}"
        payload = {"media_id": media_id.strip()}

        try:
            if self._client is not None:
                response = self._client.post(
                    url,
                    params={"access_token": access_token},
                    json=payload,
                    timeout=self._timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.post(
                        url,
                        params={"access_token": access_token},
                        json=payload,
                    )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPStatusError, httpx.TransportError) as error:
            _raise_request_error(
                error,
                error_code="wechat_publish_request_failed",
                message="WeChat publish request failed",
            )
        except (ValueError, TypeError) as error:
            raise WeChatApiError(
                "wechat_publish_request_failed",
                "WeChat publish request failed",
            ) from error

        if not isinstance(result, dict):
            raise WeChatApiError(
                "wechat_publish_response_invalid",
                "WeChat publish response was invalid",
            )

        if result.get("errcode") not in (None, 0):
            raise WeChatApiError(
                "wechat_publish_rejected",
                "WeChat rejected the publish request",
                provider_code=(
                    result.get("errcode")
                    if isinstance(result.get("errcode"), (int, str))
                    else None
                ),
            )

        publish_id = result.get("publish_id")
        if not isinstance(publish_id, str) or not publish_id.strip():
            raise WeChatApiError(
                "wechat_publish_response_invalid",
                "WeChat publish response did not include a publish id",
            )

        return publish_id.strip()

    def get_publish_status(self, publish_id: str) -> WeChatPublishStatus:
        """Read the current status of one asynchronous WeChat publish task."""

        if not isinstance(publish_id, str) or not publish_id.strip():
            raise WeChatApiError(
                "wechat_publish_id_invalid",
                "WeChat publish id is invalid",
            )

        access_token = self._token_provider.get_access_token()
        url = f"{self._api_base_url}{self.FREE_PUBLISH_STATUS_PATH}"

        try:
            if self._client is not None:
                response = self._client.post(
                    url,
                    params={"access_token": access_token},
                    json={"publish_id": publish_id.strip()},
                    timeout=self._timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.post(
                        url,
                        params={"access_token": access_token},
                        json={"publish_id": publish_id.strip()},
                    )
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as error:
            raise WeChatApiError(
                "wechat_publish_status_request_failed",
                "WeChat publish status request failed",
            ) from error

        if not isinstance(result, dict):
            raise WeChatApiError(
                "wechat_publish_status_response_invalid",
                "WeChat publish status response was invalid",
            )

        if result.get("errcode") not in (None, 0):
            raise WeChatApiError(
                "wechat_publish_status_rejected",
                "WeChat rejected the publish status request",
                provider_code=(
                    result.get("errcode")
                    if isinstance(result.get("errcode"), (int, str))
                    else None
                ),
            )

        publish_status = result.get("publish_status")
        if isinstance(publish_status, bool) or not isinstance(
            publish_status,
            int,
        ):
            raise WeChatApiError(
                "wechat_publish_status_response_invalid",
                "WeChat publish status was invalid",
            )

        article_id = result.get("article_id")
        if article_id is not None and not isinstance(article_id, str):
            article_id = None

        article_url: str | None = None
        article_detail = result.get("article_detail")
        if isinstance(article_detail, dict):
            items = article_detail.get("item")
            if isinstance(items, list) and items:
                first_item = items[0]
                if isinstance(first_item, dict):
                    candidate_url = first_item.get("article_url")
                    if isinstance(candidate_url, str) and candidate_url.strip():
                        article_url = candidate_url.strip()

        raw_fail_indexes = result.get("fail_idx")
        fail_indexes: tuple[int, ...] = ()
        if isinstance(raw_fail_indexes, list):
            fail_indexes = tuple(
                value
                for value in raw_fail_indexes
                if isinstance(value, int) and not isinstance(value, bool)
            )

        return WeChatPublishStatus(
            publish_id=publish_id.strip(),
            publish_status=publish_status,
            article_id=article_id,
            article_url=article_url,
            fail_indexes=fail_indexes,
        )


__all__ = [
    "WeChatAccessToken",
    "WeChatAccessTokenProvider",
    "WeChatApiError",
    "WeChatDeliveryUnknownError",
    "WeChatCredentials",
    "WeChatOfficialAccountClient",
    "WeChatPublishStatus",
    "WeChatTokenError",
]
