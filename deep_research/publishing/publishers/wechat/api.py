"""Compatibility-free package boundary for the WeChat API client."""

from ..wechat_api import (
    WeChatAccessToken,
    WeChatAccessTokenProvider,
    WeChatApiError,
    WeChatDeliveryUnknownError,
    WeChatCredentials,
    WeChatOfficialAccountClient,
    WeChatPublishStatus,
    WeChatTokenError,
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
