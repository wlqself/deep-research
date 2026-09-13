"""Xiaohongshu publishing adapter package.

The browser automation itself lives in a separate local service.  This package
only owns the stable HTTP contract used by the publishing workflow.
"""

from .client import (
    XiaohongshuApiError,
    XiaohongshuClient,
    XiaohongshuDeliveryUnknownError,
    XiaohongshuLoginStatus,
    XiaohongshuPublishReceipt,
)
from .formatter import XiaohongshuContent, format_xiaohongshu_content
from .publisher import XiaohongshuPublisher

__all__ = [
    "XiaohongshuApiError",
    "XiaohongshuClient",
    "XiaohongshuContent",
    "XiaohongshuDeliveryUnknownError",
    "XiaohongshuLoginStatus",
    "XiaohongshuPublishReceipt",
    "XiaohongshuPublisher",
    "format_xiaohongshu_content",
]
