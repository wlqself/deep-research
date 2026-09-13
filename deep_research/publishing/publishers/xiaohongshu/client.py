"""HTTP client for a local Xiaohongshu browser-automation service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from ...models import PublicationStatus


class XiaohongshuApiError(RuntimeError):
    """A stable error returned by the local Xiaohongshu service."""

    def __init__(
        self,
        error_code: str,
        message: str = "Xiaohongshu request failed",
        *,
        http_status: int | None = None,
    ) -> None:
        self.error_code = error_code
        self.http_status = http_status
        super().__init__(message)


class XiaohongshuDeliveryUnknownError(XiaohongshuApiError):
    """The publish request may have reached the browser service without a receipt."""

    def __init__(self, error_code: str, *, external_id: str | None = None) -> None:
        self.external_id = external_id
        super().__init__(error_code)


@dataclass(frozen=True)
class XiaohongshuPublishReceipt:
    external_id: str
    status: PublicationStatus
    public_url: str | None = None


@dataclass(frozen=True)
class XiaohongshuLoginStatus:
    logged_in: bool
    raw: dict[str, Any]


class XiaohongshuClient:
    """Call a loopback REST service that owns Playwright/CDP and cookies."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        normalized = base_url.strip().rstrip("/") if isinstance(base_url, str) else ""
        if not normalized:
            raise ValueError("base_url must not be empty")
        self.base_url = normalized
        self.timeout_seconds = timeout_seconds
        self._http_client = http_client

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            if self._http_client is not None:
                response = self._http_client.request(
                    method,
                    url,
                    timeout=self.timeout_seconds,
                    **kwargs,
                )
            else:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.request(method, url, **kwargs)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as error:
            raise XiaohongshuApiError(
                "xiaohongshu_http_error",
                "Xiaohongshu service returned an HTTP error",
                http_status=error.response.status_code,
            ) from error
        except (httpx.TransportError, ValueError, TypeError) as error:
            raise XiaohongshuDeliveryUnknownError(
                "xiaohongshu_service_unreachable"
            ) from error
        if not isinstance(payload, dict):
            raise XiaohongshuApiError(
                "xiaohongshu_response_invalid",
                "Xiaohongshu service returned an invalid response",
            )
        if (
            payload.get("success") is False
            or payload.get("ok") is False
            or (payload.get("error") and payload.get("code"))
        ):
            raise XiaohongshuApiError(
                str(
                    payload.get("error_code")
                    or payload.get("code")
                    or "xiaohongshu_rejected"
                ),
                "Xiaohongshu service rejected the request",
            )
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def get_login_status(self) -> XiaohongshuLoginStatus:
        payload = self._request("GET", "/api/v1/login/status")
        nested = payload.get("data")
        if isinstance(nested, dict):
            payload = {**payload, **nested}
        logged_in = payload.get("logged_in", payload.get("is_logged_in", False))
        return XiaohongshuLoginStatus(logged_in=bool(logged_in), raw=payload)

    def publish_note(
        self,
        *,
        title: str,
        content: str,
        images: list[str],
        tags: list[str],
    ) -> XiaohongshuPublishReceipt:
        payload = self._request(
            "POST",
            "/api/v1/publish",
            json={
                "title": title,
                "content": content,
                "images": images,
                "tags": tags,
            },
        )
        receipt = self._receipt_from_payload(payload)
        if receipt is None:
            raise XiaohongshuDeliveryUnknownError(
                "xiaohongshu_publish_receipt_missing"
            )
        return receipt

    def get_publication_status(self, external_id: str) -> XiaohongshuPublishReceipt:
        if not isinstance(external_id, str) or not external_id.strip():
            raise XiaohongshuApiError("xiaohongshu_publish_id_invalid")
        normalized_id = external_id.strip()
        try:
            payload = self._request(
                "GET",
                f"/api/v1/publish/{quote(normalized_id, safe='')}",
            )
        except XiaohongshuApiError as error:
            # The reference xiaohongshu-mcp service publishes synchronously and
            # does not expose /publish/{id}. Fall back to the account feed so
            # an older in-flight record can still be checked safely.
            if error.http_status != 404:
                raise
            return self._status_from_feed(normalized_id)

        receipt = self._receipt_from_payload(payload)
        if receipt is None:
            raise XiaohongshuApiError("xiaohongshu_status_response_invalid")
        return receipt

    def _status_from_feed(self, external_id: str) -> XiaohongshuPublishReceipt:
        payload = self._request("GET", "/api/v1/feeds/list")
        nested = payload.get("data")
        feeds = nested.get("feeds") if isinstance(nested, dict) else None
        if isinstance(feeds, list):
            for feed in feeds:
                if not isinstance(feed, dict):
                    continue
                feed_id = next(
                    (
                        feed.get(key)
                        for key in ("note_id", "noteId", "post_id", "postId", "id")
                        if isinstance(feed.get(key), (str, int))
                        and str(feed.get(key)).strip()
                    ),
                    None,
                )
                if feed_id is not None and str(feed_id).strip() == external_id:
                    return XiaohongshuPublishReceipt(
                        external_id=external_id,
                        status=PublicationStatus.PUBLISHED,
                    )

        # Absence from the feed is not proof of failure: the feed can lag
        # behind the browser's publish confirmation.
        return XiaohongshuPublishReceipt(
            external_id=external_id,
            status=PublicationStatus.PUBLISHING,
        )

    @staticmethod
    def _receipt_from_payload(payload: dict[str, Any]) -> XiaohongshuPublishReceipt | None:
        # The reference service returns {success: true, data: {note_id: ...}}
        # but different releases may wrap the result more than once. Flatten
        # dictionary envelopes only; never scan arbitrary response text for an
        # ID that could be unrelated to the published note.
        flattened = dict(payload)
        envelope = flattened
        for _ in range(3):
            nested = next(
                (
                    envelope.get(key)
                    for key in ("data", "result", "payload")
                    if isinstance(envelope.get(key), dict)
                ),
                None,
            )
            if nested is None:
                break
            flattened = {**flattened, **nested}
            envelope = nested
        payload = flattened
        external_id = next(
            (
                payload.get(key)
                for key in (
                    "note_id",
                    "noteId",
                    "external_id",
                    "task_id",
                    "publish_id",
                    "post_id",
                    "postId",
                    "id",
                )
                if isinstance(payload.get(key), (str, int))
                and str(payload.get(key)).strip()
            ),
            None,
        )
        if external_id is None:
            return None
        status = str(payload.get("status", payload.get("state", "publishing"))).casefold()
        if status in {
            "published",
            "success",
            "succeeded",
            "completed",
            "done",
            "发布完成",
            "发布成功",
        }:
            mapped_status = PublicationStatus.PUBLISHED
        elif status in {"failed", "error", "rejected", "发布失败"}:
            raise XiaohongshuApiError("xiaohongshu_publish_failed")
        else:
            mapped_status = PublicationStatus.PUBLISHING
        public_url = next(
            (
                payload.get(key).strip()
                for key in ("public_url", "note_url", "url")
                if isinstance(payload.get(key), str) and payload.get(key).strip()
            ),
            None,
        )
        if public_url is None and payload.get("note_id"):
            public_url = (
                "https://www.xiaohongshu.com/explore/"
                f"{str(payload['note_id']).strip()}"
            )
        return XiaohongshuPublishReceipt(
            external_id=str(external_id).strip(),
            status=mapped_status,
            public_url=public_url,
        )


__all__ = [
    "XiaohongshuApiError",
    "XiaohongshuClient",
    "XiaohongshuDeliveryUnknownError",
    "XiaohongshuLoginStatus",
    "XiaohongshuPublishReceipt",
]
