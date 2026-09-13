"""HTTP client for a local Douyin browser-automation service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
import time

from ...models import PublicationStatus


class DouyinApiError(RuntimeError):
    """A stable error returned by the local Douyin service."""

    def __init__(self, error_code: str, message: str = "Douyin request failed") -> None:
        self.error_code = error_code
        super().__init__(message)


class DouyinDeliveryUnknownError(DouyinApiError):
    """The publish request may have reached the browser service without a receipt."""

    def __init__(self, error_code: str, *, external_id: str | None = None) -> None:
        self.external_id = external_id
        super().__init__(error_code)


@dataclass(frozen=True)
class DouyinPublishReceipt:
    external_id: str
    status: PublicationStatus
    public_url: str | None = None


@dataclass(frozen=True)
class DouyinLoginStatus:
    logged_in: bool
    raw: dict[str, Any]


class DouyinClient:
    """Call a loopback REST service that owns Playwright/CDP and cookies.

    The service contract is intentionally small:
    ``POST /api/v1/publish`` accepts an image-text post and returns a receipt;
    ``GET /api/v1/publish/{id}`` returns its current state.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 1.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        normalized = base_url.strip().rstrip("/") if isinstance(base_url, str) else ""
        if not normalized:
            raise ValueError("base_url must not be empty")
        self.base_url = normalized
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = max(0.0, poll_interval_seconds)
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
            raise DouyinApiError(
                "douyin_http_error",
                "Douyin service returned an HTTP error",
            ) from error
        except (httpx.TransportError, ValueError, TypeError) as error:
            raise DouyinDeliveryUnknownError(
                "douyin_service_unreachable"
            ) from error
        if not isinstance(payload, dict):
            raise DouyinApiError(
                "douyin_response_invalid",
                "Douyin service returned an invalid response",
            )
        if payload.get("success") is False or payload.get("ok") is False:
            raise DouyinApiError(
                str(payload.get("error_code") or "douyin_rejected"),
                "Douyin service rejected the request",
            )
        return payload

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def get_login_status(self) -> DouyinLoginStatus:
        payload = self._request("GET", "/api/v1/login/status")
        logged_in = payload.get("logged_in", payload.get("is_logged_in", False))
        return DouyinLoginStatus(logged_in=bool(logged_in), raw=payload)

    def publish_image_note(
        self,
        *,
        title: str,
        content: str,
        images: list[str],
        tags: list[str],
    ) -> DouyinPublishReceipt:
        payload = self._request(
            "POST",
            "/api/v1/publish",
            json={
                "mode": "image",
                "title": title,
                "content": content,
                "images": images,
                "tags": tags,
            },
        )
        receipt = self._receipt_from_payload(payload)
        if receipt is None:
            raise DouyinDeliveryUnknownError("douyin_publish_receipt_missing")
        if receipt.status is PublicationStatus.PUBLISHED:
            return receipt

        deadline = time.monotonic() + self.timeout_seconds
        latest = receipt
        while time.monotonic() < deadline:
            if self.poll_interval_seconds:
                time.sleep(self.poll_interval_seconds)
            latest = self.get_publication_status(latest.external_id)
            if latest.status is PublicationStatus.PUBLISHED:
                return latest

        raise DouyinDeliveryUnknownError(
            "douyin_publish_timeout",
            external_id=latest.external_id,
        )

    def get_publication_status(self, external_id: str) -> DouyinPublishReceipt:
        if not isinstance(external_id, str) or not external_id.strip():
            raise DouyinApiError("douyin_publish_id_invalid")
        stable_external_id = external_id.strip()
        payload = self._request(
            "GET",
            f"/api/v1/publish/{quote(stable_external_id, safe='')}",
        )
        receipt = self._receipt_from_payload(payload)
        if receipt is None:
            raise DouyinApiError("douyin_status_response_invalid")
        # The browser service may return its final item_id after the task has
        # completed.  Keep the task id as the stable polling key so refreshes
        # continue to address the same service task.
        return DouyinPublishReceipt(
            external_id=stable_external_id,
            status=receipt.status,
            public_url=receipt.public_url,
        )

    @staticmethod
    def _receipt_from_payload(
        payload: dict[str, Any],
    ) -> DouyinPublishReceipt | None:
        nested = payload.get("data")
        if isinstance(nested, dict):
            payload = {**payload, **nested}
        external_id = next(
            (
                payload.get(key)
                for key in ("external_id", "task_id", "publish_id", "item_id", "id")
                if isinstance(payload.get(key), (str, int))
                and str(payload.get(key)).strip()
            ),
            None,
        )
        if external_id is None:
            return None
        status = str(payload.get("status", payload.get("state", "publishing"))).casefold()
        if status in {"published", "success", "succeeded", "completed", "done"}:
            mapped_status = PublicationStatus.PUBLISHED
        elif status in {"failed", "error", "rejected"}:
            raise DouyinApiError(
                str(payload.get("error_code") or "douyin_publish_failed"),
                str(payload.get("message") or "Douyin publish task failed"),
            )
        elif status in {"manual_required", "awaiting_manual_publish", "needs_verification"}:
            raise DouyinApiError(
                "douyin_manual_required",
                str(payload.get("message") or "Douyin requires manual verification"),
            )
        else:
            mapped_status = PublicationStatus.PUBLISHING
        public_url = next(
            (
                payload.get(key).strip()
                for key in ("public_url", "item_url", "url")
                if isinstance(payload.get(key), str) and payload.get(key).strip()
            ),
            None,
        )
        return DouyinPublishReceipt(
            external_id=str(external_id).strip(),
            status=mapped_status,
            public_url=public_url,
        )


__all__ = [
    "DouyinApiError",
    "DouyinClient",
    "DouyinDeliveryUnknownError",
    "DouyinLoginStatus",
    "DouyinPublishReceipt",
]
