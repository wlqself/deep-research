import unittest
from datetime import datetime, timedelta, timezone

import httpx

from deep_research.publishing.publishers.wechat_api import (
    WeChatAccessTokenProvider,
    WeChatApiError,
    WeChatCredentials,
    WeChatOfficialAccountClient,
    WeChatTokenError,
)


class WeChatAccessTokenProviderTests(unittest.TestCase):
    def test_credentials_do_not_expose_app_secret_in_repr(self):
        credentials = WeChatCredentials("wx-test", "secret-value")

        self.assertNotIn("secret-value", repr(credentials))

    def test_token_is_cached_until_refresh_window(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "access_token": "token-1",
                    "expires_in": 7200,
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = WeChatAccessTokenProvider(
            WeChatCredentials("wx-test", "secret-value"),
            client=client,
        )
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)

        try:
            self.assertEqual(provider.get_access_token(now=now), "token-1")
            self.assertEqual(
                provider.get_access_token(
                    now=now + timedelta(seconds=6000),
                ),
                "token-1",
            )
        finally:
            client.close()

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].method, "POST")
        self.assertEqual(
            requests[0].url.path,
            "/cgi-bin/stable_token",
        )

    def test_token_refreshes_inside_safety_window(self):
        responses = iter(
            [
                {"access_token": "token-1", "expires_in": 7200},
                {"access_token": "token-2", "expires_in": 7200},
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=next(responses))

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = WeChatAccessTokenProvider(
            WeChatCredentials("wx-test", "secret-value"),
            client=client,
        )
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)

        try:
            self.assertEqual(provider.get_access_token(now=now), "token-1")
            self.assertEqual(
                provider.get_access_token(
                    now=now + timedelta(seconds=6900),
                ),
                "token-2",
            )
        finally:
            client.close()

    def test_rejected_response_returns_safe_error_code(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"errcode": 40001, "errmsg": "secret leaked here"},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = WeChatAccessTokenProvider(
            WeChatCredentials("wx-test", "secret-value"),
            client=client,
        )

        try:
            with self.assertRaises(WeChatTokenError) as context:
                provider.get_access_token()
        finally:
            client.close()

        self.assertEqual(
            context.exception.error_code,
            "wechat_token_rejected",
        )
        self.assertEqual(context.exception.provider_code, 40001)
        self.assertEqual(
            context.exception.provider_message,
            "secret leaked here",
        )
        self.assertNotIn("secret leaked here", str(context.exception))

    def test_incomplete_response_is_rejected(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"expires_in": 7200})

        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = WeChatAccessTokenProvider(
            WeChatCredentials("wx-test", "secret-value"),
            client=client,
        )

        try:
            with self.assertRaises(WeChatTokenError) as context:
                provider.get_access_token()
        finally:
            client.close()

        self.assertEqual(
            context.exception.error_code,
            "wechat_token_response_invalid",
        )


class WeChatOfficialAccountClientTests(unittest.TestCase):
    def _build_client(self, handler):
        requests: list[httpx.Request] = []

        def recording_handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return handler(request)

        client = httpx.Client(
            transport=httpx.MockTransport(recording_handler),
        )
        provider = WeChatAccessTokenProvider(
            WeChatCredentials("wx-test", "secret-value"),
            client=client,
        )
        api = WeChatOfficialAccountClient(provider, client=client)
        return api, client, requests

    @staticmethod
    def _response_for(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/stable_token"):
            return httpx.Response(
                200,
                json={
                    "access_token": "token-1",
                    "expires_in": 7200,
                },
            )
        return httpx.Response(200, json={})

    def test_add_draft_returns_media_id(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/stable_token"):
                return self._response_for(request)
            return httpx.Response(200, json={"media_id": "draft-1"})

        api, client, requests = self._build_client(handler)
        try:
            self.assertEqual(
                api.add_draft({"articles": [{"title": "Test"}]}),
                "draft-1",
            )
        finally:
            client.close()

        self.assertEqual(len(requests), 2)
        self.assertEqual(
            requests[1].url.path,
            "/cgi-bin/draft/add",
        )
        self.assertEqual(
            requests[1].url.params["access_token"],
            "token-1",
        )

    def test_uploads_cover_and_inline_image_without_file_paths(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/stable_token"):
                return self._response_for(request)
            if request.url.path.endswith("/add_material"):
                return httpx.Response(200, json={"media_id": "cover-1"})
            return httpx.Response(
                200,
                json={"url": "https://mmbiz.qpic.cn/body.jpg"},
            )

        api, client, requests = self._build_client(handler)
        try:
            self.assertEqual(
                api.upload_cover(b"cover", filename="cover.jpg"),
                "cover-1",
            )
            self.assertEqual(
                api.upload_inline_image(b"body", filename="body.jpg"),
                "https://mmbiz.qpic.cn/body.jpg",
            )
        finally:
            client.close()

        self.assertEqual(
            requests[1].url.params["type"],
            "thumb",
        )
        self.assertEqual(
            requests[2].url.path,
            "/cgi-bin/media/uploadimg",
        )

    def test_submit_and_status_are_structured(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/stable_token"):
                return self._response_for(request)
            if request.url.path.endswith("/submit"):
                return httpx.Response(200, json={"publish_id": "publish-1"})
            return httpx.Response(
                200,
                json={
                    "publish_status": 0,
                    "article_id": "article-1",
                    "article_detail": {
                        "item": [
                            {"article_url": "https://mp.weixin.qq.com/s/test"}
                        ]
                    },
                },
            )

        api, client, requests = self._build_client(handler)
        try:
            self.assertEqual(api.submit_draft("draft-1"), "publish-1")
            status = api.get_publish_status("publish-1")
        finally:
            client.close()

        self.assertEqual(status.publish_status, 0)
        self.assertEqual(status.article_id, "article-1")
        self.assertEqual(
            status.article_url,
            "https://mp.weixin.qq.com/s/test",
        )
        self.assertEqual(requests[1].url.path, "/cgi-bin/freepublish/submit")
        self.assertEqual(requests[2].url.path, "/cgi-bin/freepublish/get")

    def test_unsafe_cover_inputs_use_stable_error_codes(self):
        api, client, _ = self._build_client(self._response_for)
        try:
            with self.assertRaises(WeChatApiError) as context:
                api.upload_cover(b"cover", filename="..\\cover.jpg")
        finally:
            client.close()

        self.assertEqual(
            context.exception.error_code,
            "wechat_cover_filename_invalid",
        )

    def test_cover_rejection_keeps_provider_diagnostic(self):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/stable_token"):
                return self._response_for(request)
            return httpx.Response(
                200,
                json={
                    "errcode": 40164,
                    "errmsg": "invalid ip not in whitelist",
                },
            )

        api, client, _ = self._build_client(handler)
        try:
            with self.assertRaises(WeChatApiError) as context:
                api.upload_cover(b"cover", filename="cover.jpg")
        finally:
            client.close()

        self.assertEqual(
            context.exception.error_code,
            "wechat_cover_rejected",
        )
        self.assertEqual(context.exception.provider_code, 40164)
        self.assertEqual(
            context.exception.provider_message,
            "invalid ip not in whitelist",
        )


if __name__ == "__main__":
    unittest.main()
