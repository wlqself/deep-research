import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from deep_research import app_lifecycle
from deep_research.publishing.models import PublicationChannel


class _Publisher:
    def get_publication_status(self, external_id):
        return None


class _Service:
    def __init__(self):
        self.publications = {
            PublicationChannel.WECHAT_OFFICIAL_ACCOUNT: [
                SimpleNamespace(
                    publication_id="wechat-publication",
                    article_id="wechat-article",
                    attempt_count=0,
                )
            ],
            PublicationChannel.XIAOHONGSHU: [
                SimpleNamespace(
                    publication_id="xiaohongshu-publication",
                    article_id="xiaohongshu-article",
                    attempt_count=0,
                )
            ],
            PublicationChannel.DOUYIN: [
                SimpleNamespace(
                    publication_id="douyin-publication",
                    article_id="douyin-article",
                    attempt_count=0,
                )
            ],
        }
        self.calls = []

    def list_recoverable_publications(self, *, channel):
        return self.publications.get(channel, [])

    def resume_publication_for_article(self, article_id, *, channel, publisher):
        self.calls.append((article_id, channel, publisher))


class PublicationRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_recovery_loop_polls_all_configured_channels(self):
        service = _Service()
        publishers = {
            PublicationChannel.WECHAT_OFFICIAL_ACCOUNT: _Publisher(),
            PublicationChannel.XIAOHONGSHU: _Publisher(),
            PublicationChannel.DOUYIN: _Publisher(),
        }

        async def stop_after_one_pass(_interval):
            raise asyncio.CancelledError

        with patch.object(
            app_lifecycle.asyncio,
            "sleep",
            new=AsyncMock(side_effect=stop_after_one_pass),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await app_lifecycle._publication_recovery_loop(
                    service,
                    publishers,
                    interval_seconds=0,
                    max_attempts=3,
                )

        self.assertEqual(
            {call[1] for call in service.calls},
            {
                PublicationChannel.WECHAT_OFFICIAL_ACCOUNT,
                PublicationChannel.XIAOHONGSHU,
                PublicationChannel.DOUYIN,
            },
        )
        self.assertEqual(len(service.calls), 3)

    async def test_recovery_loop_skips_attempt_limit_and_missing_status_reader(self):
        service = _Service()
        service.publications[PublicationChannel.DOUYIN][0].attempt_count = 3
        publishers = {
            PublicationChannel.WECHAT_OFFICIAL_ACCOUNT: object(),
            PublicationChannel.XIAOHONGSHU: _Publisher(),
            PublicationChannel.DOUYIN: _Publisher(),
        }

        async def stop_after_one_pass(_interval):
            raise asyncio.CancelledError

        with patch.object(
            app_lifecycle.asyncio,
            "sleep",
            new=AsyncMock(side_effect=stop_after_one_pass),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await app_lifecycle._publication_recovery_loop(
                    service,
                    publishers,
                    interval_seconds=0,
                    max_attempts=3,
                )

        self.assertEqual(
            [call[1] for call in service.calls],
            [PublicationChannel.XIAOHONGSHU],
        )


if __name__ == "__main__":
    unittest.main()
