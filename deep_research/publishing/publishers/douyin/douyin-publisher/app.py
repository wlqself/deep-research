"""Standalone Douyin image-text publishing service.

The main application talks to this process through the same small HTTP
contract used by the other publishing adapters.  Browser state is kept in a
persistent Chromium profile so login is performed once by the user.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


logger = logging.getLogger("douyin-publisher")
ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    # Load a service-local .env first, then fall back through the repository
    # parents.  This keeps the service usable after being moved under the
    # publisher package while still honoring the project's root .env.
    for candidate in (ROOT, *ROOT.parents):
        env_file = candidate / ".env"
        if env_file.is_file():
            load_dotenv(env_file, override=False)
except ImportError:
    pass


def _project_root() -> Path:
    for candidate in (ROOT, *ROOT.parents):
        if (candidate / ".env").is_file() and (candidate / "data").is_dir():
            return candidate
    return ROOT


DATA_DIR = Path(os.getenv("DOUYIN_DATA_DIR", str(_project_root() / "data" / "douyin-publisher")))
TASKS_FILE = DATA_DIR / "tasks.json"
UPLOAD_URL = "https://creator.douyin.com/creator-micro/content/upload?default-tab=3"


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _profile_path() -> Path:
    configured = os.getenv("DOUYIN_USER_DATA_DIR", "")
    path = Path(configured) if configured else DATA_DIR / "browser-profile"
    if not path.is_absolute():
        path = (_project_root() / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


class PublishRequest(BaseModel):
    mode: str = "image"
    title: str = Field(min_length=1, max_length=100)
    content: str = Field(default="", max_length=20000)
    images: list[str] = Field(min_length=1, max_length=30)
    tags: list[str] = Field(default_factory=list, max_length=30)

    @field_validator("mode")
    @classmethod
    def image_mode_only(cls, value: str) -> str:
        if value != "image":
            raise ValueError("only image mode is supported")
        return value


@dataclass
class Task:
    task_id: str
    payload: dict[str, Any]
    status: str = "queued"
    error_code: str | None = None
    message: str | None = None
    item_id: str | None = None
    url: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "error_code": self.error_code,
            "message": self.message,
            "item_id": self.item_id,
            "url": self.url,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TaskStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[str, Task] = {}
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not TASKS_FILE.exists():
            return
        try:
            raw = json.loads(TASKS_FILE.read_text(encoding="utf-8"))
            for value in raw:
                task = Task(**value)
                self._tasks[task.task_id] = task
        except (OSError, ValueError, TypeError):
            # A corrupt task journal must not prevent the browser service
            # from starting.  New tasks will create a fresh valid journal.
            self._tasks = {}

    def _save(self) -> None:
        temporary = TASKS_FILE.with_suffix(".tmp")
        temporary.write_text(
            json.dumps([task.__dict__ for task in self._tasks.values()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(TASKS_FILE)

    async def put(self, task: Task) -> None:
        async with self._lock:
            self._tasks[task.task_id] = task
            self._save()

    async def update(self, task_id: str, **changes: Any) -> Task:
        async with self._lock:
            task = self._tasks[task_id]
            for key, value in changes.items():
                setattr(task, key, value)
            task.updated_at = time.time()
            self._save()
            return task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)


ProgressCallback = Callable[[str, str], Awaitable[None]]


class DouyinBrowser:
    """One serialized persistent browser session for the single-user setup."""

    def __init__(self) -> None:
        self._playwright: Any = None
        self._context: Any = None
        self._lock = asyncio.Lock()

    @staticmethod
    def runtime_available() -> bool:
        try:
            import playwright.async_api  # noqa: F401

            return True
        except ImportError:
            return False

    def state(self) -> str:
        """Return a cheap diagnostic state without launching a browser."""

        if self._context is None:
            return "stopped"
        try:
            # Accessing pages raises when the underlying context disconnected.
            self._context.pages
        except Exception:
            return "closed"
        return "ready"

    @staticmethod
    def _is_closed_error(error: BaseException) -> bool:
        message = str(error).casefold()
        return any(
            marker in message
            for marker in (
                "target page",
                "browsercontext",
                "browser context",
                "context or browser has been closed",
                "target closed",
                "browser has been closed",
            )
        )

    async def reset(self) -> None:
        """Drop a disconnected browser so the next operation can relaunch it."""

        context, playwright = self._context, self._playwright
        self._context = None
        self._playwright = None
        if context is not None:
            try:
                await context.close()
            except Exception:
                logger.debug("ignoring error while closing stale browser context", exc_info=True)
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                logger.debug("ignoring error while stopping stale Playwright", exc_info=True)

    async def ensure(self) -> Any:
        if self._context is not None:
            return self._context
        if not self.runtime_available():
            raise RuntimeError("playwright_not_installed")

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        executable = os.getenv("DOUYIN_CHROME_PATH", "").strip()
        if not executable:
            for candidate in (
                Path(os.environ.get("PROGRAMFILES", "C:\\Program Files"))
                / "Google/Chrome/Application/chrome.exe",
                Path(os.environ.get("LOCALAPPDATA", ""))
                / "Google/Chrome/Application/chrome.exe",
            ):
                if candidate.is_file():
                    executable = str(candidate)
                    break
        kwargs: dict[str, Any] = {
            "headless": _bool_env("DOUYIN_HEADLESS", False),
            "viewport": {"width": 1440, "height": 1000},
            # The creator center asks for geolocation even though publishing
            # does not use it.  The native permission bubble can sit above the
            # page and make an otherwise-ready publish action look frozen.
            "args": ["--deny-permission-prompts"],
        }
        if executable:
            kwargs["executable_path"] = executable
        try:
            self._context = await self._playwright.chromium.launch_persistent_context(
                str(_profile_path()),
                **kwargs,
            )
        except Exception:
            logger.exception(
                "failed to launch persistent browser",
                extra={"profile_dir": str(_profile_path()), "executable": executable or None},
            )
            await self.reset()
            raise
        logger.info(
            "persistent browser ready",
            extra={"profile_dir": str(_profile_path()), "executable": executable or None},
        )
        return self._context

    async def page(self) -> Any:
        last_error: BaseException | None = None
        for attempt in range(2):
            context = await self.ensure()
            try:
                for page in context.pages:
                    if not page.is_closed():
                        return page
                return await context.new_page()
            except Exception as error:
                last_error = error
                if attempt == 0 and self._is_closed_error(error):
                    logger.warning(
                        "browser context closed; rebuilding it before retry",
                        extra={"error": str(error)},
                    )
                    await self.reset()
                    continue
                raise
        assert last_error is not None
        raise last_error

    async def login_status(self) -> dict[str, Any]:
        async with self._lock:
            page = await self.page()
            await page.goto(UPLOAD_URL, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(1500)
            text = (await page.locator("body").inner_text(timeout=10000)).lower()
            # Do not treat every occurrence of “登录” as logged-out: the
            # creator center may keep a login-related menu item visible even
            # after authentication.  Use the strong login-screen markers.
            login_words = ("扫码登录", "登录抖音", "手机号登录", "验证码登录", "请先登录", "log in", "sign in")
            logged_in = not any(word.lower() in text for word in login_words)
            return {"is_logged_in": logged_in, "username": None}

    async def _find_publish_button(self, page: Any) -> Any | None:
        """Find only the final publish action, not similarly named controls."""

        exact_name = re.compile(r"^(?:发布|发布作品|立即发布|Publish)$", re.I)
        candidates = [
            page.get_by_role("button", name=exact_name),
            page.locator("[data-testid='publish-btn']"),
        ]
        for locator in candidates:
            try:
                count = await locator.count()
            except Exception:
                continue
            for index in range(count - 1, -1, -1):
                candidate = locator.nth(index)
                try:
                    if await candidate.is_visible():
                        return candidate
                except Exception:
                    continue
        return None

    @staticmethod
    async def _button_enabled(button: Any) -> bool:
        try:
            if await button.is_disabled():
                return False
            if (await button.get_attribute("aria-disabled")) == "true":
                return False
            classes = (await button.get_attribute("class")) or ""
            return "disabled" not in classes.casefold()
        except Exception:
            return False

    @staticmethod
    def _manual_markers(body_text: str) -> tuple[str, ...]:
        return (
            "安全验证",
            "人机验证",
            "滑块验证",
            "验证码",
            "手机号验证",
            "请完成验证",
            "请先验证",
        )

    @staticmethod
    def _validation_markers(body_text: str) -> tuple[str, ...]:
        return (
            "发布失败",
            "提交失败",
            "内容违规",
            "标题不能为空",
            "内容不能为空",
            "请修改后再试",
        )

    @staticmethod
    def _checking_markers() -> tuple[str, ...]:
        return (
            "内容检测中",
            "正在检测",
            "检测中",
            "正在校验",
            "校验中",
            "处理中",
            "上传中",
        )

    @staticmethod
    def _success_markers() -> tuple[str, ...]:
        # Douyin changes the wording of the post-submit toast from time to
        # time.  Keep this list deliberately limited to post-submit states;
        # “作品未见异常” only means validation passed and must not be treated
        # as a successful publication.
        return (
            "发布成功",
            "提交成功",
            "作品已发布",
            "审核中",
            "已发布",
        )

    @classmethod
    def _publication_was_acknowledged(
        cls,
        body_text: str,
        *,
        current_url: str = "",
        initial_url: str = "",
    ) -> bool:
        text = body_text.casefold()
        if any(marker.casefold() in text for marker in cls._success_markers()):
            return True

        # Some creator-center versions navigate to content management without
        # showing a toast.  A navigation away from the upload editor is a
        # valid acknowledgement, whereas staying on the editor is not.
        if initial_url and current_url and current_url != initial_url:
            managed_paths = (
                "/creator-micro/content/manage",
                "/creator-micro/content/list",
                "/creator-micro/content/management",
            )
            if any(path in current_url.casefold() for path in managed_paths):
                return True
        return False

    async def _wait_until_publishable(
        self,
        page: Any,
        *,
        timeout_ms: int,
        task_id: str,
        progress: ProgressCallback | None,
    ) -> tuple[Any | None, str | None]:
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            body_text = (await page.locator("body").inner_text(timeout=10000)).casefold()
            if any(marker.casefold() in body_text for marker in self._manual_markers(body_text)):
                await page.screenshot(path=str(DATA_DIR / f"manual-required-{task_id}.png"), full_page=True)
                return None, "manual_required"
            if any(marker.casefold() in body_text for marker in self._validation_markers(body_text)):
                raise ValueError("douyin_validation_failed: platform rejected the content")

            # An enabled-looking button is not enough while the platform is
            # still validating the upload.  Wait for those markers to clear
            # before allowing the final click.
            if any(marker.casefold() in body_text for marker in self._checking_markers()):
                await page.wait_for_timeout(500)
                continue

            button = await self._find_publish_button(page)
            if button is not None and await self._button_enabled(button):
                if progress:
                    await progress("ready_to_publish", "内容检测完成，发布按钮已可用")
                return button, None
            await page.wait_for_timeout(500)

        raise TimeoutError("douyin_publish_button_unavailable: publish button did not become enabled")

    async def _click_confirmation_if_present(self, page: Any, timeout_ms: int) -> bool:
        confirm = page.get_by_role(
            "button",
            name=re.compile(r"^(?:确认发布|确认提交|继续发布|确认)$", re.I),
        )
        deadline = time.monotonic() + min(timeout_ms, 5000) / 1000
        while time.monotonic() < deadline:
            try:
                count = await confirm.count()
                for index in range(count - 1, -1, -1):
                    candidate = confirm.nth(index)
                    if await candidate.is_visible() and await self._button_enabled(candidate):
                        await candidate.click()
                        return True
            except Exception:
                pass
            await page.wait_for_timeout(250)
        return False

    async def publish(
        self,
        task: Task,
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        payload = task.payload
        images = payload["images"]
        for image in images:
            if not Path(image).is_file():
                raise ValueError(f"image_not_found: {image}")

        async with self._lock:
            page = await self.page()
            timeout_ms = _int_env("DOUYIN_PUBLISH_TIMEOUT_SECONDS", 180) * 1000
            page.set_default_timeout(timeout_ms)
            if progress:
                await progress("uploading", "正在打开抖音创作者中心并上传图片")
            await page.goto(UPLOAD_URL, wait_until="domcontentloaded", timeout=timeout_ms)
            await page.wait_for_timeout(1500)

            file_input = page.locator("input[type='file']").first
            await file_input.wait_for(state="attached", timeout=timeout_ms)
            await file_input.set_input_files(images)
            await page.wait_for_timeout(1000)

            if progress:
                await progress("filling", "图片上传完成，正在填写标题和正文")

            title = page.locator("input[placeholder*='标题']").first
            if await title.count() == 0:
                title = page.locator("input").first
            await title.fill(payload["title"])

            body = page.locator("[contenteditable='true']").first
            if await body.count() == 0:
                body = page.locator("textarea").first
            text = _with_hashtags(payload["content"], payload.get("tags", []))
            await body.fill(text)

            if _bool_env("DOUYIN_DRY_RUN", False):
                await page.screenshot(path=str(DATA_DIR / f"dry-run-{task.task_id}.png"), full_page=True)
                return {
                    "status": "manual_required",
                    "message": "dry-run completed; final publish button was not clicked",
                }

            if progress:
                await progress("checking", "等待抖音完成内容检测")
            button, manual_status = await self._wait_until_publishable(
                page,
                timeout_ms=timeout_ms,
                task_id=task.task_id,
                progress=progress,
            )
            if manual_status == "manual_required":
                return {
                    "status": "manual_required",
                    "message": "抖音要求完成安全验证，请在浏览器中处理后重试",
                }
            assert button is not None

            # Re-resolve the final button immediately before the action.  The
            # creator center re-renders this area after validation and an old
            # locator can otherwise point at a detached/disabled element.
            button = await self._find_publish_button(page)
            if button is None or not await self._button_enabled(button):
                raise TimeoutError(
                    "douyin_publish_button_unavailable: final publish button disappeared"
                )
            initial_url = page.url
            await button.scroll_into_view_if_needed(timeout=timeout_ms)
            await page.screenshot(path=str(DATA_DIR / f"before-publish-{task.task_id}.png"), full_page=True)
            if progress:
                await progress("submitting", "内容检测完成，正在自动点击最终发布按钮")
            await button.click(timeout=timeout_ms)
            logger.info(
                "final Douyin publish button clicked automatically",
                extra={"task_id": task.task_id},
            )
            if progress:
                await progress("clicked", "最终发布按钮已自动点击，正在等待抖音确认")
            await self._click_confirmation_if_present(page, timeout_ms)

            deadline = time.monotonic() + timeout_ms / 1000
            while time.monotonic() < deadline:
                body_text = (await page.locator("body").inner_text(timeout=10000)).casefold()
                if any(marker.casefold() in body_text for marker in self._manual_markers(body_text)):
                    await page.screenshot(path=str(DATA_DIR / f"manual-required-{task.task_id}.png"), full_page=True)
                    return {
                        "status": "manual_required",
                        "message": "抖音要求完成安全验证，请在浏览器中处理后重试",
                    }
                if any(marker.casefold() in body_text for marker in self._validation_markers(body_text)):
                    raise ValueError("douyin_validation_failed: platform rejected the content")

                if self._publication_was_acknowledged(
                    body_text,
                    current_url=page.url,
                    initial_url=initial_url,
                ):
                    await page.screenshot(path=str(DATA_DIR / f"after-publish-{task.task_id}.png"), full_page=True)
                    return {
                        "status": "published",
                        "message": "publish submitted and platform acknowledgement verified",
                    }
                await page.wait_for_timeout(500)

            await page.screenshot(path=str(DATA_DIR / f"after-publish-{task.task_id}.png"), full_page=True)
            logger.warning(
                "Douyin publish click completed but no platform acknowledgement was observed",
                extra={"task_id": task.task_id, "url": page.url},
            )
            return {
                "status": "publishing",
                "message": "final publish button clicked automatically; platform receipt pending",
            }


def _with_hashtags(content: str, tags: list[str]) -> str:
    result = content.strip()
    existing = {tag.lower().lstrip("#") for tag in re.findall(r"#[^\s#]+", result)}
    suffix = [f"#{tag.lstrip('#')}" for tag in tags if tag.strip() and tag.lower().lstrip("#") not in existing]
    return (result + "\n\n" if result and suffix else result) + " ".join(suffix)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    await browser.reset()


app = FastAPI(title="Douyin Publisher", version="0.1.0", lifespan=lifespan)
store = TaskStore()
browser = DouyinBrowser()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "douyin-publisher",
        "browser_runtime_available": browser.runtime_available(),
        "browser_context": browser.state(),
        "profile_dir": str(_profile_path()),
        "headless": _bool_env("DOUYIN_HEADLESS", False),
    }


@app.get("/api/v1/login/status")
async def login_status() -> dict[str, Any]:
    try:
        data = await browser.login_status()
    except Exception as exc:
        logger.exception("login status check failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"success": True, "data": data, "message": "检查登录状态成功"}


async def _run_task(task: Task) -> None:
    try:
        await store.update(task.task_id, status="running")

        async def progress(status: str, message: str) -> None:
            await store.update(task.task_id, status=status, message=message)

        result = await browser.publish(task, progress=progress)
        status = result.get("status", "publishing")
        await store.update(
            task.task_id,
            status=status,
            item_id=task.task_id,
            message=result.get("message"),
        )
    except asyncio.CancelledError:
        raise
    except ValueError as exc:
        await store.update(task.task_id, status="failed", error_code="invalid_request", message=str(exc))
    except Exception as exc:
        error = str(exc)
        logger.exception("Douyin publish task failed", extra={"task_id": task.task_id})
        code = (
            "douyin_publish_timeout"
            if isinstance(exc, TimeoutError)
            else
            "douyin_browser_unavailable"
            if DouyinBrowser._is_closed_error(exc)
            else "playwright_failed"
            if "playwright" in error.lower()
            else "douyin_publish_failed"
        )
        await store.update(task.task_id, status="failed", error_code=code, message=error)


@app.post("/api/v1/publish", status_code=202)
async def publish(request: PublishRequest) -> dict[str, Any]:
    if len(request.images) > _int_env("DOUYIN_MAX_IMAGES", 30):
        raise HTTPException(status_code=400, detail="too_many_images")
    task_id = str(uuid.uuid4())
    task = Task(task_id=task_id, payload=request.model_dump())
    await store.put(task)
    asyncio.create_task(_run_task(task), name=f"douyin-publish-{task_id}")
    return {"success": True, "task_id": task_id, "status": "queued"}


@app.get("/api/v1/publish/{task_id}")
async def publish_status(task_id: str) -> dict[str, Any]:
    task = store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task_not_found")
    return {"success": True, **task.as_dict()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("DOUYIN_PUBLISHER_HOST", "127.0.0.1"),
        port=_int_env("DOUYIN_PUBLISHER_PORT", 18070),
    )
