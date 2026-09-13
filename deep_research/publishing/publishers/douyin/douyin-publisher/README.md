# Douyin Publisher

这是独立的抖音图文浏览器发布服务。主项目不直接依赖 Playwright，而是通过
`http://127.0.0.1:18070` 调用它。

## 安装

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\deep_research\publishing\publishers\douyin\douyin-publisher\requirements.txt
```

如果使用本机 Chrome，可在 `.env` 中设置：

```ini
DOUYIN_HEADLESS=false
DOUYIN_CHROME_PATH=C:\Program Files\Google\Chrome\Application\chrome.exe
DOUYIN_USER_DATA_DIR=data\douyin-publisher\browser-profile
DOUYIN_DRY_RUN=false
```

也可以让 Playwright 使用自己的 Chromium：

```powershell
.\.venv\Scripts\python.exe -m playwright install chromium
```

## 启动

```powershell
.\.venv\Scripts\python.exe .\deep_research\publishing\publishers\douyin\douyin-publisher\app.py
```

第一次调用登录状态时会打开持久化浏览器配置目录。用户完成抖音创作者中心登录
后，后续发布复用同一个登录态。服务只串行执行一个浏览器发布任务，适合当前单用户
单浏览器场景。

## HTTP 契约

```text
GET  /health
GET  /api/v1/login/status
POST /api/v1/publish
GET  /api/v1/publish/{task_id}
```

`POST /api/v1/publish` 接收 `mode= image`、`title`、`content`、本地绝对路径
`images` 和 `tags`，先返回 `task_id`，再由状态接口返回 `queued/running/publishing/
published/failed`。完整流程状态为：

`queued -> running -> uploading -> filling -> checking -> ready_to_publish -> submitting -> published`

也可能以 `failed` 或 `manual_required` 结束。初始 `202 Accepted` 只代表浏览器任务
已排队，不代表已经发布；主项目会继续轮询 `/api/v1/publish/{task_id}`。

发布前会等待内容检测标记消失，并使用精确的语义按钮选择器，避免误点“高清发布”等
相似按钮。服务会把发布前、发布后或人工验证所需的截图保存到数据目录，便于排查。

建议第一次验证时使用 `DOUYIN_DRY_RUN=true`：它会登录、打开图文发布页、上传图片、
填写标题正文并保存截图，但不会点击最终发布按钮。确认页面选择器稳定后，再改为
`false` 做真实发布。
