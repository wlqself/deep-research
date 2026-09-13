# 小红书发布适配器

本目录只负责把项目里的发布请求转换成小红书发布服务请求；浏览器、登录态和小红书页面操作由单独运行的本地服务负责。

推荐使用成熟的 [`YuriGao/xiaohongshu-mcp`](https://github.com/YuriGao/xiaohongshu-mcp)：

1. 按该项目的 Windows 指南下载 Release 或从源码编译。
2. 先启动登录程序，使用小红书 App 扫码登录。
3. 启动服务，默认监听 `http://127.0.0.1:18060`。
4. 确认以下请求可以返回成功：

   ```text
   GET http://127.0.0.1:18060/health
   GET http://127.0.0.1:18060/api/v1/login/status
   ```

5. 在主项目 `.env` 中配置：

   ```ini
   XIAOHONGSHU_BASE_URL=http://127.0.0.1:18060
   XIAOHONGSHU_TIMEOUT_SECONDS=120
   XIAOHONGSHU_MAX_IMAGES=9
   XIAOHONGSHU_TITLE_MAX_LENGTH=20
   ```

## Windows 本地目录启动

迁移到本目录后，必须从 exe 所在目录启动，这样程序才能按相对路径读取同目录的
`cookies.json`：

```powershell
$serviceDir = "E:\my_agent\old coding\deep_research\publishing\publishers\xiaohongshu\xiaohongshu-mcp"
Set-Location $serviceDir

# 仅首次登录或登录态失效时执行
.\xiaohongshu-login-windows-amd64.exe

# 登录完成后启动 HTTP 服务；此窗口需要保持运行
.\xiaohongshu-mcp-windows-amd64.exe -headless=false -port ":18060"
```

不要启动带有 `backup` 或 `before-pr` 名称的备份程序。若出现 `18060` 端口占用，说明
已有小红书 MCP 在运行，不需要再次启动。

## 兼容的发布请求

适配器调用：

```text
POST /api/v1/publish
Content-Type: application/json
```

请求体为：

```json
{
  "title": "标题",
  "content": "正文",
  "images": ["C:/absolute/path/image.jpg"],
  "tags": ["标签"]
}
```

第三方服务的成功响应通常是：

```json
{
  "success": true,
  "data": {
    "status": "published",
    "note_id": "笔记ID"
  }
}
```

适配器已经兼容：

- `data.is_logged_in` 登录状态字段；
- `data.note_id` 发布编号字段（优先使用）；同时兼容 `post_id`、`external_id` 等旧字段；
- `published`、`发布成功`、`发布完成` 等成功状态；
- `error` + `code` 错误响应；
- 第三方服务没有 `/api/v1/publish/{id}` 时，使用 `/api/v1/feeds/list` 按 `note_id` 兜底核验。

图片路径必须是主项目本机可访问的绝对路径。主项目会根据发布中心选中的 `attachment_ids` 解析图片，不区分图片来自哪个对话。

## 重要边界

小红书服务是外部进程，不会随主项目自动启动。主项目启动时只会根据 `XIAOHONGSHU_BASE_URL` 注册客户端；没有配置时，小红书发布功能保持禁用。

发布接口是浏览器自动化操作，服务返回成功后主项目会记录 `note_id`。如果浏览器操作超时或服务返回成功但没有编号，主项目会将结果标记为需要人工确认，避免自动重复发布。
