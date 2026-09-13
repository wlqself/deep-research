# Douyin browser service contract

The main application does not own a Chrome process. Configure
`DOUYIN_BASE_URL` to point to a local Playwright/CDP service that owns the
persistent Douyin login profile.

## Endpoints

`GET /health`

`GET /api/v1/login/status`

`POST /api/v1/publish`

```json
{
  "mode": "image",
  "title": "作品标题",
  "content": "作品正文",
  "images": ["C:/absolute/path/01.jpg"],
  "tags": ["AI"]
}
```

The publish endpoint must return a JSON object containing an external task ID
(`external_id`, `task_id`, `publish_id`, `item_id`, or `id`) and a status such
as `publishing` or `published`. `GET /api/v1/publish/{id}` must return the
same shape. A missing receipt is treated as `delivery_unknown` so the caller
does not blindly publish again.
