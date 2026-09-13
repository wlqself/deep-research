# Two levels of activity

对话界面应该显示简要进度，例如“已分配给 Researcher 任务：比较两种 retrieval strategy”。日志界面可以显示时间戳、完整任务描述、工具名、elapsed time、source 数量和安全的 failure code。两个界面都应该是同一条 activity event stream 的不同 projection。

# Information boundary

面向用户的日志可以展示经过清理的任务描述和工具结果摘要，但不应暴露 system prompt、API key、原始 provider error、hidden chain-of-thought 或不受限制的完整网页内容。
