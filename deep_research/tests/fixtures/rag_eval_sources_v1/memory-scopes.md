# Checkpoint and store scope

checkpointer 持久化 thread 范围内的图状态，例如 messages、tasks 和短期对话进度。store 用于跨 thread recall 的数据，例如长期记忆。两者的选择取决于生命周期和作用域，而不只是数据是否存储在 SQLite 中。

# Recall boundary

长期记忆 recall 应该只返回数量受控的一小组相关条目。系统不应把完整对话或原始网页全文复制进长期记忆。Main Agent 控制 memory write 和 recall，Researcher 不应直接访问 long-term memory store。
