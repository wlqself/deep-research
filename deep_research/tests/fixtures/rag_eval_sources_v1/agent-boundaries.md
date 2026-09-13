# Delegation and local knowledge

Main Agent 负责监督对话，并把多步骤研究委派给 Researcher 子 agent。当问题可能由已上传材料回答时，Researcher 应先查询 local knowledge base，再使用 web search。最终面向用户的回答仍由 Main Agent 负责。

# Evidence bookkeeping

知识库检索结果会注册为 S1、S2 这样的真实 source ID。Finding 保存重要且有依据的结论，并引用对应 source ID。检索 similarity score 只表示排序相似度，不是经过校准的事实可信度。
