from pydantic import BaseModel

# 知识库统计响应模型
class KnowledgeStatsResponse(BaseModel):
    document_count: int # 知识库中的文档总数
    chunk_count: int # 知识库中的文本块总数
    indexed_count: int # 已完成索引的文档数量
    processing_count: int # 正在处理中的文档数量
    failed_count: int # 处理失败的文档数量
    pending_count: int # 等待处理的文档数量

# 单个文档详情响应模型
class DocumentResponse(BaseModel):
    document_id: str # 文档的唯一标识
    filename: str # 原始文件名
    mime_type: str # MIME 类型,前端据此显示对应图标或决定预览方式
    size_bytes: int # 文件大小
    status: str # 文档当前状态
    page_count: int # 文档的页数
    chunk_count: int # 该文档被切分出的 chunk 数量
    error: str # 存放错误信息
    created_at: str # 文档创建时间
    updated_at: str # 文档最后更新时间