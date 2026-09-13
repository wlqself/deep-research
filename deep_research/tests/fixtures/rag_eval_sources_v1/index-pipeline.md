# Parsing and chunking

入库流程支持 PDF、Markdown 和纯文本文件。PDF 按页解析，Markdown 先按标题拆分再继续切 chunk，纯文本作为一个源文档处理。每个 chunk 都保留 document_id、filename、page_number、section_title、chunk_index 和 content_hash 元数据。

# Index lifecycle

文档会经历 pending、processing 和 indexed 状态。重新索引时，系统先让旧 chunk 不可检索，再删除旧的 vector points，创建新的 chunk 和 embedding，最后才把文档标记为 indexed。chunk ID 由 document 和 chunk 位置确定，因此重建时可以替换同一批逻辑数据。
