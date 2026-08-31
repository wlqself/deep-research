import sqlite3
from pathlib import Path
from typing import Literal, TypedDict

# 限定文档状态，包括待处理、处理中、已索引、失败、删除中、已删除
DocumentStatus = Literal[
    "pending",
    "processing",
    "indexed",
    "failed",
    "deleting",
    "deleted",
]

class DocumentRecord(TypedDict):
    document_id: str # 服务端生成的 UUID
    collection_id: str # Qdrant Collection 名称
    filename: str  # 用户展示名称
    safe_filename: str # 服务端安全文件名
    mime_type: str
    size_bytes: int
    sha256: str # 文件去重
    status: DocumentStatus # 文档生命周期状态
    page_count: int # 解析页数
    chunk_count: int # 写入 Chunk 数量
    error: str # 安全错误码，不保存原始异常
    created_at: str
    updated_at: str


class DocumentRegistry:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)

    # 创建数据库目录、documents 表，以及按状态查询的索引。
    def initialize(self) -> None:
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        connection = sqlite3.connect(
        str(self.db_path)
        )       
        try:
            connection.executescript(
                                """
                                CREATE TABLE IF NOT EXISTS documents (
                                    document_id TEXT PRIMARY KEY,
                                    collection_id TEXT NOT NULL,
                                    filename TEXT NOT NULL,
                                    safe_filename TEXT NOT NULL,
                                    mime_type TEXT NOT NULL,
                                    size_bytes INTEGER NOT NULL,
                                    sha256 TEXT NOT NULL UNIQUE,
                                    status TEXT NOT NULL,
                                    page_count INTEGER NOT NULL DEFAULT 0,
                                    chunk_count INTEGER NOT NULL DEFAULT 0,
                                    error TEXT NOT NULL DEFAULT '',
                                    created_at TEXT NOT NULL,
                                    updated_at TEXT NOT NULL,
                                    CHECK (
                                        status IN (
                                            'pending',
                                            'processing',
                                            'indexed',
                                            'failed',
                                            'deleting',
                                            'deleted'
                                        )
                                    )
                                );

                                CREATE INDEX IF NOT EXISTS
                                idx_documents_status
                                ON documents(status);

                                CREATE TABLE IF NOT EXISTS document_audit (
                                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    document_id TEXT NOT NULL,
                                    collection_id TEXT NOT NULL,
                                    filename TEXT NOT NULL,
                                    safe_filename TEXT NOT NULL,
                                    mime_type TEXT NOT NULL,
                                    size_bytes INTEGER NOT NULL,
                                    sha256 TEXT NOT NULL,
                                    event_type TEXT NOT NULL,
                                    status_before TEXT NOT NULL,
                                    status_after TEXT NOT NULL,
                                    error TEXT NOT NULL DEFAULT '',
                                    created_at TEXT NOT NULL
                                );

                                CREATE INDEX IF NOT EXISTS
                                idx_document_audit_document_id
                                ON document_audit(document_id);

                                CREATE INDEX IF NOT EXISTS
                                idx_document_audit_sha256
                                ON document_audit(sha256);

                                INSERT INTO document_audit (
                                    document_id,
                                    collection_id,
                                    filename,
                                    safe_filename,
                                    mime_type,
                                    size_bytes,
                                    sha256,
                                    event_type,
                                    status_before,
                                    status_after,
                                    error,
                                    created_at
                                )
                                SELECT
                                    d.document_id,
                                    d.collection_id,
                                    d.filename,
                                    d.safe_filename,
                                    d.mime_type,
                                    d.size_bytes,
                                    d.sha256,
                                    'legacy_deleted_cleanup',
                                    d.status,
                                    'deleted',
                                    d.error,
                                    d.updated_at
                                FROM documents AS d
                                WHERE d.status = 'deleted'
                                  AND NOT EXISTS (
                                      SELECT 1
                                      FROM document_audit AS a
                                      WHERE a.document_id = d.document_id
                                        AND a.event_type =
                                            'legacy_deleted_cleanup'
                                  );

                                DELETE FROM documents
                                WHERE status = 'deleted';
                                """
                            )


            connection.commit() # commit -> 提交事务
        finally:
            connection.close() # close  -> 释放文件锁

    # 插入一条记录。成功后提交，失败则回滚；sha256 唯一，可用于文件去重。
    def create_document(
        self,
        record: DocumentRecord,
    ) -> None:
        connection = sqlite3.connect(
            str(self.db_path)
        )

        try:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    collection_id,
                    filename,
                    safe_filename,
                    mime_type,
                    size_bytes,
                    sha256,
                    status,
                    page_count,
                    chunk_count,
                    error,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["document_id"],
                    record["collection_id"],
                    record["filename"],
                    record["safe_filename"],
                    record["mime_type"],
                    record["size_bytes"],
                    record["sha256"],
                    record["status"],
                    record["page_count"],
                    record["chunk_count"],
                    record["error"],
                    record["created_at"],
                    record["updated_at"],
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    # 根据 document_id 查询文档
    def get_document(
        self,
        document_id: str,
    ) -> DocumentRecord | None:
        connection = sqlite3.connect(
            str(self.db_path)
        )
        connection.row_factory = sqlite3.Row

        try:
            row = connection.execute(
                """
                SELECT
                    document_id,
                    collection_id,
                    filename,
                    safe_filename,
                    mime_type,
                    size_bytes,
                    sha256,
                    status,
                    page_count,
                    chunk_count,
                    error,
                    created_at,
                    updated_at
                FROM documents
                WHERE document_id = ?
                  AND status != 'deleted'
                """,
                (document_id,),
            ).fetchone()
        finally:
            connection.close()

        if row is None:
            return None

        return DocumentRecord(**dict(row))
    
    # 根据文件哈希查询，常用于上传前判断文件是否已存在。
    def get_document_by_sha256(
        self,
        sha256: str,
    ) -> DocumentRecord | None:
        connection = sqlite3.connect(
            str(self.db_path)
        )
        connection.row_factory = sqlite3.Row

        try:
            row = connection.execute(
                """
                SELECT
                    document_id,
                    collection_id,
                    filename,
                    safe_filename,
                    mime_type,
                    size_bytes,
                    sha256,
                    status,
                    page_count,
                    chunk_count,
                    error,
                    created_at,
                    updated_at
                FROM documents
                WHERE sha256 = ?
                  AND status != 'deleted'
                """,
                (sha256,),
            ).fetchone()
        finally:
            connection.close()

        if row is None:
            return None

        return DocumentRecord(**dict(row))
    
    def update_document_status(
        self,
        document_id: str,
        *,
        status: DocumentStatus,
        updated_at: str,
        page_count: int | None = None,
        chunk_count: int | None = None,
        error: str = "",
    ) -> None:
        connection = sqlite3.connect(
            str(self.db_path)
        )

        try:
            cursor = connection.execute(
                """
                UPDATE documents
                SET
                    status = ?,
                    page_count = COALESCE(?, page_count),
                    chunk_count = COALESCE(?, chunk_count), 
                    error = ?,
                    updated_at = ?
                WHERE document_id = ?
                """,
                (
                    status,
                    page_count,
                    chunk_count,
                    error,
                    updated_at,
                    document_id,
                ),
            )

            if cursor.rowcount == 0:
                raise KeyError(
                    f"Document not found: {document_id}"
                )

            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def list_documents(
            self,
            status: DocumentStatus | None = None,
    ) -> list[DocumentRecord]:
        connection = sqlite3.connect(
        str(self.db_path)
        )
        #  让查询结果可以像字典一样按列名取值
        connection.row_factory = sqlite3.Row

        try:
            query = """
                SELECT
                    document_id,
                    collection_id,
                    filename,
                    safe_filename,
                    mime_type,
                    size_bytes,
                    sha256,
                    status,
                    page_count,
                    chunk_count,
                    error,
                    created_at,
                    updated_at
                FROM documents
            """
            parameters: tuple[str, ...] = ()

            if status is not None:
                query += (
                    " WHERE status = ?"
                    " AND status != 'deleted'"
                )
                parameters = (status,)
            else:
                query += " WHERE status != 'deleted'"
            # 追加排序
            query += """
                ORDER BY updated_at DESC, document_id DESC
            """

            rows = connection.execute(
                query,
                parameters,
            ).fetchall()
            
        finally:
            connection.close()

        return [
            DocumentRecord(**dict(row))
            for row in rows
        ]

    def archive_and_delete_document(
        self,
        record: DocumentRecord,
        *,
        event_type: str,
        created_at: str,
        error: str = "",
    ) -> None:
        connection = sqlite3.connect(
            str(self.db_path)
        )

        try:
            connection.execute(
                # 写入审计表
                """
                INSERT INTO document_audit ( 
                    document_id,
                    collection_id,
                    filename,
                    safe_filename,
                    mime_type,
                    size_bytes,
                    sha256,
                    event_type,
                    status_before,
                    status_after,
                    error,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["document_id"],
                    record["collection_id"],
                    record["filename"],
                    record["safe_filename"],
                    record["mime_type"],
                    record["size_bytes"],
                    record["sha256"],
                    event_type,
                    record["status"],
                    "deleted",               # 哪些状态是删除状态
                    error,
                    created_at,
                ),
            )

            cursor = connection.execute(
                """
                DELETE FROM documents
                WHERE document_id = ?
                """,
                (record["document_id"],),
            )

            if cursor.rowcount == 0:
                raise KeyError(
                    f"Document not found: {record['document_id']}"
                )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    def get_statistics(self) -> dict[str, int]:
        # 链接数据库
        connection = sqlite3.connect(
            str(self.db_path)
        )

        try:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS document_count,
                    COALESCE(SUM(chunk_count), 0)
                        AS chunk_count,
                    COALESCE(SUM(
                        CASE
                            WHEN status = 'indexed'
                            THEN 1 ELSE 0
                        END
                    ), 0) AS indexed_count,
                    COALESCE(SUM(
                        CASE
                            WHEN status = 'processing'
                            THEN 1 ELSE 0
                        END
                    ), 0) AS processing_count,
                    COALESCE(SUM(
                        CASE
                            WHEN status = 'failed'
                            THEN 1 ELSE 0
                        END
                    ), 0) AS failed_count,
                    COALESCE(SUM(
                        CASE
                            WHEN status = 'pending'
                            THEN 1 ELSE 0
                        END
                    ), 0) AS pending_count
                FROM documents
                WHERE status != 'deleted'
                """
            ).fetchone()
        finally:
            connection.close()

        return {
            "document_count": int(row[0]),
            "chunk_count": int(row[1]),
            "indexed_count": int(row[2]),
            "processing_count": int(row[3]),
            "failed_count": int(row[4]),
            "pending_count": int(row[5]),
        }
