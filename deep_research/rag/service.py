"""
完整流程：
SQLite pending
  ↓
SQLite processing
  ↓
解析
  ↓
切 Chunk
  ↓
写入 Qdrant，Chunk status=pending
  ↓
校验写入数量
  ↓
Qdrant status=indexed
  ↓
SQLite status=indexed
"""
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from .lexical import (
    SqliteBm25Index,
    build_lexical_query,
)
from .query_rewrite import (
    QueryRewriter,
    build_rewritten_lexical_query,
)
from .fusion import reciprocal_rank_fusion_scores
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)
from .retrieval_text import (
    build_metadata_weighted_embedding_text,
    restore_original_content,
)

class RagService:
    def __init__(
        self,
        *,
        client: QdrantClient,
        embeddings: Embeddings,
        vector_store: QdrantVectorStore,
        collection_name: str,
        metadata_weight: int = 2,
        bm25_index: SqliteBm25Index | None = None,
        query_rewriter: QueryRewriter | None = None,
        candidate_k: int = 32,
        rrf_k: int = 60,
    ) -> None:
        self.client = client
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.collection_name = collection_name
        if metadata_weight < 0:
            raise ValueError("metadata_weight must be non-negative")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")
        self.metadata_weight = metadata_weight
        self.bm25_index = bm25_index
        self.query_rewriter = query_rewriter
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def close(self) -> None:
        if self.bm25_index is not None:
            self.bm25_index.close()
        self.client.close()

    @staticmethod
    def _bm25_row(document: Document) -> dict[str, object]:
        metadata = dict(document.metadata)
        document_id = metadata.get("document_id")
        chunk_id = metadata.get("chunk_id")
        status = metadata.get("status")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("chunk metadata must contain document_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("chunk metadata must contain chunk_id")
        if status not in {"pending", "indexed"}:
            raise ValueError("chunk metadata must contain a valid status")
        return {
            "_id": chunk_id,
            "text": document.page_content,
            "title": metadata.get("title") or metadata.get("filename") or document_id,
            "metadata": metadata,
        }

    def rebuild_lexical_index(self) -> None:
        """Rebuild BM25 from the indexed Qdrant payloads at application startup."""
        if self.bm25_index is None:
            return

        rows: list[dict[str, object]] = []
        offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100_000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = getattr(point, "payload", None) or {}
                content = payload.get("page_content")
                metadata = payload.get("metadata")
                if not isinstance(content, str) or not isinstance(metadata, dict):
                    continue
                if metadata.get("status") != "indexed":
                    continue
                chunk_id = metadata.get("chunk_id")
                document_id = metadata.get("document_id")
                if not isinstance(chunk_id, str) or not chunk_id:
                    continue
                if not isinstance(document_id, str) or not document_id:
                    continue
                document = Document(
                    page_content=restore_original_content(content),
                    metadata=dict(metadata),
                )
                try:
                    rows.append(self._bm25_row(document))
                except ValueError:
                    continue

            if next_offset is None:
                break
            offset = next_offset

        self.bm25_index.replace_chunks(rows)

    # 在metadata设置status状态
    def mark_document_chunks_pending(
        self,
        document_id: str,
    ) -> None:
        normalized_document_id = document_id.strip()
        if not normalized_document_id:
            raise ValueError("document_id must not be empty")

        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"status": "pending"},       #  写 "pending"
            key="metadata",                      #  合并进 metadata 嵌套，不覆盖其他字段
            points=Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=normalized_document_id),
                    )
                ]
            ),
            wait=True,
        )
        if self.bm25_index is not None:
            self.bm25_index.update_document_status(
                normalized_document_id,
                "pending",
            )

    def index_chunks(
        self,
        chunks: list[Document],
    ) -> int:
        # 判断是否空chunk
        if not chunks:
            return 0

        ids: list[str] = []

        #循环取出chunks并
        for chunk in chunks:
            raw_chunk_id = chunk.metadata.get("chunk_id")
            if not isinstance(raw_chunk_id, str):
                raise ValueError("chunk metadata must contain chunk_id")
            ids.append(raw_chunk_id)

        normalized_chunks: list[Document] = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            status = metadata.get("status", "pending")
            if status not in {"pending", "indexed"}:
                raise ValueError("chunk metadata status must be pending or indexed")
            if "status" in metadata:
                normalized_chunks.append(chunk)
            else:
                normalized_chunks.append(
                    Document(
                        page_content=chunk.page_content,
                        metadata={**metadata, "status": status},
                    )
                )

        embedding_documents: list[Document] = []
        for chunk in normalized_chunks:
            embedding_text = build_metadata_weighted_embedding_text(
                chunk.page_content,
                chunk.metadata,
                weight=self.metadata_weight,
            )
            if embedding_text == chunk.page_content:
                embedding_documents.append(chunk)
            else:
                embedding_documents.append(
                    Document(
                        page_content=embedding_text,
                        metadata=dict(chunk.metadata),
                    )
                )

        written_ids = self.vector_store.add_documents(
            documents=embedding_documents,
            ids=ids,
        )
        if self.bm25_index is not None:
            self.bm25_index.upsert_chunks(
                self._bm25_row(chunk)
                for chunk in normalized_chunks
            )
        return len(written_ids)

    #  Qdrant 向量删除方法
    def delete_document_chunks(
        self,
        document_id: str,
    ) -> None:
        # 去掉首尾空格
        normalized_document_id = (
            document_id.strip()
        )
        # 空字符串拦截
        if not normalized_document_id:
            raise ValueError(
                "document_id must not be empty"
            )
        # 删除该文档所有 Chunk
        self.client.delete(
            collection_name=self.collection_name,
            # 精确匹配 payload 里的 document_id
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(
                            value=normalized_document_id
                        ),
                    )
                ]
            ),
            # 同步等删除完成才返回
            wait=True
        )
        if self.bm25_index is not None:
            self.bm25_index.delete_document(normalized_document_id)

    def mark_document_chunks_indexed(
        self,
        document_id: str,
    ) -> None:
        normalized_document_id = document_id.strip()
        if not normalized_document_id:
            raise ValueError("document_id must not be empty")

        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"status": "indexed"},       # 写 "indexed"
            key="metadata",                      # 合并进 metadata 嵌套，不覆盖其他字段
            points=Filter(
                must=[
                    FieldCondition(
                        key="metadata.document_id",
                        match=MatchValue(value=normalized_document_id),
                    )
                ]
            ),
            wait=True,
        )
        if self.bm25_index is not None:
            self.bm25_index.update_document_status(
                normalized_document_id,
                "indexed",
            )

    def search_chunks(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "query must not be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )
        # 条件，找出那些status是indexed的chunk
        conditions = [
            FieldCondition(
                key="metadata.status",
                match=MatchValue(
                    value="indexed"
                ),
            )
        ]

        # 标准化 document_id
        if document_id is not None:
            normalized_document_id = (
                document_id.strip()
            )

            if not normalized_document_id:
                raise ValueError(
                    "document_id must not be empty"
                )

            # 只在指定文档的 chunk 里搜，而不是在所有文档里搜。
            conditions.append(
                FieldCondition(
                    key="metadata.document_id",
                    match=MatchValue(
                        value=normalized_document_id
                    ),
                )
            )

        raw_results = self.vector_store.similarity_search_with_score(
            query=normalized_query,
            k=top_k,
            filter=Filter(
                must=conditions
            ),
        )
        return [
            (
                Document(
                    page_content=restore_original_content(document.page_content),
                    metadata=dict(document.metadata),
                )
                if restore_original_content(document.page_content)
                != document.page_content
                else document,
                score,
            )
            for document, score in raw_results
        ]

    def _retrieve_documents_by_ids(
        self,
        chunk_ids: list[str],
        *,
        document_id: str | None = None,
    ) -> dict[str, Document]:
        if not chunk_ids:
            return {}
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=chunk_ids,
            with_payload=True,
            with_vectors=False,
        )
        documents: dict[str, Document] = {}
        for record in records:
            payload = getattr(record, "payload", None) or {}
            content = payload.get("page_content")
            metadata = payload.get("metadata")
            if not isinstance(content, str) or not isinstance(metadata, dict):
                continue
            chunk_id = metadata.get("chunk_id")
            if (
                not isinstance(chunk_id, str)
                or not chunk_id
                or metadata.get("status") != "indexed"
                or (
                    document_id is not None
                    and metadata.get("document_id") != document_id
                )
            ):
                continue
            documents[chunk_id] = Document(
                page_content=restore_original_content(content),
                metadata=dict(metadata),
            )
        return documents

    def _search_hybrid_children(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        normalized_document_id = None
        if document_id is not None:
            normalized_document_id = document_id.strip()
            if not normalized_document_id:
                raise ValueError("document_id must not be empty")

        candidate_k = max(self.candidate_k, top_k)
        dense_results = self.search_chunks(
            normalized_query,
            top_k=candidate_k,
            document_id=normalized_document_id,
        )
        dense_ids = [
            document.metadata["chunk_id"]
            for document, _score in dense_results
            if isinstance(document.metadata.get("chunk_id"), str)
        ]

        if self.bm25_index is None:
            return dense_results

        try:
            if self.query_rewriter is None:
                lexical_query = build_lexical_query(normalized_query)
            else:
                rewrite = self.query_rewriter.rewrite(normalized_query)
                lexical_query = build_rewritten_lexical_query(
                    normalized_query,
                    rewrite.terms,
                )
        except Exception:
            # A rewrite failure falls back to the original lexical query.
            lexical_query = build_lexical_query(normalized_query)

        try:
            bm25_ids = self.bm25_index.search_ids(
                lexical_query,
                top_k=candidate_k,
                document_id=normalized_document_id,
            )
        except Exception:
            # BM25 is derived from Qdrant; Dense remains usable if it fails.
            return dense_results

        scores, first_seen = reciprocal_rank_fusion_scores(
            [dense_ids, bm25_ids],
            rrf_k=self.rrf_k,
        )
        fused_ids = [
            chunk_id
            for chunk_id, _score in sorted(
                scores.items(),
                key=lambda item: (-item[1], first_seen[item[0]], item[0]),
            )
        ]
        documents_by_id = {
            document.metadata["chunk_id"]: document
            for document, _score in dense_results
            if isinstance(document.metadata.get("chunk_id"), str)
        }
        missing_ids = [chunk_id for chunk_id in fused_ids if chunk_id not in documents_by_id]
        documents_by_id.update(
            self._retrieve_documents_by_ids(
                missing_ids,
                document_id=normalized_document_id,
            )
        )
        return [
            (documents_by_id[chunk_id], scores[chunk_id])
            for chunk_id in fused_ids
            if chunk_id in documents_by_id
        ]

    def search_hybrid_chunks(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Return fused child chunks for offline and lower-level callers."""
        return self._search_hybrid_children(
            query,
            top_k=top_k,
            document_id=document_id,
        )[:top_k]

    def _expand_one_parent(
        self,
        document: Document,
        score: float,
    ) -> tuple[Document, float]:
        metadata = document.metadata
        parent_id = metadata.get("parent_id")
        child_ids = metadata.get("parent_child_ids")
        if (
            not isinstance(parent_id, str)
            or not parent_id
            or not isinstance(child_ids, list)
            or not child_ids
            or not all(isinstance(child_id, str) and child_id for child_id in child_ids)
        ):
            return document, score

        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=child_ids,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return document, score

        parts: list[tuple[int, str, str]] = []
        records_by_id = {}
        for record in records:
            payload = getattr(record, "payload", None) or {}
            record_metadata = payload.get("metadata")
            record_content = payload.get("page_content")
            if not isinstance(record_metadata, dict) or not isinstance(record_content, str):
                return document, score
            record_id = record_metadata.get("chunk_id")
            chunk_index = record_metadata.get("chunk_index")
            if (
                not isinstance(record_id, str)
                or record_id not in child_ids
                or record_metadata.get("document_id") != metadata.get("document_id")
                or record_metadata.get("status") != "indexed"
                or not isinstance(chunk_index, int)
            ):
                return document, score
            records_by_id[record_id] = record
            parts.append((chunk_index, record_id, restore_original_content(record_content).strip()))

        if len(records_by_id) != len(child_ids) or any(not text for _index, _id, text in parts):
            return document, score

        parts.sort(key=lambda item: (item[0], item[1]))
        parent_metadata = dict(metadata)
        parent_metadata["parent_content"] = True
        return (
            Document(
                page_content="\n\n".join(text for _index, _id, text in parts),
                metadata=parent_metadata,
            ),
            score,
        )

    def search_contexts(
        self,
        query: str,
        *,
        top_k: int,
        document_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Return at most ``top_k`` distinct parent contexts for the tool."""
        child_results = self._search_hybrid_children(
            query,
            top_k=top_k,
            document_id=document_id,
        )
        contexts: list[tuple[Document, float]] = []
        seen_parent_ids: set[str] = set()
        for document, score in child_results:
            metadata = document.metadata
            parent_id = metadata.get("parent_id")
            child_ids = metadata.get("parent_child_ids")
            has_parent = (
                isinstance(parent_id, str)
                and bool(parent_id)
                and isinstance(child_ids, list)
                and bool(child_ids)
                and all(isinstance(child_id, str) and child_id for child_id in child_ids)
            )
            chunk_id = metadata.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                continue
            key = f"parent:{parent_id}" if has_parent else f"chunk:{chunk_id}"
            if key in seen_parent_ids:
                continue
            seen_parent_ids.add(key)
            contexts.append(
                self._expand_one_parent(document, score)
                if has_parent
                else (document, score)
            )
            if len(contexts) >= top_k:
                break
        return contexts

    def expand_results_to_parents(
        self,
        results: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        """Return one larger context document per retrieved parent window."""
        expanded: list[tuple[Document, float]] = []
        seen_parent_ids: set[str] = set()

        for document, score in results:
            metadata = document.metadata
            parent_id = metadata.get("parent_id")
            child_ids = metadata.get("parent_child_ids")
            if (
                not isinstance(parent_id, str)
                or not parent_id
                or not isinstance(child_ids, list)
                or not child_ids
                or parent_id in seen_parent_ids
            ):
                if parent_id not in seen_parent_ids:
                    expanded.append((document, score))
                continue

            try:
                records = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[str(child_id) for child_id in child_ids],
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:
                expanded.append((document, score))
                seen_parent_ids.add(parent_id)
                continue

            parts: list[tuple[int, str]] = []
            for record in records:
                payload = getattr(record, "payload", None) or {}
                record_metadata = payload.get("metadata")
                record_content = payload.get("page_content")
                if not isinstance(record_metadata, dict) or not isinstance(record_content, str):
                    continue
                content = restore_original_content(record_content).strip()
                if not content:
                    continue
                chunk_index = record_metadata.get("chunk_index", 0)
                parts.append((int(chunk_index), content))

            if not parts:
                expanded.append((document, score))
            else:
                parts.sort(key=lambda item: item[0])
                parent_metadata = dict(metadata)
                parent_metadata["parent_content"] = True
                expanded.append(
                    (
                        Document(
                            page_content="\n\n".join(content for _index, content in parts),
                            metadata=parent_metadata,
                        ),
                        score,
                    )
                )
            seen_parent_ids.add(parent_id)

        return expanded
