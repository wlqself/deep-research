# Dense retrieval

Dense retrieval 会先把 query 转成 embedding，再与 Qdrant 中的 chunk 向量比较。当前 collection 使用 cosine distance。Qdrant 的 dense approximate-nearest-neighbor 路径由 HNSW 层支持；HNSW 是 dense retrieval 背后的索引，不是第二套 lexical retriever。

# Hybrid and reranking

Hybrid retriever 先分别从 dense retrieval 和 BM25 lexical retrieval 收集候选。两套排序结果应该用 Reciprocal Rank Fusion 合并，因为 cosine 分数和 BM25 分数不能直接比较。之后 reranker 对融合后的候选集重新打分，并返回最终较小的上下文集合。
