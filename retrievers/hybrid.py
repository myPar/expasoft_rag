from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from fastembed import SparseTextEmbedding


class HybridRetriever:
    def __init__(
        self,
        hybrid_vector_storage: QdrantVectorStore,
        embed_model,    # dense embedder
        dense_top_k: int = 4,
        sparse_top_k: int = 20,
    ):
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=hybrid_vector_storage,
            embed_model=embed_model,
        )

        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k

    def as_query_engine(self, llm, node_postprocessors=None):
        return self.index.as_query_engine(
            llm=llm,
            similarity_top_k=self.dense_top_k,
            sparse_top_k=self.sparse_top_k,
            vector_store_query_mode="hybrid",
            node_postprocessors=node_postprocessors
        )
