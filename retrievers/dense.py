from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore


class DenseRetriever:
    def __init__(
        self,
        dense_vector_storage: QdrantVectorStore,
        embed_model,
        similarity_top_k: int = 5,
    ):
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=dense_vector_storage,
            embed_model=embed_model,
        )

        self.similarity_top_k = similarity_top_k

    def as_query_engine(
        self,
        llm,
        node_postprocessors=None,
    ):
        return self.index.as_query_engine(
            llm=llm,
            similarity_top_k=self.similarity_top_k,
            node_postprocessors=node_postprocessors,
        )
