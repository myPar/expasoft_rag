from qdrant_client import AsyncQdrantClient, QdrantClient, models
from llama_index.vector_stores.qdrant import QdrantVectorStore

from itertools import count
from tqdm import tqdm
from fastembed import SparseTextEmbedding
import pickle
import os
from typing import List
from .base import BaseQdrantStoreFactory


sparse_embedder = SparseTextEmbedding(model_name="Qdrant/bm25", language="russian")


def sparse_query_fn(texts):
    """For queries: return (list of indices, list of values)"""
    embeddings = list(sparse_embedder.embed(texts))
    indices = [emb.indices.tolist() for emb in embeddings]
    values = [emb.values.tolist() for emb in embeddings]
    return indices, values

def sparse_doc_fn(texts):
    """For documents: return list of SparseEmbedding objects"""
    return list(sparse_embedder.embed(texts))


class QdrantHybridStoreFactory(BaseQdrantStoreFactory):
    def __init__(
        self,
        url: str = "http://localhost:6333",
        dense_vector_name: str = "text-dense",
        sparse_vector_name: str = "text-sparse",
        dense_dim: int = 1024,
    ):
        self.url = url
        self.async_client = AsyncQdrantClient(url=url)
        self.sync_client = QdrantClient(url=url)

        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name

        self.dense_dim = dense_dim

    async def create_collection(self, collection_name: str, 
                                recreate:bool=False, 
                                distance: models.Distance = models.Distance.COSINE
                                ):
        exists = await self.async_client.collection_exists(collection_name)

        if exists and recreate:
            await self.async_client.delete_collection(collection_name)
            exists = False

        if not exists:
            await self.async_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    self.dense_vector_name: models.VectorParams(
                        size=self.dense_dim,
                        distance=distance,
                    ),
                },
                sparse_vectors_config={
                    self.sparse_vector_name: models.SparseVectorParams(
                        index=models.SparseIndexParams(),
                    ),
                },
            )

    async def ingest(
        self,
        collection_name: str,
        texts: list[dict],
        dense_embeddings: List[List[float]],
        batch_size: int = 32,
    ):

        sparse_embedder = SparseTextEmbedding(
            model_name="Qdrant/bm25",
            language="russian",
        )
        sparse_embeddings = list(
            sparse_embedder.embed([p["text"] for p in texts])
        )

        id_counter = count(start=0)

        def batch_iter():
            for i in range(0, len(texts), batch_size):
                yield (
                    texts[i: i + batch_size],
                    dense_embeddings[i: i + batch_size],
                    sparse_embeddings[i: i + batch_size],
                )

        pbar = tqdm(total=(len(texts) + batch_size - 1) // batch_size)

        # insert embeddings by batch:
        for batch in batch_iter():
            pars, dense_embs, sparse_embs = batch

            await self.async_client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=next(id_counter),
                        vector={
                            self.dense_vector_name: dense_embs[i],
                            self.sparse_vector_name: sparse_embs[i].as_object(),
                        },
                        payload={
                            "paragraph_id": pars[i]["uid"],
                            "text": pars[i]["text"],
                        },
                    )
                    for i in range(len(pars))
                ],
            )
            pbar.update(1)

    # llama index compatible Qdrant vector storage
    def as_vector_store(
        self,
        collection_name: str,
    ) -> QdrantVectorStore:

        return QdrantVectorStore(
            collection_name=collection_name,
            aclient=self.async_client,
            client=self.sync_client,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
            sparse_vector_name=self.sparse_vector_name,
            sparse_query_fn=sparse_query_fn,
            sparse_doc_fn=sparse_doc_fn,            
        )
