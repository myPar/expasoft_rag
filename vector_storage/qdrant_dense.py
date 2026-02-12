from typing import List, Dict, Optional
from itertools import count
from qdrant_client import AsyncQdrantClient, QdrantClient, models
from llama_index.vector_stores.qdrant import QdrantVectorStore
from tqdm import tqdm
from .base import BaseQdrantStoreFactory


class QdrantDenseStoreFactory(BaseQdrantStoreFactory):
    """
    Factory для создания и наполнения dense-only коллекций в Qdrant.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        vector_dim: int = 1024,
    ):
        self.url = url
        self.vector_name = "text-dense"
        self.vector_dim = vector_dim

        self.async_client = AsyncQdrantClient(url=url)
        self.sync_client = QdrantClient(url=url)

    async def create_collection(
        self,
        collection_name: str,
        distance: models.Distance = models.Distance.COSINE,
        recreate: bool = False,
    ):
        exists = await self.async_client.collection_exists(collection_name)

        if exists and recreate:
            await self.async_client.delete_collection(collection_name)
            exists = False

        if not exists:
            await self.async_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=distance,
                ),
            )

    async def ingest(
        self,
        collection_name: str,
        texts: List[Dict],
        embeddings: List[List[float]],
        batch_size: int = 64,
    ):
        assert len(texts) == len(embeddings), "Texts and embeddings must have same length"

        id_counter = count(start=0)

        def batch_iter():
            for i in range(0, len(texts), batch_size):
                yield texts[i:i + batch_size], embeddings[i:i + batch_size]

        total_batches = (len(texts) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches)

        for batch_texts, batch_embs in batch_iter():
            await self.async_client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=next(id_counter),
                        vector={
                            self.vector_name: batch_embs[i],
                        },
                        payload={
                            "uid": batch_texts[i]["uid"],
                            "text": batch_texts[i]["text"],
                        },
                    )
                    for i in range(len(batch_texts))
                ],
            )
            pbar.update(1)

        pbar.close()

    # llama index compatible Qdrant vector storage
    def as_vector_store(
        self,
        collection_name: str,
    ) -> QdrantVectorStore:

        return QdrantVectorStore(
            collection_name=collection_name,
            aclient=self.async_client,
            client=self.sync_client,
        )
