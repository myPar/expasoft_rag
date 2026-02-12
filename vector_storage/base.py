from abc import ABC, abstractmethod
from typing import List, Dict, Any
from qdrant_client import models


class BaseQdrantStoreFactory(ABC):
    """
    Abstract interface for vector store factories.
    Defines lifecycle: create collection, ingest data, expose vector store.
    """

    @abstractmethod
    async def create_collection(
        self,
        collection_name: str,
        distance: models.Distance = models.Distance.COSINE,
        recreate: bool = False,
    ) -> None:
        """Create collection if not exists."""
        pass

    @abstractmethod
    async def ingest(
        self,
        collection_name: str,
        texts: List[Dict],
        embeddings: Any,
        batch_size: int = 64,
    ) -> None:
        """Insert embeddings + payload into collection."""
        pass

    @abstractmethod
    def as_vector_store(
        self,
        collection_name: str,
    ):
        """Return LlamaIndex-compatible vector store wrapper."""
        pass
