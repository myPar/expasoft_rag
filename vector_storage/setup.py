import sys
import json
from typing import Any
from embeddings.build_embeddings import build_embeddings, load_cached_embeddings
from .base import BaseQdrantStoreFactory
from llama_index.core.embeddings import BaseEmbedding


async def setup_database(
    base_name: str,
    build_database: bool,
    load_embeddings: bool,
    cache_embeddings: bool,
    embeddings_cache_file: str,
    paragraphs_path: str,
    qdrant_factory: BaseQdrantStoreFactory,
    embedder: BaseEmbedding,
) -> int:
    # create collection (if not exists)
    try:
        await qdrant_factory.create_collection(
            collection_name=base_name
        )
    except Exception as e:
        print(f"Failed to create collection: {e}", file=sys.stderr)
        return 1

    # build database (if requested)
    if not build_database:
        return 0

    # load paragraphs texts:
    with open(paragraphs_path, "r", encoding="utf-8") as f:
        inputs = json.load(f)[:2000]    # TODO: remove slicing (it just for debugging)

    texts = [item["text"] for item in inputs]

    # load or build embeddings
    if load_embeddings:
        try:
            embeddings = load_cached_embeddings(embeddings_cache_file)
        except Exception as e:
            print(f"Failed to load embeddings: {e}", file=sys.stderr)
            return 1
    else:
        try:
            embeddings = build_embeddings(
                embedder,
                inputs=texts,   # texts as an input
                cache_embeddings=cache_embeddings,
                cache_name=embeddings_cache_file,
            )

        except Exception as e:
            print(f"Failed to build embeddings: {e}", file=sys.stderr)
            return 1

    # ingest
    try:
        await qdrant_factory.ingest(    # paragraphs as an input
            base_name,
            texts=inputs,
            dense_embeddings=embeddings,
        )
    except Exception as e:
        print(f"Failed to ingest data: {e}", file=sys.stderr)
        return 1

    return 0
