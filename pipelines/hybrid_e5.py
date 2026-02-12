import argparse
import asyncio
import sys
import json

from embeddings.build_embeddings import build_embeddings, load_embeddings
from embeddings.e5 import E5InstructEmbedding
from vector_storage.qdrant_hybrid import QdrantHybridStoreFactory


def positive_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")

    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be an integer greater than 0")

    return ivalue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process similarity and sparse top-k parameters."
    )
    parser.add_argument("--similarity_top_k", type=positive_int, required=True)
    parser.add_argument("--sparse_top_k", type=positive_int, required=True)

    parser.add_argument(
        "--config_path",
        type=str,
        default="../configs/benchmark_config.json",
        help="Path to benchmark configuration file",
    )

    parser.add_argument(
        "--load_embeddings",
        action="store_true",
        help="Load embeddings from cache file.",
    )

    parser.add_argument(
        "--cache_embeddings",
        action="store_true",
        help="Save built embeddings as a pickle file (ignored if --load_embeddings is set).",
    )

    parser.add_argument(
        "--embeddings_cache_file",
        type=str,
        default=None,
        help="Path to embeddings cache file.",
    )

    parser.add_argument(
        "--paragraphs_path",
        type=str,
        default="../dataset/doc_base/chunked_paragraphs.json",
        help="Path to wiki paragraphs.",
    )

    parser.add_argument(
        "--build_database",
        action="store_true",
        help="Create new Qdrant database with embeddings building/loading.",
    )

    parser.add_argument(
        "--base_name",
        type=str,
        required=True,
        help="Qdrant database name.",
    )

    return parser.parse_args()


async def main(args: argparse.Namespace) -> int:
    # embedder
    embedder = E5InstructEmbedding()
    embeddings_dim = embedder.embeddings_dim

    # storage
    qdrant_factory = QdrantHybridStoreFactory(dense_dim=embeddings_dim)

    # create collection (if not exists)
    try:
        await qdrant_factory.create_collection(
            collection_name=args.base_name
        )
    except Exception as e:
        print(f"Failed to create collection: {e}", file=sys.stderr)
        return 1

    # build our database:
    if args.build_database:
        # load or build embeddings
        if args.load_embeddings:
            try:
                embeddings = load_embeddings(args.embeddings_cache_file)
            except Exception as e:
                print(f"Failed to load embeddings: {e}", file=sys.stderr)
                return 1
        else:
            try:
                with open(args.paragraphs_path, "r", encoding="utf-8") as f:
                    inputs = json.load(f)

                texts = [item["text"] for item in inputs]

                embeddings = build_embeddings(
                    embedder,
                    inputs=texts,
                    cache_embeddings=args.cache_embeddings,
                    cache_name=args.embeddings_cache_file,
                )
            except Exception as e:
                print(f"Failed to build embeddings: {e}", file=sys.stderr)
                return 1
        # insert embeddings and inputs:
        qdrant_factory.ingest(args.base_name, texts=inputs, dense_embeddings=embeddings)

    # get qdrant storage compatible with llamaindex
    qdrant_storage = qdrant_factory.as_vector_store(args.base_name)


    return 0


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)
