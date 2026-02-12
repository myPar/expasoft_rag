import argparse
import asyncio
import sys
import json

from embeddings.e5 import E5InstructEmbedding
from vector_storage.qdrant_hybrid import QdrantHybridStoreFactory
from llm.open_router import create_llm
from vector_storage.setup import setup_database
from retrievers.hybrid import HybridRetriever
from evaluation.benchmark_runner import AsyncBenchmarkRunner
from evaluation.evaluator import MetricsEvaluator
from evaluation.metrics import get_main_metrics
from dataset.build_benchmark import create_bench
from evaluation.data import Query


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
        help="Path to benchmark configuration file"
    )

    parser.add_argument(
        "--load_embeddings",
        action="store_true",
        help="Load embeddings from cache file."
    )

    parser.add_argument(
        "--cache_embeddings",
        action="store_true",
        help="Save built embeddings as a pickle file (ignored if --load_embeddings is set)."
    )

    parser.add_argument(
        "--embeddings_cache_file",
        type=str,
        default=None,
        help="Path to embeddings cache file."
    )

    parser.add_argument(
        "--paragraphs_path",
        type=str,
        default="../dataset/doc_base/chunked_paragraphs.json",
        help="Path to wiki paragraphs."
    )
    parser.add_argument(
        "--queries_path",
        type=str,
        default="../dataset/doc_base/RuBQ_2.0_test.json",
        help="Path to queries benchmark will build from"
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
    # setup database - create collection, build embeddings
    await setup_database(qdrant_factory=qdrant_factory,
                         embedder=embedder,
                         base_name=args.base_name,
                         build_database=args.build_database,
                         load_embeddings=args.load_embeddings,
                         cache_embeddings=args.cache_embeddings,
                         embeddings_cache_file=args.embeddings_cache_file,
                         paragraphs_path=args.paragraphs_path
                         )

    # get qdrant storage compatible with llamaindex
    qdrant_storage = qdrant_factory.as_vector_store(args.base_name)

    # load bench config and queries:
    with open(args.config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    with open(args.queries_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    # llms:
    engine_llm = create_llm(config, 'engine_llm')
    judge_llm = create_llm(config, 'judge_llm')

    # retriever:
    hybrid_retriever = HybridRetriever(qdrant_storage, 
                                       embedder, 
                                       dense_top_k=args.similarity_top_k, 
                                       sparse_top_k=args.sparse_top_k
                                       )
    query_engine = hybrid_retriever.as_query_engine(llm=engine_llm)
    
    evaluator = MetricsEvaluator(metrics=get_main_metrics(judge_llm), attempts=config['request_attempts'])
    bench_runner = AsyncBenchmarkRunner(evaluator=evaluator, max_concurrent=config['max_concurrent'])
    bench_data = create_bench(bench_size=config['benchmark']['size'], queries=queries,random_state=config['benchmark']['size'])
    await bench_runner.run(query_engine=query_engine, bench_data=bench_data, bench_name=args.base_name)

    return 0


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)
