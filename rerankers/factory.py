from .jinaii_reranker import JinaiiReranker
from .bge_reranker import BGEReranker


def build_reranker(reranker_name:str, rerank_top_k:int=4):
    assert reranker_name.lower() in {'jinaii', 'bge'}
    if reranker_name == 'jinaii':
        return JinaiiReranker(top_n=rerank_top_k)
    elif reranker_name == 'bge':
        return BGEReranker(top_n=rerank_top_k)

    assert False
