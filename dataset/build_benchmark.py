import numpy as np
import json


# get paragraphs to remove:
with open('no_wiki_paragraphs.json', 'r', encoding='utf-8') as f:
    no_wiki_paragraphs = json.loads(f.read())
no_wiki_pids = {par['uid'] for par in no_wiki_paragraphs}


def get_no_wiki_queries(queries):
    global no_wiki_pids
    answer = set()

    for query in queries:
        answer_paragraphs = set(query['paragraphs_uids']['with_answer'])
        if answer_paragraphs.issubset(no_wiki_pids):
            answer.add(query['uid'])

    return answer


def create_bench(bench_size, queries: list[dict], random_state = 42):
    skip_queries: set = get_no_wiki_queries(queries)
    # filter bad queries: no answer paragraphs or in skip_queries
    queries = [q for q in queries if len(q['paragraphs_uids']['with_answer']) > 0 and q['uid'] not in skip_queries]
    np.random.seed(seed=random_state)
    idxs = np.arange(0, len(queries))
    np.random.shuffle(idxs) # shuffle inplace

    bench_queries = np.array(queries)[idxs][:bench_size]
    result = []

    for i, query in enumerate(bench_queries):
        item = {'uid' : query['uid'],
                'text': query['question_text'],
                'answers': [item['label'] for item in query['answers']],
                'answer_text': query['answer_text'],
                'answer_paragraphs': query['paragraphs_uids']['with_answer']
                }
        result.append(item)

    return result
