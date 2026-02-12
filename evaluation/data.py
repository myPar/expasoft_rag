from dataclasses import dataclass
from typing import List


@dataclass
class Query:
    query: str
    response: str
    contexts: List[str]
    reference: str

    @staticmethod
    def from_dict(dict_data: dict):
        query = dict_data['query']
        response = dict_data['response']
        contexts = dict_data['contexts']
        reference = dict_data['reference']

        return Query(query, response, contexts, reference)