from dataclasses import dataclass
from typing import List


@dataclass
class Query:
    text: str
    answer_text: str
    uid: int

    @staticmethod
    def from_dict(dict_data: dict):
        text = dict_data['text']
        answer_text = dict_data['answer_text']
        uid = dict_data['uid']

        return Query(text, answer_text, uid)
