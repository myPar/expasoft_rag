from llama_index.core.postprocessor.types import BaseNodePostprocessor
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEReranker(BaseNodePostprocessor):
    device: str = 'cpu'
    tokenizer: AutoTokenizer = None
    model: AutoModelForSequenceClassification = None
    top_n: int = 2

    def __init__(self, top_n=2, device:str='cuda'):
        super().__init__(top_n=top_n)
        self.top_n = top_n
        self.device = 'cpu'
        if device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
        self.model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3').to(self.device)

    def _postprocess_nodes(self, nodes, query_bundle=None):
        if query_bundle is None or not nodes:
            return []
        # Собираем тексты: сначала query, затем чанки
        with torch.no_grad():
            texts = [[query_bundle.query_str, node.node.get_content()] for node in nodes]
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()     # 0-1 scores

        for node, score in zip(nodes, scores.tolist()):
            node.score = score  # set scores and make node with score
        return sorted(nodes, key=lambda x: x.score)[-self.top_n:]
