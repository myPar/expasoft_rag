from llama_index.core.postprocessor.types import BaseNodePostprocessor
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class JinaiiReranker(BaseNodePostprocessor):
    device: str = 'cpu'
    tokenizer: AutoTokenizer = None
    model: AutoModelForSequenceClassification = None
    top_n: int = 4

    def __init__(self, top_n=4, device:str='cuda'):
        super().__init__(top_n=top_n)
        self.top_n = top_n
        self.device = 'cpu'
        if device == 'cuda':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = AutoModelForSequenceClassification.from_pretrained('models/jinaii', 
                                                                        local_files_only=True, 
                                                                        torch_dtype="auto",
                                                                        trust_remote_code=True,
                                                                        use_flash_attn=False,
                                                                        ).to(self.device)
        self.model.eval()

    def _postprocess_nodes(self, nodes, query_bundle=None):
        if query_bundle is None or not nodes:
            return []
        # Собираем тексты: сначала query, затем чанки
        with torch.no_grad():
            texts = [[query_bundle.query_str, node.node.get_content()] for node in nodes]
            scores = self.model.compute_score(texts, max_length=1024)     # 0-1 scores, list of floats

        for node, score in zip(nodes, scores):
            node.score = score  # set scores and make node with score
        return sorted(nodes, key=lambda x: x.score)[-self.top_n:]
