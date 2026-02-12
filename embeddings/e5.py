import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from llama_index.core.embeddings import BaseEmbedding
from torch import Tensor
import torch.nn.functional as F
from typing import List


class E5InstructEmbedding(BaseEmbedding):
    def __init__(self, 
                 model_name: str='intfloat/multilingual-e5-large-instruct', 
                 device: str = "cuda",
                 query_prefix: str | None = f'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ',
                 document_prefix: str | None = None # None document prefix for e5
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.embeddings_dim: int = 1024

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _embed(self, texts, prefix: str | None = None):
        if prefix:
            texts = [prefix + t for t in texts]

        inputs = self.tokenizer(
            texts, 
            max_length=512, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = self.average_pool(outputs.last_hidden_state, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()

    def _get_text_embeddings(self, texts: list[str]) -> List[List[float]]:
        return self._embed(texts, self.document_prefix)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text], self.document_prefix)[0]
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query], self.query_prefix)[0]

    # async:
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
