import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from llama_index.core.embeddings import BaseEmbedding
from torch import Tensor
import torch.nn.functional as F
from typing import List
from pydantic import PrivateAttr


class E5InstructEmbedding(BaseEmbedding):
    _tokenizer: AutoTokenizer = PrivateAttr()
    _model: AutoModel = PrivateAttr()
    _device: str = "cuda"
    _query_prefix: str = ""
    _document_prefix: str | None = None
    _embeddings_dim: int = 1024

    def __init__(self, 
                 model_name: str='intfloat/multilingual-e5-large-instruct', 
                 device: str = "cuda",
                 query_prefix: str | None = f'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ',
                 document_prefix: str | None = None, # None document prefix for e5,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        model_name='models/e5'  # TODO, remove later
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self._model.eval()
        self._device = device
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._embeddings_dim: int = 1024

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _embed(self, texts, prefix: str | None = None):
        if prefix:
            texts = [prefix + t for t in texts]

        inputs = self._tokenizer(
            texts, 
            max_length=512, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        embeddings = self.average_pool(outputs.last_hidden_state, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()

    def _get_text_embeddings(self, texts: list[str]) -> List[List[float]]:
        return self._embed(texts, self._document_prefix)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text], self._document_prefix)[0]
    
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query], self._query_prefix)[0]

    # async:
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
