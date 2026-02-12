import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from llama_index.core.embeddings import BaseEmbedding
from torch import Tensor
import torch.nn.functional as F
from typing import List


class BGEm3Embedding(BaseEmbedding):
    def __init__(self, 
                 model_name: str="deepvk/USER-bge-m3",
                 device: str = "cuda",
                 query_prefix: str | None = None,
                 document_prefix: str | None = None
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.embeddings_dim: int = 1024

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
        embeddings = outputs[0][:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

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
