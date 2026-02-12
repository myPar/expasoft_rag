import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from llama_index.core.embeddings import BaseEmbedding
from typing import List


class BertaEmbedding(BaseEmbedding):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        pooling: str = "mean",
        document_prefix: str = "search_document: ",
        query_prefix: str = "search_query: ",
    ):
        self.device = device

        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling

        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self.embeddings_dim: int = 768

    def pool(self, hidden_state, mask, pooling_method="mean"):
        if pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif pooling_method == "cls":
            return hidden_state[:, 0]
        else:
            assert False and "invalid pooling method"

    def _embed(self, texts: list[str], prefix: str | None = None):
        if prefix:
            texts = [prefix + t for t in texts]

        inputs = self.tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = self.pool(
            outputs.last_hidden_state,
            inputs["attention_mask"],
            pooling_method='mean'
        )
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
    