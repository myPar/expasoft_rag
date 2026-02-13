from llama_index.core.embeddings import BaseEmbedding
from typing import List
from tqdm import tqdm
import torch
import os
import pickle


def build_embeddings(embedder: BaseEmbedding, 
                     inputs: List[str], 
                     batch_size: int = 32, 
                     cache_embeddings:bool=False,
                     cache_name:str='embeddings.pkl'
                     ):
    batch_count = (len(inputs) + batch_size - 1) // batch_size
    result_embeddings = []

    with torch.no_grad():
        with tqdm(total=batch_count, desc="creating embeddings...") as pbar:
            for i in range(batch_count):
                batch = inputs[i * batch_size: (i + 1) * batch_size]
                batch_embeddings = embedder._get_text_embeddings(batch) # embed documents for storage
                result_embeddings += batch_embeddings
                pbar.update(1)

    if cache_embeddings:
        if not os.path.isfile(cache_name):
            with open(cache_name, 'wb') as f:
                pickle.dump(result_embeddings, f)

    return result_embeddings


def load_cached_embeddings(emb_cache_path:str):
    try:
        with open(emb_cache_path, 'rb') as f:
            embeddings = pickle.load(f)
    except Exception as e:
        raise Exception(f"can't load embeddings - {str(e)}")
    return embeddings
