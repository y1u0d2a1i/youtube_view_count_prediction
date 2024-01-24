from typing import List, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class WordEmbeddingService():
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def get_embedding(self, corpus: Union[str, List[str]]) -> np.ndarray:
        return self.model.encode(corpus)
    

    def get_embedding_df(self, corpus: Union[str, List[str]]) -> pd.DataFrame:
        embeddings = self.get_embedding(corpus)
        embedding_cols = [f"text_embeddings_{i}" for i in range(embeddings.shape[1])]
        embedding_df = pd.DataFrame(
            embeddings,
            columns=embedding_cols
        )
        return embedding_df