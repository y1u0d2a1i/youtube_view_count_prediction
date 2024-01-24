import pandas as pd
from pandera.typing import DataFrame
from lib.interface.pipeline import BasePipeline
from lib.entity.schema import YoutubeVideoSchema
from lib.service.feature import WordEmbeddingService

from sentence_transformers import SentenceTransformer


class FeatureExtractionPipeline(BasePipeline):
    def __init__(
            self,
            original_df: DataFrame[YoutubeVideoSchema],text_embedding_model_name:str="paraphrase-multilingual-MiniLM-L12-v2"
        ):
        self.original_df = original_df
        self.text_embedding_model_name = text_embedding_model_name

    def run(self):
        transformed_df = self.original_df.copy()
        transformed_df = pd.get_dummies(
            transformed_df,
            columns=["day_of_week_str"],
            dtype=float
        )

        # embeddingを追加する
        model = SentenceTransformer(self.text_embedding_model_name)
        corpus = list(transformed_df["title"])
        corpus_embedding_service = WordEmbeddingService(model)
        corpus_embedding_df = corpus_embedding_service.get_embedding_df(corpus)

        transformed_df = pd.concat([transformed_df, corpus_embedding_df], axis=1)

        return transformed_df