from pandera.typing import DataFrame

from lib.entity.schema import RawYoutubeVideoSchema, YoutubeVideoSchema
from lib.interface.repository import BaseYoutubeVideoRepository
from lib.service.preprocess import transform_youtube_video_data


class YoutubeVideoDataService():
    def __init__(self, repository: BaseYoutubeVideoRepository):
        self.repository = repository

    def get_raw_data(self) -> DataFrame[RawYoutubeVideoSchema]:
        df = self.repository.load_data()
        return df

    def get_transformed_data(self) -> DataFrame[YoutubeVideoSchema]:
        df = self.get_raw_data()
        df_transformed = transform_youtube_video_data(df)
        return df_transformed




