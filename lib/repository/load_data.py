import pandas as pd
from pandera.typing import DataFrame

from lib.entity.schema import RawYoutubeVideoSchema
from lib.interface.repository import BaseYoutubeVideoRepository


class YoutubeVideoDataRepository(BaseYoutubeVideoRepository):
    def __init__(self, path2data: str):
        self.path2data = path2data

    def load_data(self) -> DataFrame[RawYoutubeVideoSchema]:
        df = pd.read_excel(self.path2data, sheet_name="Sheet1")
        df["channel_title"] = df["channel_title"].astype(str)
        df["channel_name"] = df["channel_name"].astype(str)
        df["title"] = df["title"].astype(str)
        df = RawYoutubeVideoSchema.validate(df)
        return df