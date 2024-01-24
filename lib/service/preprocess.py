from lib.entity.schema import RawYoutubeVideoSchema, YoutubeVideoSchema
import pandera as pa
import numpy as np
import pandas as pd
from pandera.typing import DataFrame


@pa.check_types
def transform_youtube_video_data(df: DataFrame[RawYoutubeVideoSchema]) -> YoutubeVideoSchema:
    df.dropna(subset=["extracted_at", "comment_count", "like_count", "view_count"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["extracted_at"] = df["extracted_at"].dt.tz_localize("Asia/Tokyo")
    # change tz from UTC to JST
    df["published_at"] = pd.to_datetime(df["published_at"])
    df["published_at"] = df["published_at"].dt.tz_convert("Asia/Tokyo")

    df["view_count"] = df["view_count"].astype(int)
    df["comment_count"] = df["comment_count"].astype(int)
    df["like_count"] = df["like_count"].astype(int)

    df["minutes_diff"] = (df["extracted_at"] - df["published_at"]).dt.total_seconds() / 60
    df["day_of_week_str"] = df["published_at"].dt.day_name()
    df["hour"] = df["published_at"].dt.hour
    df["sin_hour"] = np.sin(2*np.pi*df["hour"]/24)
    df["cos_hour"] = np.cos(2*np.pi*df["hour"]/24)
    df["duration_min"] = df["duration"].apply(lambda x: pd.Timedelta(x).total_seconds()/60)
    return df