import pandera as pa
from pandera.engines import pandas_engine
from pandera.typing import Series


class RawYoutubeVideoSchema(pa.DataFrameModel):
    channel_name: Series[str] = pa.Field()
    channel_country: Series[str] = pa.Field(nullable=True)
    channel_default_language: Series[str] = pa.Field(nullable=True)
    subscriber_count: Series[int] = pa.Field()
    video_count: Series[int] = pa.Field(nullable=True)
    channel_topic_ids: Series[str] = pa.Field(nullable=True)
    channel_topic_catrgories: Series[str] = pa.Field(nullable=True)
    published_at: Series[str] = pa.Field(nullable=True)
    channel_id: Series[str] = pa.Field(nullable=True)
    title: Series[str] = pa.Field(nullable=True)
    description: Series[str] = pa.Field(nullable=True)
    thumbnail_url: Series[str] = pa.Field(nullable=True)
    channel_title: Series[str] = pa.Field(nullable=True)
    tags: Series[str] = pa.Field(nullable=True)
    category_id: Series[int] = pa.Field(nullable=True)
    default_language: Series[str] = pa.Field(nullable=True)
    duration: Series[str] = pa.Field(nullable=True)
    view_count: Series[float] = pa.Field(nullable=True)
    favorite_count: Series[int] = pa.Field(nullable=True)
    comment_count: Series[float] = pa.Field(nullable=True)
    like_count: Series[float] = pa.Field(nullable=True)
    video_id: Series[str] = pa.Field(nullable=True)
    extracted_at: Series[pandas_engine.DateTime] = pa.Field(nullable=True)


class YoutubeVideoSchema(RawYoutubeVideoSchema):
    published_at: Series[pandas_engine.DateTime(to_datetime_kwargs = {"format":"%Y-%m-%dT%H:%M:%S"}, tz="Asia/Tokyo")] = pa.Field(nullable=True)
    extracted_at: Series[pandas_engine.DateTime(to_datetime_kwargs = {"format":"%Y-%m-%dT%H:%M:%S"}, tz="Asia/Tokyo")] = pa.Field(nullable=True)
    minutes_diff: Series[int] = pa.Field(gt=0)
    day_of_week_str: Series[str] = pa.Field()
    hour: Series[int] = pa.Field(in_range={"min_value": 1, "max_value": 24})
    sin_hour: Series[float] = pa.Field()
    cos_hour: Series[float] = pa.Field()
    duration_min: Series[float] = pa.Field(ge=0)
    view_count: Series[int] = pa.Field(ge=0)
    comment_count: Series[int] = pa.Field(ge=0)
    like_count: Series[int] = pa.Field(ge=0)