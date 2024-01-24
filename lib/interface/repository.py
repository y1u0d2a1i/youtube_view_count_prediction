import abc

from pandera.typing import DataFrame

from lib.entity.schema import RawYoutubeVideoSchema


class BaseYoutubeVideoRepository(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load_data(self) -> DataFrame[RawYoutubeVideoSchema]:
        raise NotImplementedError