from __future__ import annotations

import logging

from dataguard import Filter
from polars import DataFrame
from typing import TYPE_CHECKING

from pypeh.core.interfaces.dataops import DataExtractInterface
from pypeh.core.models.extract_dto import FilterConfig
from pypeh.adapters.extract.parsers import parse_filter_config
from pypeh.adapters.dataops.dataframe_adapter import DataFrameAdapter

if TYPE_CHECKING:
    from typing import Mapping

logger = logging.getLogger(__name__)


class DataFrameExtractAdapter(
    DataFrameAdapter, DataExtractInterface[DataFrame]
):
    data_format = DataFrame

    def parse_configuration(self, config: FilterConfig) -> Mapping:
        return parse_filter_config(config)

    def _filter(self, data: DataFrame, config: FilterConfig) -> DataFrame:
        config_map = self.parse_configuration(config)
        filt = Filter.config_from_mapping(config_map)
        return filt.apply(data)
