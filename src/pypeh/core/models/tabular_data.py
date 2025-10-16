from __future__ import annotations

import logging

from collections import defaultdict
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING

from pypeh.core.cache.containers import CacheContainerView
from pypeh.core.models.constants import ObservablePropertyValueType

if TYPE_CHECKING:
    from polars import DataFrame

logger = logging.getLogger(__name__)


class Column(BaseModel):
    label: str
    datatype: str | None = None
    observable_property_id: str
    table_id: str | None = None


class Table(BaseModel):
    id: str
    label: str
    columns: dict[str, Column] = Field(default_factory=dict, description="Column label to column map")
    schema: dict[str, str] | None = Field(None, description="Column label to datatype mapping")
    observation_id: str

    def model_post_init(self, __context):
        for column in self.columns.values():
            column.table_id = self.id

    def _add_column(self, column: Column):
        self.columns[column.label] = column
        column.table_id = self.id


class TableGroup(dict[str, Table]):
    """Metadata on a collection of dataframes comprising a single dataset"""


class DataSet:
    def __init__(self, data: dict[str, DataFrame], cache_view: CacheContainerView):
        self.data = dict[str, DataFrame]
        self.metadata: TableGroup = self.collect_metadata(cache_view)

        # label-based view on the DataSet
        self._label_index: dict[str, set[str]] = defaultdict(set)

    def _add_table(self):
        pass

    def _add_column(self):
        pass

    def get_table_by_label(self, label: str) -> Table:
        pass

    def get_column_by_label(self, label: str, table_label: str) -> Column:
        pass

    def collect_metadata(self, cache_view: CacheContainerView) -> TableGroup:
        pass

    def collect_schema(self, cache_view: CacheContainerView) -> dict[str, dict[str, ObservablePropertyValueType]]:
        ret = {}

        # filter out DataLayout or DataLayoutSections

        """
        sections = getattr(layout, "sections")
        if sections is None:
            raise ValueError("No sections found in DataLayout")
        for section in sections:
            label = getattr(section, "ui_label")
            elements = getattr(section, "elements")
            if elements is None:
                logger.info("DataLayout does not contain elements. Cannot determine observable_entity_value_types.")
                return None
            for element in elements:
                element_label = getattr(element, "label")
                observable_property_id = getattr(element, "observable_property")
                observable_property = self.cache.get(observable_property_id, "ObservableProperty")
                if observable_property is None:
                    logger.info(
                        f"Could not find {observable_property_id} in cache. Cannot determine observable_property_value_types. "
                    )
                    return None
                assert isinstance(label, str)
                value_type = getattr(observable_property, "value_type")
                if flatten:
                    ret[element_label] = ObservablePropertyValueType(value_type)
                else:
                    if label not in ret:
                        ret[label] = {}
                    ret[label][element_label] = ObservablePropertyValueType(value_type)

        """
        return ret
