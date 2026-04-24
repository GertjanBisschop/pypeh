"""
This part of the code should contain the reusable part
of the PyHBM library, aka, the part that would otherwise
be copied across projects. Ideally, it also contains a
specification on what the custom part should look like.
"""

from __future__ import annotations

import logging


from contextlib import contextmanager
from dataguard import Validator, ErrorCollector
from polars import DataFrame
from typing import TYPE_CHECKING

from pypeh.core.interfaces.outbound.dataops import ValidationInterface
from pypeh.core.models.validation_errors import ValidationErrorReport, EntityLocation
from pypeh.core.models.validation_dto import ValidationConfig
from pypeh.adapters.outbound.validation.pandera_adapter.parsers import parse_config, parse_error_report
from pypeh.adapters.outbound.dataops.dataframe_adapter import DataFrameAdapter

if TYPE_CHECKING:
    from typing import Mapping

logger = logging.getLogger(__name__)


class DataFrameValidationAdapter(DataFrameAdapter, ValidationInterface[DataFrame]):
    data_format = DataFrame

    def parse_configuration(self, config: ValidationConfig) -> Mapping:
        return parse_config(config)

    @contextmanager
    def get_error_collector(self):
        collector = ErrorCollector()
        try:
            yield collector
        finally:
            collector.clear_errors()

    def _validate(self, data: dict[str, list] | DataFrame, config: ValidationConfig) -> ValidationErrorReport:
        config_map = self.parse_configuration(config)
        validator = Validator.config_from_mapping(config=config_map, logger=logger)
        _ = validator.validate(data)

        with self.get_error_collector() as error_collector:
            report = parse_error_report(error_collector.get_errors())

        # Replace DataframeLocations with corresponding EntityLocation entries
        def get_data_item(data, row_index, column_name):
            if isinstance(data, dict):
                return data[column_name][row_index]
            if isinstance(data, DataFrame):
                return data.item(row_index, column_name)

        for group in report.groups:
            for error in group.errors:
                new_location_list = []
                if not error.locations:
                    continue
                for location in error.locations:
                    row_ids = getattr(location, "row_ids", None)
                    key_columns = getattr(location, "key_columns", None)
                    column_names = getattr(location, "column_names", None)
                    if row_ids and key_columns:
                        entity_ids = [
                            tuple(get_data_item(data, row_id, id_obs_prop) for id_obs_prop in key_columns)
                            for row_id in row_ids
                        ]
                        new_location_list.append(
                            EntityLocation(
                                location_type="entity",
                                identifying_property_list=key_columns,
                                identifying_property_values=entity_ids,
                                property_names=column_names,
                            )
                        )
                    else:
                        new_location_list.append(location)
                error.locations = new_location_list

        return report

    def _join_data(
        self,
        data: DataFrame,
        other_data: list[DataFrame],
        join_on: list[str],
        subset_fields_other: list[list[str]],
    ) -> DataFrame:
        assert len(other_data) == len(subset_fields_other)
        this = data
        for other, subset_fields in zip(other_data, subset_fields_other):
            selected = other.select(subset_fields)

            this = this.join(
                selected,
                on=join_on,
                how="left",
            )

        return this
