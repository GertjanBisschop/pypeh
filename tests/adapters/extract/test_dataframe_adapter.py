import pytest

from pypeh.core.interfaces.dataops import DataExtractInterface
from pypeh.core.models.extract_dto import FilterConfig, FilterExpression


@pytest.mark.dataframe
class TestDataFrameExtractAdapter:
    def get_adapter(self):
        from pypeh.adapters.extract.dataframe_adapter import (
            DataFrameExtractAdapter,
        )

        return DataFrameExtractAdapter()

    def test_getting_default_adapter_from_interface(self):
        from pypeh.adapters.extract.dataframe_adapter import (
            DataFrameExtractAdapter,
        )

        adapter_class = DataExtractInterface.get_default_adapter_class()

        assert adapter_class is DataFrameExtractAdapter
        assert isinstance(adapter_class(), DataExtractInterface)

    def test_filter_simple_expression(self):
        import polars as pl

        adapter = self.get_adapter()
        df = pl.DataFrame({"age": [10, 20, 30], "country": ["BE", "BR", "NL"]})
        config = FilterConfig(
            name="adults",
            filter_expression=FilterExpression(
                command="is_greater_than_or_equal_to",
                subject=["age"],
                arg_values=[18],
            ),
        )

        result = adapter._filter(df, config)

        assert result["age"].to_list() == [20, 30]
        assert result["country"].to_list() == ["BR", "NL"]

    def test_filter_with_select(self):
        import polars as pl

        adapter = self.get_adapter()
        df = pl.DataFrame({"age": [10, 20, 30], "country": ["BE", "BR", "NL"]})
        config = FilterConfig(
            name="adults",
            filter_expression=FilterExpression(
                command="is_greater_than_or_equal_to",
                subject=["age"],
                arg_values=[18],
            ),
            select=["country"],
        )

        result = adapter._filter(df, config)

        assert result.columns == ["country"]
        assert result["country"].to_list() == ["BR", "NL"]

    def test_filter_conjunction(self):
        import polars as pl

        adapter = self.get_adapter()
        df = pl.DataFrame({"age": [10, 20, 30], "country": ["BE", "BR", "NL"]})
        config = FilterConfig(
            name="adult_belgians_or_brazilians",
            filter_expression=FilterExpression(
                command="conjunction",
                arg_expressions=[
                    FilterExpression(
                        command="is_greater_than_or_equal_to",
                        subject=["age"],
                        arg_values=[18],
                    ),
                    FilterExpression(
                        command="is_in",
                        subject=["country"],
                        arg_values=["BE", "BR"],
                    ),
                ],
            ),
        )

        result = adapter._filter(df, config)

        assert result["country"].to_list() == ["BR"]
