import pytest

from pypeh.adapters.extract.parsers import (
    parse_filter_expression,
    parse_filter_config,
)
from pypeh.core.models.extract_dto import FilterExpression, FilterConfig


@pytest.mark.dataframe
class TestFilterExpressionToDto:
    @pytest.mark.parametrize(
        "input_data, expected_output",
        [
            (
                FilterExpression(
                    command="is_in",
                    subject=["country"],
                    arg_values=["BE", "BR"],
                ),
                {
                    "command": "is_in",
                    "arg_values": ["BE", "BR"],
                    "arg_columns": None,
                    "subject": ["country"],
                },
            ),
            (
                FilterExpression(
                    command="is_greater_than",
                    arg_columns=["col1"],
                ),
                {
                    "command": "is_greater_than",
                    "arg_values": None,
                    "arg_columns": ["col1"],
                    "subject": None,
                },
            ),
            (
                FilterExpression(
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
                {
                    "check_case": "conjunction",
                    "expressions": [
                        {
                            "command": "is_greater_than_or_equal_to",
                            "arg_values": [18],
                            "arg_columns": None,
                            "subject": ["age"],
                        },
                        {
                            "command": "is_in",
                            "arg_values": ["BE", "BR"],
                            "arg_columns": None,
                            "subject": ["country"],
                        },
                    ],
                },
            ),
        ],
    )
    def test_parse_filter_expression(self, input_data, expected_output):
        assert parse_filter_expression(input_data) == expected_output

    def test_parse_filter_expression_conjunction_more_than_two(self):
        expression = FilterExpression(
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
                FilterExpression(
                    command="is_greater_than",
                    subject=["income"],
                    arg_values=[1000],
                ),
            ],
        )

        assert parse_filter_expression(expression) == {
            "check_case": "conjunction",
            "expressions": [
                {
                    "command": "is_greater_than_or_equal_to",
                    "arg_values": [18],
                    "arg_columns": None,
                    "subject": ["age"],
                },
                {
                    "command": "is_in",
                    "arg_values": ["BE", "BR"],
                    "arg_columns": None,
                    "subject": ["country"],
                },
                {
                    "command": "is_greater_than",
                    "arg_values": [1000],
                    "arg_columns": None,
                    "subject": ["income"],
                },
            ],
        }


@pytest.mark.dataframe
class TestFilterConfigToDto:
    def test_parse_filter_config_without_select(self):
        config = FilterConfig(
            name="adults",
            filter_expression=FilterExpression(
                command="is_greater_than_or_equal_to",
                subject=["age"],
                arg_values=[18],
            ),
        )

        result = parse_filter_config(config)

        assert result == {
            "name": "adults",
            "filter": {
                "command": "is_greater_than_or_equal_to",
                "arg_values": [18],
                "arg_columns": None,
                "subject": ["age"],
            },
        }

    def test_parse_filter_config_with_select(self):
        config = FilterConfig(
            name="adults",
            filter_expression=FilterExpression(
                command="is_greater_than_or_equal_to",
                subject=["age"],
                arg_values=[18],
            ),
            select=["age", "country"],
        )

        result = parse_filter_config(config)

        assert result["select"] == ["age", "country"]
