from __future__ import annotations

from typing import Mapping, TYPE_CHECKING

from pypeh.core.models.extract_dto import FilterConfig, FilterExpression

if TYPE_CHECKING:
    pass


def parse_single_filter_expression(expression: FilterExpression) -> Mapping:
    return {
        "command": expression.command,
        "arg_values": expression.arg_values,
        "arg_columns": expression.arg_columns,
        "subject": expression.subject,
    }


def parse_filter_expression(expression: FilterExpression) -> Mapping:
    if conditional_expr := expression.conditional_expression:
        case = "condition"
        exp_1 = parse_filter_expression(conditional_expr)
        arg_expressions = expression.arg_expressions
        if arg_expressions is None or (
            isinstance(arg_expressions, list) and len(arg_expressions) == 0
        ):
            exp = expression.model_copy()
            exp.conditional_expression = None
            exp_2 = parse_filter_expression(exp)
        else:
            if len(arg_expressions) != 1:
                raise NotImplementedError(
                    "Conditional expressions with a filter condition "
                    "currently support exactly one arg expression. "
                    f"received={len(arg_expressions)}."
                )
            exp_2 = parse_filter_expression(arg_expressions[0])

        return {
            "check_case": case,
            "expressions": [exp_1, exp_2],
        }
    if expression.command in ("conjunction", "disjunction"):
        if expression.arg_expressions is not None:
            if len(expression.arg_expressions) < 2:
                raise NotImplementedError(
                    "Conjunction/disjunction expressions require at least "
                    "two arg expressions. "
                    f"received={len(expression.arg_expressions)}."
                )
            case = expression.command
            expressions = [
                parse_filter_expression(arg_expression)
                for arg_expression in expression.arg_expressions
            ]
            return {
                "check_case": case,
                "expressions": expressions,
            }
        else:
            raise ValueError
    return parse_single_filter_expression(expression)


def parse_filter_config(config: FilterConfig) -> Mapping:
    ret = {
        "name": config.name,
        "filter": parse_filter_expression(config.filter_expression),
    }
    if config.select is not None:
        ret["select"] = config.select

    return ret
