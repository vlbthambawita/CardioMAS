from __future__ import annotations

import ast
import operator as op

from cardiomas.schemas.tools import ToolResult


SAFE_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
}


def calculate_expression(expression: str) -> ToolResult:
    try:
        result = _evaluate(ast.parse(expression, mode="eval").body)
        return ToolResult(
            tool_name="calculate",
            ok=True,
            summary=f"Calculated result for expression: {expression}",
            data={"expression": expression, "result": result},
        )
    except Exception as exc:
        return ToolResult(tool_name="calculate", ok=False, summary="", error=str(exc))


def _evaluate(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Only simple arithmetic expressions are allowed.")
