"""Small exact-arithmetic verifier for cited math steps; no general theorem claims."""
from __future__ import annotations

import ast
import hashlib
import operator
from fractions import Fraction
from typing import Any

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow}


def _exact(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 1_000_000:
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _exact(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        left, right = _exact(node.left), _exact(node.right)
        if isinstance(node.op, ast.Pow) and (right.denominator != 1 or abs(right) > 12):
            raise ValueError("unsupported exponent")
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("division by zero")
        value = OPS[type(node.op)](left, right)
        if value.numerator.bit_length() > 10000 or value.denominator.bit_length() > 10000:
            raise ValueError("arithmetic expression too large")
        return value
    raise ValueError("unsupported mathematical expression")


def verify_cited_equality(store: Any, trace_id: str, event_id: str, equality: str) -> dict[str, Any]:
    """Verify only a literal equality present in one full original Train event."""
    text = store.get(trace_id, event_id)["text"]
    if len(equality) > 160 or equality.count("=") != 1 or equality not in text:
        return {"status": "unverifiable", "reason": "equality_not_literal_in_cited_event"}
    left, right = (part.strip() for part in equality.split("="))
    try:
        lhs = _exact(ast.parse(left, mode="eval").body)
        rhs = _exact(ast.parse(right, mode="eval").body)
    except (SyntaxError, ValueError, OverflowError, ZeroDivisionError):
        return {"status": "unverifiable", "reason": "unsupported_or_unsafe_expression"}
    return {"status": "mathematically_verified" if lhs == rhs else "mathematically_refuted",
            "method": "exact_rational_ast", "trace_id": trace_id, "event_id": event_id,
            "event_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "equality_sha256": hashlib.sha256(equality.encode()).hexdigest(),
            "scope": "this cited arithmetic equality only; not the whole solution or causal root"}


def classify_math_insight(insight: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    # A procedural hypothesis may coexist with verified local calculations.
    verified = [item for item in checks if item.get("status") == "mathematically_verified"]
    return {"insight_id": insight["insight_id"],
            "level": "mathematically_verified" if verified else "procedural_hypothesis",
            "verified_local_claims": len(verified),
            "causal_status": "hypothesis_only", "checks": checks}
