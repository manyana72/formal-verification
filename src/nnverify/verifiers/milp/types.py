from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    FALSIFIED = "FALSIFIED"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"


@dataclass
class VerificationResult:
    status: VerificationStatus
    worst_margin: float | None
    solve_time: float | None
    solver_name: str
    adv_example: list[float] | None


@dataclass
class LinExpr:
    terms: Dict[Any, float]
    const: float = 0.0


def expr_const(c: float) -> LinExpr:
    return LinExpr(terms={}, const=float(c))


def expr_var(var: Any, coeff: float = 1.0) -> LinExpr:
    return LinExpr(terms={var: float(coeff)}, const=0.0)


def expr_add(a: LinExpr, b: LinExpr) -> LinExpr:
    terms: Dict[Any, float] = dict(a.terms)
    for v, c in b.terms.items():
        terms[v] = terms.get(v, 0.0) + c
    return LinExpr(terms=terms, const=a.const + b.const)


def expr_sub(a: LinExpr, b: LinExpr) -> LinExpr:
    return expr_add(a, expr_mul(b, -1.0))


def expr_mul(a: LinExpr, scalar: float) -> LinExpr:
    scalar = float(scalar)
    return LinExpr(
        terms={v: c * scalar for v, c in a.terms.items()},
        const=a.const * scalar,
    )

