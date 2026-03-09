from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Tuple

from nnverify.verifiers.milp.types import LinExpr, expr_add, expr_const, expr_mul, expr_sub


class BaseBackend:
    solver_name: str

    def add_var(self, lb: float, ub: float, binary: bool = False) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def add_binary(self) -> Any:
        return self.add_var(0.0, 1.0, binary=True)

    def add_le(self, lhs: LinExpr, rhs: LinExpr) -> None:
        raise NotImplementedError  # pragma: no cover - interface

    def add_ge(self, lhs: LinExpr, rhs: LinExpr) -> None:
        raise NotImplementedError  # pragma: no cover - interface

    def add_eq(self, lhs: LinExpr, rhs: LinExpr) -> None:
        raise NotImplementedError  # pragma: no cover - interface

    def set_objective_max(self, expr: LinExpr) -> None:
        raise NotImplementedError  # pragma: no cover - interface

    def solve(self, time_limit_s: float | None = None) -> Tuple[str, float | None, float]:
        raise NotImplementedError  # pragma: no cover - interface

    def get_value(self, var: Any) -> float:
        raise NotImplementedError  # pragma: no cover - interface

    @staticmethod
    def expr_const(c: float) -> LinExpr:
        return expr_const(c)

    @staticmethod
    def expr_var(var: Any, coeff: float = 1.0) -> LinExpr:
        from nnverify.verifiers.milp.types import expr_var as _expr_var

        return _expr_var(var, coeff=coeff)

    @staticmethod
    def expr_add(a: LinExpr, b: LinExpr) -> LinExpr:
        return expr_add(a, b)

    @staticmethod
    def expr_sub(a: LinExpr, b: LinExpr) -> LinExpr:
        return expr_sub(a, b)

    @staticmethod
    def expr_mul(a: LinExpr, scalar: float) -> LinExpr:
        return expr_mul(a, scalar)


class OrtoolsCbcBackend(BaseBackend):
    def __init__(self) -> None:
        from ortools.linear_solver import pywraplp

        solver = pywraplp.Solver.CreateSolver("CBC")
        if solver is None:
            solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
        if solver is None:
            raise RuntimeError("Failed to create CBC solver via OR-Tools.")
        self.solver = solver
        self.solver_name = "CBC"
        self._vars: list[Any] = []
        self._objective_expr: LinExpr | None = None

    def add_var(self, lb: float, ub: float, binary: bool = False) -> Any:
        if binary:
            v = self.solver.IntVar(0.0, 1.0, "")
        else:
            v = self.solver.NumVar(float(lb), float(ub), "")
        self._vars.append(v)
        return v

    def _to_solver_expr(self, expr: LinExpr):
        lin = self.solver.Sum([coeff * var for var, coeff in expr.terms.items()])
        if expr.const:
            lin = lin + float(expr.const)
        return lin

    def add_le(self, lhs: LinExpr, rhs: LinExpr) -> None:
        diff = expr_sub(lhs, rhs)
        self.solver.Add(self._to_solver_expr(diff) <= 0.0)

    def add_ge(self, lhs: LinExpr, rhs: LinExpr) -> None:
        diff = expr_sub(lhs, rhs)
        self.solver.Add(self._to_solver_expr(diff) >= 0.0)

    def add_eq(self, lhs: LinExpr, rhs: LinExpr) -> None:
        diff = expr_sub(lhs, rhs)
        self.solver.Add(self._to_solver_expr(diff) == 0.0)

    def set_objective_max(self, expr: LinExpr) -> None:
        self._objective_expr = expr
        self.solver.Maximize(self._to_solver_expr(expr))

    def solve(self, time_limit_s: float | None = None) -> Tuple[str, float | None, float]:
        if time_limit_s is not None:
            self.solver.set_time_limit(int(max(time_limit_s, 0.0) * 1000))
        t0 = time.time()
        status = self.solver.Solve()
        elapsed = time.time() - t0

        from ortools.linear_solver import pywraplp

        status_map: Dict[int, str] = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
        }
        status_str = status_map.get(status, "UNKNOWN")
        obj_val: float | None
        if self._objective_expr is not None and status in (
            pywraplp.Solver.OPTIMAL,
            pywraplp.Solver.FEASIBLE,
        ):
            obj_val = float(self.solver.Objective().Value())
        else:
            obj_val = None
        return status_str, obj_val, elapsed

    def get_value(self, var: Any) -> float:
        return float(var.solution_value())


class GurobiBackend(BaseBackend):
    def __init__(self) -> None:
        import gurobipy as gp

        self.gp = gp
        self.model = gp.Model()
        self.model.Params.OutputFlag = 0
        self.solver_name = "Gurobi"
        self._vars: list[Any] = []

    def add_var(self, lb: float, ub: float, binary: bool = False) -> Any:
        if binary:
            vtype = self.gp.GRB.BINARY
        else:
            vtype = self.gp.GRB.CONTINUOUS
        v = self.model.addVar(lb=float(lb), ub=float(ub), vtype=vtype)
        self._vars.append(v)
        return v

    def _to_gurobi_expr(self, expr: LinExpr):
        lin = self.gp.LinExpr()
        for var, coeff in expr.terms.items():
            lin.add(var, float(coeff))
        if expr.const:
            lin += float(expr.const)
        return lin

    def add_le(self, lhs: LinExpr, rhs: LinExpr) -> None:
        diff = expr_sub(lhs, rhs)
        self.model.addLConstr(self._to_gurobi_expr(diff) <= 0.0)

    def add_ge(self, lhs: LinExpr, rhs: LinExpr) -> None:
        diff = expr_sub(lhs, rhs)
        self.model.addLConstr(self._to_gurobi_expr(diff) >= 0.0)

    def add_eq(self, lhs: LinExpr, rhs: LinExpr) -> None:
        diff = expr_sub(lhs, rhs)
        self.model.addLConstr(self._to_gurobi_expr(diff) == 0.0)

    def set_objective_max(self, expr: LinExpr) -> None:
        self.model.setObjective(self._to_gurobi_expr(expr), sense=self.gp.GRB.MAXIMIZE)

    def solve(self, time_limit_s: float | None = None) -> Tuple[str, float | None, float]:
        if time_limit_s is not None:
            self.model.Params.TimeLimit = float(max(time_limit_s, 0.0))
        t0 = time.time()
        self.model.optimize()
        elapsed = time.time() - t0

        status_code = self.model.Status
        status_map: Dict[int, str] = {
            self.gp.GRB.OPTIMAL: "OPTIMAL",
            self.gp.GRB.SUBOPTIMAL: "FEASIBLE",
            self.gp.GRB.TIME_LIMIT: "TIME_LIMIT",
            self.gp.GRB.INFEASIBLE: "INFEASIBLE",
            self.gp.GRB.UNBOUNDED: "UNBOUNDED",
        }
        status_str = status_map.get(status_code, "UNKNOWN")
        obj_val: float | None
        if self.model.SolCount > 0:
            obj_val = float(self.model.objVal)
        else:
            obj_val = None
        return status_str, obj_val, elapsed

    def get_value(self, var: Any) -> float:
        return float(var.X)


def get_backend(name: str) -> BaseBackend:
    lname = name.lower()
    if lname == "cbc":
        return OrtoolsCbcBackend()
    if lname == "gurobi":
        return GurobiBackend()
    if lname == "auto":
        try:
            return GurobiBackend()
        except Exception:
            return OrtoolsCbcBackend()
    raise ValueError(f"Unknown backend name {name!r}")

