from dataclasses import dataclass
from typing import Callable

import sympy as sp

ExampleBuilder = Callable[[], list[tuple[sp.Function, sp.Expr]]]


@dataclass
class ExamplePDE:
    description: str
    diff_ord: int
    builder: ExampleBuilder
    first_indep: sp.Symbol = sp.symbols("t")


def _kdv() -> list[tuple[sp.Function, sp.Expr]]:
    t, x = sp.symbols("t x")
    a = sp.symbols("a", constant=True)
    u = sp.Function("u")(t, x)
    u_t = a * u**2 * sp.Derivative(u, x) - sp.Derivative(u, x, 3)
    return [(u, u_t)]


def _allen_cahn() -> list[tuple[sp.Function, sp.Expr]]:
    t, x = sp.symbols("t x")
    u = sp.Function("u")(t, x)
    u_t = sp.Derivative(u, x, 2) + u - u**3
    return [(u, u_t)]


def _brusselator() -> list[tuple[sp.Function, sp.Expr]]:
    t, x = sp.symbols("t x")
    d_1, d_2, a, b = sp.symbols("d_1 d_2 a b", constant=True)
    lambd = sp.symbols("lambda", constant=True)
    u = sp.Function("u")(t, x)
    v = sp.Function("v")(t, x)
    u_t = d_1 * sp.Derivative(u, x) + lambd * (1 - (b + 1) * u + b * u**2 * v)
    v_t = d_2 * sp.Derivative(v, x) + lambd * a**2 * (u - u**2 * v)
    return [(u, u_t), (v, v_t)]


EXAMPLES: dict[str, ExamplePDE] = {
    "kdv": ExamplePDE(
        description="Korteweg–de Vries equation u_t = a u^2 u_x - u_xxx",
        diff_ord=3,
        builder=_kdv,
    ),
    "allen-cahn": ExamplePDE(
        description="Allen–Cahn equation u_t = u_xx + u - u^3",
        diff_ord=2,
        builder=_allen_cahn,
    ),
    "brusselator": ExamplePDE(
        description="Brusselator system with two species",
        diff_ord=1,
        builder=_brusselator,
    ),
}
