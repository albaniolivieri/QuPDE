from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import sympy as sp


ExampleBuilder = Callable[[], List[Tuple[sp.Function, sp.Expr]]]


@dataclass
class ExamplePDE:
    """Container for a PDE example used by the CLI."""

    description: str
    diff_ord: int
    builder: ExampleBuilder
    first_indep: sp.Symbol = sp.symbols("t")


def _kdv() -> List[Tuple[sp.Function, sp.Expr]]:
    """Korteweg–de Vries equation."""
    t, x = sp.symbols("t x")
    a = sp.symbols("a", constant=True)
    u = sp.Function("u")(t, x)
    u_t = a * u**2 * sp.Derivative(u, x) - sp.Derivative(u, x, 3)
    return [(u, u_t)]


def _allen_cahn() -> List[Tuple[sp.Function, sp.Expr]]:
    """Allen–Cahn equation."""
    t, x = sp.symbols("t x")
    u = sp.Function("u")(t, x)
    u_t = sp.Derivative(u, x, 2) + u - u**3
    return [(u, u_t)]


def _brusselator() -> List[Tuple[sp.Function, sp.Expr]]:
    """Brusselator reaction-diffusion system."""
    t, x = sp.symbols("t x")
    d_1, d_2, a, b = sp.symbols("d_1 d_2 a b", constant=True)
    lambd = sp.symbols("lambda", constant=True)
    u = sp.Function("u")(t, x)
    v = sp.Function("v")(t, x)
    u_t = d_1 * sp.Derivative(u, x) + lambd * (1 - (b + 1) * u + b * u**2 * v)
    v_t = d_2 * sp.Derivative(v, x) + lambd * a**2 * (u - u**2 * v)
    return [(u, u_t), (v, v_t)]


EXAMPLES: Dict[str, ExamplePDE] = {
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
