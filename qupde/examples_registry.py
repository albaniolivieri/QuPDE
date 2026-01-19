from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from importlib.util import module_from_spec, spec_from_file_location
from pprint import pprint
from typing import Iterable

import sympy as sp


@dataclass(frozen=True)
class ExampleSpec:
    id: str
    name: str
    filename: str
    func_eq: list[tuple[str, str]]
    diff_ord: int
    first_indep: str = "t"


@dataclass(frozen=True)
class ExampleData:
    id: str
    name: str
    description: str
    diff_ord: int
    first_indep: str
    func_eq: list[tuple[sp.Function, sp.Expr]]
    vars: str
    funcs: str
    equations: list[str]
    equations_latex: list[str]


EXAMPLE_SPECS: tuple[ExampleSpec, ...] = (
    ExampleSpec(
        id="kdv",
        name="Korteweg-de Vries",
        filename="KDV.py",
        func_eq=[("u", "u_t")],
        diff_ord=3,
    ),
    ExampleSpec(
        id="allen_cahn_equation",
        name="Allen-Cahn",
        filename="allen_cahn_equation.py",
        func_eq=[("u", "u_t")],
        diff_ord=2,
    ),
    ExampleSpec(
        id="brusselator_system",
        name="Brusselator",
        filename="brusselator_system.py",
        func_eq=[("u", "u_t"), ("v", "v_t")],
        diff_ord=1,
    ),
    ExampleSpec(
        id="cahn_hilliard_equation",
        name="Cahn-Hilliard",
        filename="cahn_hilliard_equation.py",
        func_eq=[("u", "u_t")],
        diff_ord=3,
    ),
    ExampleSpec(
        id="compacton_equations",
        name="Compacton",
        filename="compacton_equations.py",
        func_eq=[("u", "u_t")],
        diff_ord=3,
    ),
    ExampleSpec(
        id="dym_equation",
        name="Dym",
        filename="dym_equation.py",
        func_eq=[("u", "u_t")],
        diff_ord=4,
    ),
    ExampleSpec(
        id="euler_equations",
        name="Euler",
        filename="euler_equations.py",
        func_eq=[("rho", "rho_t"), ("u", "u_t"), ("p", "p_t")],
        diff_ord=1,
    ),
    ExampleSpec(
        id="fitz-hugh-nagamo",
        name="FitzHugh-Nagumo",
        filename="fitz-hugh-nagamo.py",
        func_eq=[("v", "v_t"), ("y", "y_t")],
        diff_ord=2,
    ),
    ExampleSpec(
        id="gray_scott_equations",
        name="Gray-Scott",
        filename="gray_scott_equations.py",
        func_eq=[("u", "u_t"), ("v", "v_t")],
        diff_ord=3,
    ),
    ExampleSpec(
        id="nonlinear_heat_equation",
        name="Nonlinear Heat",
        filename="nonlinear_heat_equation.py",
        func_eq=[("u", "u_t")],
        diff_ord=2,
    ),
    ExampleSpec(
        id="porous_medium_equation",
        name="Porous Medium",
        filename="porous_medium_equation.py",
        func_eq=[("u", "u_t")],
        diff_ord=3,
    ),
    ExampleSpec(
        id="schlogl_model",
        name="Schlogl",
        filename="schlogl_model.py",
        func_eq=[("u", "u_t")],
        diff_ord=2,
    ),
    ExampleSpec(
        id="schnakenberg_equations",
        name="Schnakenberg",
        filename="schnakenberg_equations.py",
        func_eq=[("u", "u_t"), ("v", "v_t")],
        diff_ord=2,
    ),
    ExampleSpec(
        id="schrodinger_equation",
        name="Schrodinger",
        filename="schrodinger_equation.py",
        func_eq=[("u", "u_t")],
        diff_ord=3,
    ),
    ExampleSpec(
        id="solar_wind",
        name="Solar Wind",
        filename="solar_wind.py",
        func_eq=[("v", "v_r")],
        diff_ord=1,
        first_indep="r",
    ),
    ExampleSpec(
        id="swift_hohenberg",
        name="Swift-Hohenberg",
        filename="swift_hohenberg.py",
        func_eq=[("u", "u_t")],
        diff_ord=4,
    ),
    ExampleSpec(
        id="tubular_reactor",
        name="Tubular Reactor",
        filename="tubular_reactor.py",
        func_eq=[("psi", "psi_t"), ("theta", "theta_t")],
        diff_ord=2,
    ),
    ExampleSpec(
        id="tubular_reactor_arrhenius",
        name="Tubular Reactor (Arrhenius)",
        filename="tubular_reactor_arrhenius.py",
        func_eq=[("psi", "psi_t"), ("theta", "theta_t"), ("y", "y_t")],
        diff_ord=2,
    ),
)


def _safe_module_name(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in stem)


def _load_module(filename: str) -> dict:
    module_name = f"qupde.examples.{_safe_module_name(filename)}"
    with resources.path("qupde.examples", filename) as module_path:
        spec = spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load example module {filename}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.__dict__


def _func_name(func: sp.Function) -> str:
    func_obj = func.func
    name = getattr(func_obj, "__name__", None) or getattr(func_obj, "name", None)
    return name if name else str(func_obj)


def _unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


@lru_cache(maxsize=1)
def load_examples() -> dict[str, ExampleData]:
    examples: dict[str, ExampleData] = {}

    for spec in EXAMPLE_SPECS:
        namespace = _load_module(spec.filename)
        description = namespace.get("__doc__") or ""

        func_eq: list[tuple[sp.Function, sp.Expr]] = []
        for func_name, expr_name in spec.func_eq:
            func_obj = namespace.get(func_name)
            expr_obj = namespace.get(expr_name)
            if not isinstance(func_obj, sp.Function):
                raise ValueError(f"Function {func_name} not found in {spec.filename}.")
            if not isinstance(expr_obj, sp.Expr):
                raise ValueError(
                    f"Expression {expr_name} not found in {spec.filename}."
                )
            func_eq.append((func_obj, expr_obj))

        indep_symbol = sp.symbols(spec.first_indep)
        first_func = func_eq[0][0]
        if first_func.args:
            indep_symbol = (
                indep_symbol if indep_symbol in first_func.args else first_func.args[0]
            )

        vars_str = ",".join(str(arg) for arg in first_func.args)
        funcs_str = ",".join(_unique(_func_name(func) for func, _ in func_eq))

        equations = []
        equations_latex = []
        for func, expr in func_eq:
            lhs = sp.Derivative(func, indep_symbol)
            eq = sp.Eq(lhs, expr)
            equations.append(sp.sstr(eq))
            equations_latex.append(sp.latex(eq))

        examples[spec.id] = ExampleData(
            id=spec.id,
            name=spec.name,
            description=description.strip(),
            diff_ord=spec.diff_ord,
            first_indep=spec.first_indep,
            func_eq=func_eq,
            vars=vars_str,
            funcs=funcs_str,
            equations=equations,
            equations_latex=equations_latex,
        )

    return examples


def list_examples() -> list[ExampleData]:
    return list(load_examples().values())


def get_example(example_id: str) -> ExampleData | None:
    return load_examples().get(example_id.lower())


def _print_examples() -> None:
    for example in list_examples():
        pprint(example)


if __name__ == "__main__":
    _print_examples()
