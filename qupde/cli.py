import dataclasses
from typing import Callable, Dict, List, Optional, Tuple

import sympy as sp
import typer

from .quadratization import quadratize


app = typer.Typer(help="Command-line interface for running QuPDE quadratizations.")


ExampleBuilder = Callable[[], List[Tuple[sp.Function, sp.Expr]]]
SORT_FUNS = {"by_fun", "by_degree_order", "by_order_degree"}
SEARCH_ALGS = {"bnb", "inn"}
PRINTING_OPTIONS = {"pprint", "latex", "none"}


@dataclasses.dataclass
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


def _validate_choice(value: str, options: set[str], name: str) -> str:
    """Validate and normalize CLI choices."""
    value_lower = value.lower()
    if value_lower not in options:
        raise typer.BadParameter(
            f"Invalid {name} '{value}'. Valid options: {', '.join(sorted(options))}."
        )
    return value_lower


@app.command()
def examples() -> None:
    """List the PDE examples that ship with the CLI."""
    typer.echo("Available examples:")
    for name, example in EXAMPLES.items():
        typer.echo(f"- {name}: {example.description}")


@app.command()
def run(
    example: str = typer.Option(
        ...,
        "--example",
        "-e",
        help="Name of the PDE example to quadratize. See `qupde examples`.",
    ),
    diff_ord: Optional[int] = typer.Option(
        None,
        "--diff-ord",
        help="Override the differentiation order. Defaults to the example's suggested value.",
    ),
    sort_fun: str = typer.Option(
        "by_fun",
        "--sort-fun",
        help="Sorting heuristic for proposed variables.",
        rich_help_panel="Search configuration",
        callback=lambda v: _validate_choice(v, SORT_FUNS, "sort_fun"),
    ),
    nvars_bound: int = typer.Option(
        10,
        "--nvars-bound",
        help="Maximum number of auxiliary variables allowed during search.",
        rich_help_panel="Search configuration",
    ),
    first_indep: Optional[str] = typer.Option(
        None,
        "--first-indep",
        help="Symbol to use as the first independent variable (time).",
        rich_help_panel="Variables",
    ),
    max_der_order: Optional[int] = typer.Option(
        None,
        "--max-der-order",
        help="Maximum derivative order allowed in new variables.",
        rich_help_panel="Search configuration",
    ),
    search_alg: str = typer.Option(
        "bnb",
        "--search-alg",
        help="Search algorithm: branch-and-bound (bnb) or incremental nearest neighbor (inn).",
        rich_help_panel="Search configuration",
        callback=lambda v: _validate_choice(v, SEARCH_ALGS, "search_alg"),
    ),
    printing: str = typer.Option(
        "pprint",
        "--printing",
        help="Output mode: pretty print, LaTeX, or none.",
        rich_help_panel="Output",
        callback=lambda v: _validate_choice(v, PRINTING_OPTIONS, "printing"),
    ),
    show_nodes: bool = typer.Option(
        False,
        "--show-nodes",
        help="Display how many nodes were traversed by the search algorithm.",
        rich_help_panel="Search configuration",
    ),
) -> None:
    """Quadratize one of the built-in PDE examples."""
    example_key = example.lower()
    if example_key not in EXAMPLES:
        typer.echo(f"Unknown example '{example}'. Run `qupde examples` to see options.")
        raise typer.Exit(code=1)

    example_cfg = EXAMPLES[example_key]
    func_eq = example_cfg.builder()
    selected_diff_ord = diff_ord if diff_ord is not None else example_cfg.diff_ord
    indep_symbol = (
        sp.symbols(first_indep) if first_indep is not None else example_cfg.first_indep
    )

    printing_arg = "" if printing == "none" else printing
    quad_sort = sort_fun
    search = search_alg

    result = quadratize(
        func_eq,
        diff_ord=selected_diff_ord,
        sort_fun=quad_sort,
        nvars_bound=nvars_bound,
        first_indep=indep_symbol,
        max_der_order=max_der_order,
        search_alg=search,
        printing=printing_arg,
        show_nodes=show_nodes,
    )

    if result == []:
        typer.echo("Quadratization not found.")
        raise typer.Exit(code=1)

    poly_syst, traversed = (result, None)
    if show_nodes and isinstance(result, tuple):
        poly_syst, traversed = result

    aux_vars, frac_vars = poly_syst.get_aux_vars()
    quad_sys = poly_syst.get_quad_sys()

    typer.echo(
        f"Quadratization completed with {len(aux_vars)} polynomial and {len(frac_vars)} rational auxiliary variables."
    )
    typer.echo(f"Quadratic system size: {len(quad_sys)} equation(s).")

    if traversed is not None:
        typer.echo(f"Nodes traversed: {traversed}")


if __name__ == "__main__":
    app()
