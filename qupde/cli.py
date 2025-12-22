from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import sympy as sp
from sympy.parsing.mathematica import parse_mathematica
from sympy.parsing.sympy_parser import parse_expr
import typer

from .quadratization import quadratize


app = typer.Typer(help="Command-line interface for running QuPDE quadratizations.")


ExampleBuilder = Callable[[], List[Tuple[sp.Function, sp.Expr]]]
SORT_FUNS = {"by_fun", "by_degree_order", "by_order_degree"}
SEARCH_ALGS = {"bnb", "inn"}
PRINTING_OPTIONS = {"pprint", "latex", "none"}
INPUT_FORMATS = {"sympy", "mathematica"}


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


def _validate_choice(value: str, options: set[str], name: str) -> str:
    """Validate and normalize CLI choices."""
    value_lower = value.lower()
    if value_lower not in options:
        raise typer.BadParameter(
            f"Invalid {name} '{value}'. Valid options: {', '.join(sorted(options))}."
        )
    return value_lower


def _split_csv(raw: str, label: str) -> List[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise typer.BadParameter(f"{label} cannot be empty.")
    return values


def _normalize_symbols(expr: sp.Expr, symbol_map: Dict[str, sp.Symbol]) -> sp.Expr:
    """Replace symbols in expr with shared instances based on name."""
    replacements = {
        sym: symbol_map[sym.name] for sym in expr.free_symbols if sym.name in symbol_map
    }
    if not replacements:
        return expr
    return expr.xreplace(replacements)


def _coerce_derivatives(expr: sp.Expr) -> sp.Expr:
    """Replace Mathematica-style D(...) calls with SymPy Derivative objects."""
    return expr.replace(
        lambda e: getattr(e, "func", None) and e.func.__name__ == "D",
        lambda e: sp.Derivative(*e.args),
    )


def _to_derivative(expr: sp.Expr) -> sp.Expr:
    """Coerce Mathematica parser derivatives (D) into SymPy Derivative."""
    if isinstance(expr, sp.Derivative):
        return expr
    if getattr(expr, "func", None) and expr.func.__name__ == "D":
        return sp.Derivative(*expr.args)
    return expr


def _parse_user_equations(
    eq_strings: List[str],
    indep_vars: str,
    func_names: str,
    input_format: str,
) -> Tuple[List[Tuple[sp.Function, sp.Expr]], sp.Symbol]:
    """Parse user-provided equations into the format expected by quadratize."""
    indep_list = _split_csv(indep_vars, "vars")
    func_list = _split_csv(func_names, "funcs")

    if len(indep_list) != 2:
        raise typer.BadParameter("Exactly two independent variables are required.")

    first_indep, second_indep = (sp.symbols(name) for name in indep_list)

    func_objs = {name: sp.Function(name) for name in func_list}
    func_applied = {name: fun(first_indep, second_indep) for name, fun in func_objs.items()}

    parser_locals = {
        indep_list[0]: first_indep,
        indep_list[1]: second_indep,
        "Derivative": sp.Derivative,
        "D": sp.Derivative,
    }
    parser_locals.update(func_objs)

    symbol_map = {sym.name: sym for sym in (first_indep, second_indep)}

    func_eq: List[Tuple[sp.Function, sp.Expr]] = []
    for eq_str in eq_strings:
        if input_format == "sympy":
            if "=" not in eq_str:
                raise typer.BadParameter("SymPy format equations must contain '='.")
            lhs_str, rhs_str = eq_str.split("=", 1)
            lhs = parse_expr(lhs_str.strip(), local_dict=parser_locals, evaluate=False)
            rhs = parse_expr(rhs_str.strip(), local_dict=parser_locals, evaluate=False)
            lhs = _normalize_symbols(lhs, symbol_map)
            rhs = _normalize_symbols(rhs, symbol_map)
        else:
            if "==" not in eq_str:
                raise typer.BadParameter("Mathematica format equations must contain '=='.")
            lhs_str, rhs_str = eq_str.split("==", 1)
            lhs = parse_mathematica(lhs_str.strip())
            rhs = parse_mathematica(rhs_str.strip())
            lhs = _normalize_symbols(lhs, symbol_map)
            rhs = _normalize_symbols(rhs, symbol_map)

        lhs = _coerce_derivatives(lhs)
        rhs = _coerce_derivatives(rhs)
        lhs = _to_derivative(lhs)

        if not isinstance(lhs, sp.Derivative):
            raise typer.BadParameter(
                "Left-hand side must be a derivative in the first independent variable, e.g. Derivative(u(t,x), t)."
            )

        if not lhs.variables or lhs.variables[0] != first_indep:
            raise typer.BadParameter(
                f"Left-hand side must differentiate with respect to the first variable '{first_indep}'."
            )

        base_func = lhs.expr
        if not base_func.is_Function:
            raise typer.BadParameter("Left-hand side must be a derivative of a function of the provided variables.")

        func_name = base_func.func.__name__
        if func_name not in func_applied:
            raise typer.BadParameter(f"Function '{func_name}' not declared in --funcs.")

        if len(base_func.args) != 2 or base_func.args != (first_indep, second_indep):
            raise typer.BadParameter(
                f"Function '{func_name}' must be called with exactly the independent variables ({first_indep}, {second_indep})."
            )

        func_eq.append((func_applied[func_name], rhs))

    return func_eq, first_indep


@app.command()
def examples() -> None:
    """List the PDE examples that ship with the CLI."""
    typer.echo("Available examples:")
    for name, example in EXAMPLES.items():
        typer.echo(f"- {name}: {example.description}")


@app.command()
def run(
    eq: List[str] = typer.Option(
        [],
        "--eq",
        help="Equation(s) of the system. Use multiple --eq flags.",
    ),
    vars: Optional[str] = typer.Option(  # type: ignore
        None,
        "--vars",
        help='Independent variables, comma-separated (e.g. "t,x"). Required with --eq.',
    ),
    funcs: Optional[str] = typer.Option(
        None,
        "--funcs",
        help='Functions of the system, comma-separated (e.g. "u" or "u,v"). Required with --eq.',
    ),
    input_format: str = typer.Option(
        "sympy",
        "--format",
        help="Parser format for equations.",
        callback=lambda v: _validate_choice(v, INPUT_FORMATS, "format"),
    ),
    example: Optional[str] = typer.Option(
        None,
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
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to write a short summary of the quadratization.",
    ),
) -> None:
    """Quadratize either built-in examples or user-provided equations."""
    user_equations = len(eq) > 0
    if user_equations and example:
        typer.echo("Use either --eq (with --vars/--funcs) or --example, not both.")
        raise typer.Exit(code=1)

    if user_equations:
        if not vars or not funcs:
            typer.echo("When using --eq, both --vars and --funcs are required.")
            raise typer.Exit(code=1)
        func_eq, indep_symbol = _parse_user_equations(
            eq_strings=eq,
            indep_vars=vars,
            func_names=funcs,
            input_format=input_format,
        )
        selected_diff_ord = diff_ord if diff_ord is not None else 2
        selected_max_der_order = max_der_order if max_der_order is not None else 2
    else:
        if example is None:
            typer.echo("Provide either --eq (with --vars/--funcs) or --example.")
            raise typer.Exit(code=1)
        example_key = example.lower()
        if example_key not in EXAMPLES:
            typer.echo(
                f"Unknown example '{example}'. Run `qupde examples` to see options."
            )
            raise typer.Exit(code=1)

        example_cfg = EXAMPLES[example_key]
        func_eq = example_cfg.builder()
        selected_diff_ord = diff_ord if diff_ord is not None else example_cfg.diff_ord
        selected_max_der_order = max_der_order
        indep_symbol = (
            sp.symbols(first_indep)
            if first_indep is not None
            else example_cfg.first_indep
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
        max_der_order=selected_max_der_order,
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

    if output:
        summary_lines = [
            f"aux_vars: {len(aux_vars)}",
            f"frac_vars: {len(frac_vars)}",
            f"quadratic_system_eqs: {len(quad_sys)}",
        ]
        with open(output, "w", encoding="utf-8") as fh:
            fh.write("\n".join(summary_lines))


if __name__ == "__main__":
    app()
