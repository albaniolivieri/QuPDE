from typing import List, Optional

import sympy as sp
import typer

from .quadratization import quadratize
from .cli_examples import EXAMPLES
from .cli_parsing import (
    INPUT_FORMATS,
    PRINTING_OPTIONS,
    SEARCH_ALGS,
    SORT_FUNS,
    parse_user_equations,
    validate_choice,
)


app = typer.Typer(help="Command-line interface for running QuPDE quadratizations.")


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
        callback=lambda v: validate_choice(v, INPUT_FORMATS, "format"),
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
        callback=lambda v: validate_choice(v, SORT_FUNS, "sort_fun"),
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
        callback=lambda v: validate_choice(v, SEARCH_ALGS, "search_alg"),
    ),
    printing: str = typer.Option(
        "pprint",
        "--printing",
        help="Output mode: pretty print, LaTeX, or none.",
        rich_help_panel="Output",
        callback=lambda v: validate_choice(v, PRINTING_OPTIONS, "printing"),
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
        func_eq, indep_symbol = parse_user_equations(
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
