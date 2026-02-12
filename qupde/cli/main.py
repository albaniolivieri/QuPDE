from typing import Optional

import typer

from qupde.cli.constants import InputFormat, Printing, SearchAlg, SortFun
from qupde.cli.examples import EXAMPLES
from qupde.cli.errors import ParseError, QuadratizationError
from qupde.cli.service import QuadratizationRequest, run_quadratization


app = typer.Typer(
    help=(
        "Command-line interface for QuPDE.\n\n"
        "Pass your own PDE system on the command line (in SymPy or Mathematica "
        "syntax) and QuPDE will search for a quadratic transformation "
        "(quadratization) of the equations. You can control the search via "
        "options such as the differentiation order, search algorithm, and "
        "maximum derivative order of new variables.\n\n"
        "As a convenience, the CLI also ships with a small collection of built-in "
        "examples that you can explore with `qupde examples` and quadratize with "
        "`qupde run --example ...`. Run `qupde run --help` for the full list of "
        "options."
    )
)


def _emit_result(
    aux_vars,
    frac_vars,
    quad_sys,
    traversed,
    output: Optional[str],
) -> None:
    typer.echo(
        f"Quadratization completed with {len(aux_vars)} polynomial and {len(frac_vars)} rational auxiliary variables."
    )

    if traversed is not None:
        typer.echo(f"Nodes traversed: {traversed}")

    if output:
        summary_lines = [
            f"aux_vars: {aux_vars}",
            f"frac_vars: {[1 / var[1].as_expr() for var in frac_vars]}",
            f"quadratic_system_eqs: {quad_sys}",
        ]
        with open(output, "w", encoding="utf-8") as fh:
            fh.write("\n".join(summary_lines))


@app.command()
def examples() -> None:
    """List the PDE examples that ship with the CLI."""
    typer.echo("Available examples:")
    for name, example in EXAMPLES.items():
        typer.echo(f"- {name}: {example.description}")


@app.command()
def run(
    eq: list[str] = typer.Option(
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
    input_format: InputFormat = typer.Option(
        InputFormat.sympy,
        "--format",
        help="Parser format for equations.",
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
        help="Override the differentiation order. Defaults to 3 times the maximum derivative order of the equation.",
    ),
    sort_fun: SortFun = typer.Option(
        SortFun.by_fun,
        "--sort-fun",
        help="Sorting heuristic for proposed variables.",
        rich_help_panel="Search configuration",
        case_sensitive=False,
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
    search_alg: SearchAlg = typer.Option(
        SearchAlg.bnb,
        "--search-alg",
        help="Search algorithm: branch-and-bound (bnb) or incremental nearest neighbor (inn).",
        rich_help_panel="Search configuration",
        case_sensitive=False,
    ),
    printing: Printing = typer.Option(
        Printing.pprint,
        "--printing",
        help="Output mode: pretty print, LaTeX, or none.",
        rich_help_panel="Output",
        case_sensitive=False,
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
    """Quadratize a PDE system from a built-in example or user-provided equations."""
    user_equations = len(eq) > 0
    if user_equations and example:
        typer.echo("Use either --eq (with --vars/--funcs) or --example, not both.")
        raise typer.Exit(code=1)

    try:
        if user_equations:
            if not vars or not funcs:
                typer.echo("When using --eq, both --vars and --funcs are required.")
                raise typer.Exit(code=1)
            req = QuadratizationRequest(
                eq_strings=eq,
                indep_vars=vars,
                func_names=funcs,
                input_format=input_format,
                diff_ord=diff_ord,
                sort_fun=sort_fun,
                nvars_bound=nvars_bound,
                first_indep=first_indep,
                max_der_order=max_der_order,
                search_alg=search_alg,
                printing=printing,
                show_nodes=show_nodes,
            )
            result = run_quadratization(req)
            _emit_result(
                result.aux_vars,
                result.frac_vars,
                result.quad_sys,
                result.traversed,
                output,
            )
            return

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
        req = QuadratizationRequest(
            func_eq=func_eq,
            indep_symbol=example_cfg.first_indep if first_indep is None else None,
            first_indep=first_indep,
            diff_ord=diff_ord if diff_ord is not None else example_cfg.diff_ord,
            sort_fun=sort_fun,
            nvars_bound=nvars_bound,
            max_der_order=max_der_order,
            search_alg=search_alg,
            printing=printing,
            show_nodes=show_nodes,
        )
        result = run_quadratization(req)
        _emit_result(
            result.aux_vars,
            result.frac_vars,
            result.quad_sys,
            result.traversed,
            output,
        )

    except (ParseError, QuadratizationError) as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
