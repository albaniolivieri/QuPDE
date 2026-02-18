from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import sympy as sp

from qupde.quadratization import quadratize
from qupde.cli.constants import InputFormat, Printing, SearchAlg, SortFun
from qupde.cli.errors import ParseError, QuadratizationError
from qupde.cli.parsing import parse_user_equations


@dataclass
class QuadratizationRequest:
    eq_strings: list[str] = field(default_factory=list)
    indep_vars: Optional[str] = None
    func_names: Optional[str] = None
    func_eq: Optional[List[Tuple[sp.Function, sp.Expr]]] = None
    indep_symbol: Optional[sp.Symbol] = None
    input_format: InputFormat = InputFormat.sympy
    diff_ord: Optional[int] = None
    sort_fun: SortFun = SortFun.by_fun
    nvars_bound: int = 10
    first_indep: Optional[str] = None
    max_der_order: Optional[int] = None
    search_alg: SearchAlg = SearchAlg.bnb
    printing: Printing = Printing.none
    show_nodes: bool = False


@dataclass
class QuadratizationResult:
    aux_vars: list[sp.Expr]
    frac_vars: list[sp.Expr]
    quad_sys: list[sp.Expr]
    traversed: Optional[int]


def run_quadratization(req: QuadratizationRequest) -> QuadratizationResult:
    func_eq: List[Tuple[sp.Function, sp.Expr]]
    indep_symbol: sp.Symbol

    if req.func_eq is not None:
        func_eq = req.func_eq
        indep_symbol = (
            req.indep_symbol
            if req.indep_symbol is not None
            else (sp.symbols(req.first_indep) if req.first_indep else sp.symbols("t"))
        )
    else:
        if req.indep_vars is None or req.func_names is None:
            raise ParseError("Both --vars and --funcs are required when using --eq.")
        func_eq, indep_symbol = parse_user_equations(
            eq_strings=req.eq_strings,
            indep_vars=req.indep_vars,
            func_names=req.func_names,
            input_format=req.input_format,
        )
        if req.first_indep:
            indep_symbol = sp.symbols(req.first_indep)

    selected_diff_ord = req.diff_ord
    selected_max_der = req.max_der_order

    printing_arg = "" if req.printing == Printing.none else req.printing.value

    result = quadratize(
        func_eq,
        diff_ord=selected_diff_ord,
        sort_fun=req.sort_fun.value,
        nvars_bound=req.nvars_bound,
        first_indep=(
            sp.symbols(req.first_indep) if req.first_indep is not None else indep_symbol
        ),
        max_der_order=selected_max_der,
        search_alg=req.search_alg.value,
        printing=printing_arg,
        show_nodes=req.show_nodes,
    )

    if result == []:
        raise QuadratizationError("Quadratization not found.")

    poly_syst, traversed = (result, None)
    if req.show_nodes and isinstance(result, tuple):
        poly_syst, traversed = result

    aux_vars, frac_vars = poly_syst.get_aux_vars()
    quad_sys = poly_syst.get_quad_sys()

    return QuadratizationResult(
        aux_vars=aux_vars,
        frac_vars=frac_vars,
        quad_sys=quad_sys,
        traversed=traversed,
    )
