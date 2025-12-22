from dataclasses import dataclass
from typing import List, Optional

import sympy as sp

from qupde.quadratization import quadratize
from qupde.cli.constants import InputFormat, Printing, SearchAlg, SortFun
from qupde.cli.errors import QuadratizationError
from qupde.cli.parsing import parse_user_equations


@dataclass
class QuadratizationRequest:
    eq_strings: List[str]
    indep_vars: str
    func_names: str
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
    aux_vars: List[sp.Expr]
    frac_vars: List[sp.Expr]
    quad_sys: List[sp.Expr]
    traversed: Optional[int]


def run_quadratization(req: QuadratizationRequest) -> QuadratizationResult:
    func_eq, indep_symbol = parse_user_equations(
        eq_strings=req.eq_strings,
        indep_vars=req.indep_vars,
        func_names=req.func_names,
        input_format=req.input_format,
    )

    selected_diff_ord = req.diff_ord if req.diff_ord is not None else 2
    selected_max_der = req.max_der_order if req.max_der_order is not None else 2

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
