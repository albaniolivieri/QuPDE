import sympy as sp
from sympy.parsing.mathematica import parse_mathematica
from sympy.parsing.sympy_parser import parse_expr

from qupde.cli.constants import InputFormat
from qupde.cli.errors import ParseError


def split_csv(raw: str, label: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ParseError(f"{label} cannot be empty.")
    return values


def _normalize_symbols(expr: sp.Expr, symbol_map: dict[str, sp.Symbol]) -> sp.Expr:
    replacements = {
        sym: symbol_map[sym.name] for sym in expr.free_symbols if sym.name in symbol_map
    }
    if not replacements:
        return expr
    return expr.xreplace(replacements)


def _coerce_derivatives(expr: sp.Expr) -> sp.Expr:
    return expr.replace(
        lambda e: getattr(e, "func", None) and e.func.__name__ == "D",
        lambda e: sp.Derivative(*e.args),
    )


def _to_derivative(expr: sp.Expr) -> sp.Expr:
    if isinstance(expr, sp.Derivative):
        return expr
    if getattr(expr, "func", None) and expr.func.__name__ == "D":
        return sp.Derivative(*expr.args)
    return expr


def parse_user_equations(
    eq_strings: list[str],
    indep_vars: str,
    func_names: str,
    input_format: InputFormat,
) -> tuple[list[tuple[sp.Function, sp.Expr]], sp.Symbol]:
    indep_list = split_csv(indep_vars, "vars")
    func_list = split_csv(func_names, "funcs")

    if len(indep_list) != 2:
        raise ParseError("Exactly two independent variables are required.")

    first_indep, second_indep = (sp.symbols(name) for name in indep_list)

    func_objs = {name: sp.Function(name) for name in func_list}
    func_applied = {
        name: fun(first_indep, second_indep) for name, fun in func_objs.items()
    }

    parser_locals = {
        indep_list[0]: first_indep,
        indep_list[1]: second_indep,
        "Derivative": sp.Derivative,
        "D": sp.Derivative,
    }
    parser_locals.update(func_objs)

    symbol_map = {sym.name: sym for sym in (first_indep, second_indep)}

    func_eq: list[tuple[sp.Function, sp.Expr]] = []
    for eq_str in eq_strings:
        if input_format == InputFormat.sympy:
            if "=" not in eq_str:
                raise ParseError("SymPy format equations must contain '='.")
            lhs_str, rhs_str = eq_str.split("=", 1)
            lhs = parse_expr(lhs_str.strip(), local_dict=parser_locals, evaluate=False)
            rhs = parse_expr(rhs_str.strip(), local_dict=parser_locals, evaluate=False)
            lhs = _normalize_symbols(lhs, symbol_map)
            rhs = _normalize_symbols(rhs, symbol_map)
        else:
            if "==" not in eq_str:
                raise ParseError("Mathematica format equations must contain '=='.")
            lhs_str, rhs_str = eq_str.split("==", 1)
            lhs = parse_mathematica(lhs_str.strip())
            rhs = parse_mathematica(rhs_str.strip())
            lhs = _normalize_symbols(lhs, symbol_map)
            rhs = _normalize_symbols(rhs, symbol_map)

        lhs = _coerce_derivatives(lhs)
        rhs = _coerce_derivatives(rhs)
        lhs = _to_derivative(lhs)

        if not isinstance(lhs, sp.Derivative):
            raise ParseError(
                "Left-hand side must be a derivative in the first independent variable, e.g. Derivative(u(t,x), t)."
            )

        if not lhs.variables or lhs.variables[0] != first_indep:
            raise ParseError(
                f"Left-hand side must differentiate with respect to the first variable '{first_indep}'."
            )

        base_func = lhs.expr
        if not base_func.is_Function:
            raise ParseError(
                "Left-hand side must be a derivative of a function of the provided variables."
            )

        func_name = base_func.func.__name__
        if func_name not in func_applied:
            raise ParseError(f"Function '{func_name}' not declared in functions list.")

        if len(base_func.args) != 2 or base_func.args != (first_indep, second_indep):
            raise ParseError(
                f"Function '{func_name}' must be called with exactly the independent variables ({first_indep}, {second_indep})."
            )

        func_eq.append((func_applied[func_name], rhs))
        # print('func_eq', func_eq)

    return func_eq, first_indep
