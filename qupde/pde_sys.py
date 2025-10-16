from typing import Optional, Callable
import time
import sympy as sp
from sympy.polys.rings import PolyElement
from sympy import Derivative as D
from .verify_quad import is_quadratization
from .utils import diff_dict, get_order, remove_vars, get_decompositions, sort_vars, get_diff_order
from .fraction_decomp import FractionDecomp
from .mon_heuristics import *


class PDESys:
    """
    A class used to represent a PDE system 

    ...

    Attributes
    ----------
    max_order : int
        an integer representing the max order of derivatives in the system
    first_indep : sp.Symbol
        a Symbol that represents the first independent variable
    sec_indep : sp.Symbol
        a Symbol that represents the second independent variable
    consts : list[sp.Symbol]
        a list of all the constants in the system
    order : int
        an integer representing the differentiation order of the quadratization 
    frac_decomps : FractionDecomp
        a FractionDecomp object that stores the fraction decomposition of the system
    poly_vars : PolyElement
        all system's symbol names as a polynomial ring
    pde_eq : list[tuple]
        a list of tuples containing the PDE equations in the form
        (<left hand side as symbol>, <right hand side as polynomial>)
    new_vars : dict
        a dictionary that stores the polynomial and rational auxiliary variables
    NS_list : list[tuple[sp.Symbol, PolyElement]]
        a list of tuples with the symbol and expression of the PDE that are not quadratic
    dic_t : dict
        a dictionary that stores time derivatives
    dic_x : dict
        a dictionary that stores spatial derivatives
    frac_der_t : list[PolyElement]
        a list of the derivatives with respect to t of the fraction variables
    quad_sys : list[tuple]
        a list with tuples that represent the new quadratic PDE

    Methods
    -------
    build_ring(pde_sys, n_diff, var_indep, new_vars=[])
        Builds the polynomial ring for all the expressions
    get_dics(func_eq, symb, eqs_pol, order, max_order)
        Builds dictionary that links a symbol with
        its symbol derivative
    set_new_vars(new_vars)
        Updates the attribute new_vars with new auxiliary variables
    set_NS_list(NS_list)
        Updates the `NS_list` attribute with the list of non-quadratic PDE expressions
    set_quad_sys(quad_sys)
        Updates the `quad_sys` attribute with the quadratic PDE system
    get_aux_vars()
        Returns the polynomial and rational auxiliary variables
    get_max_order()
        Returns the max derivative order of the system
    get_quad_sys()
        Returns the quadratic PDE system as a list of tuples
    differentiate_dict(named_new_vars)
        Builds two dictionaries with new variables derivatives in first
        and second variable
    try_make_quadratic()
        Returns the quadratization of the PDE using the new variables introduced
    prop_new_vars(sort_fun)
        Proposes new variables and sorts them according to sort_fun for the quadratization of the PDE system
    """

    def __init__(
        self,
        pde_sys: list[tuple[sp.Function, sp.Expr]],
        n_diff: int,
        vars_indep: sp.Symbol,
        new_vars: Optional[list[sp.Expr]] = None,
    ) -> None:
        """
        Parameters
        ----------
        pde_sys
            Tuples with the symbol and expression of PDE
        n_diff
            The  order of the quadratization 
        var_indep
            The symbol of the second independent variable
        new_vars : optional
            List of proposed new variables
        """
        self.max_order = get_order([expr for _, expr in pde_sys])

        self.first_indep, self.sec_indep = vars_indep
        self.consts = []
        self.order = n_diff

        poly_syms, eqs_pol, new_vars_pol, frac_decomps = self.build_ring(
            pde_sys, new_vars
        )

        self.frac_decomps = frac_decomps
        self.poly_vars = poly_syms
        self.pde_eq = eqs_pol
        self.new_vars = new_vars_pol
        self.NS_list = []

        dic_t, dic_x, frac_der_t = self.get_dics(pde_sys)

        self.dic_t = dic_t
        self.dic_x = dic_x
        self.frac_der_t = frac_der_t
        self.quad_sys = []

    def build_ring(
        self, func_eq: list[tuple[sp.Function, sp.Expr]], new_vars: list[sp.Expr]
    ) -> tuple[
        list[PolyElement], list[tuple[sp.Symbol, PolyElement]], dict, FractionDecomp
    ]:
        """Builds the polynomial ring for the PDE system.

        Parameters
        ----------
        func_eq
            Tuples with the symbol and expression of PDE
        new_vars
            List of proposed new variables

        Returns
        -------
        tuple[list[PolyElement], list[tuple[sp.Symbol, PolyElement]], dict, FractionDecomp]
            a tuple with the polynomial ring, the PDE equations as polynomials,
            the new variables as polynomials and the fraction decomposition
        """
        # der_subs is used for derivatives and functions substitution to sympy symbols
        der_subs = []
        for fun, _ in func_eq:
            der_subs += [
                (
                    D(fun, self.sec_indep, i),
                    sp.symbols(f"{fun.name}_{self.sec_indep.name}{i}"),
                )
                for i in range(self.max_order, 0, -1)
            ] + [(fun, sp.symbols(fun.name))]

        symbols_list = [symbol for _, expr in func_eq for symbol in expr.free_symbols]
        constants = [
            sp.symbols(x.name, constant=True)
            for x in set(
                filter(
                    lambda x: (x != self.first_indep) and (x != self.sec_indep),
                    symbols_list,
                )
            )
        ]
        self.consts = constants

        poly_vars = []

        for fun, _ in func_eq:
            poly_vars.append(sp.symbols(fun.name))
            poly_vars.extend(
                [
                    sp.symbols(f"{fun.name}_{self.sec_indep.name}{i}")
                    for i in range(1, self.max_order + self.order + 1)
                ]
            )

        # we substite derivatives symbols and if there is an expr with a rational function,
        # we convert it to the form p/q
        func_eq = [(lhs, sp.cancel(rhs.subs(der_subs))) for lhs, rhs in func_eq]

        frac_decomp = FractionDecomp(func_eq, poly_vars, constants)

        if frac_decomp:
            func_eq = frac_decomp.pde
            for i in range(len(frac_decomp.q_syms)):
                poly_vars.append(frac_decomp.q_syms[i])
                poly_vars.extend(
                    [
                        sp.symbols(f"q_{i}{self.sec_indep.name}{k}")
                        for k in range(1, self.max_order + self.order + 1)
                    ]
                )

        QQc = sp.FractionField(sp.QQ, constants)
        R, pol_sym = sp.xring(poly_vars, QQc)

        frac_decomp.rels_as_poly(R)

        pde_pol = [
            (sp.symbols(f"{fun.name}_{self.first_indep}"), R(sp.simplify(eq)))
            for fun, eq in func_eq
        ]

        vars_pol = {"frac_vars": [], "new_vars": []}

        for i in range(len(frac_decomp.rels)):
            frac_decomp.rels[i] = (frac_decomp.rels[i][0], R(frac_decomp.rels[i][1]))
            vars_pol["frac_vars"].append(frac_decomp.rels[i])

        # if the new variables are passed as sympy expressions
        # they are also added to the polynomial ring
        if new_vars:
            vars_pol["new_vars"] = [R(var) for var in new_vars]

        return pol_sym, pde_pol, vars_pol, frac_decomp

    def get_dics(
        self, func_eq: list[tuple[sp.Function, sp.Expr]]
    ) -> tuple[dict, dict, list[tuple[sp.Symbol, PolyElement]]]:
        """Builds the mapping for the variables derivatives with respect to the first and second independent variables

        Parameters
        ----------
        func_eq
            A list of tuples with the functions and equations of the PDE

        Returns
        -------
        tuple[dict, dict, list[tuple[sp.Symbol, PolyElement]]]
            a tuple with two dictionaries that map the variables derivatives and a list with the
            derivatives of the fraction decomposition variables
        """
        dic_x = {}
        dic_t = {}
        frac_ders = []

        der_order = self.max_order + self.order
        for i in range(len(func_eq)):
            for j in range(der_order):
                dic_x[self.poly_vars[j + (der_order + 1) * i]] = self.poly_vars[
                    j + (der_order + 1) * i + 1
                ]
                last = j + (der_order + 1) * i

        count = last + 2 # we skip the last derivative symbol of the last function
        rels = self.frac_decomps.rels
        for i in range(len(rels)):
            dic_x[self.poly_vars[count]] = self.frac_decomps.diff_frac(rels[i], dic_x)
            count += der_order + 1

        for k in range(len(func_eq)):
            for i in range(der_order):
                if i != 0:
                    dic_t[self.poly_vars[i + (der_order + 1) * k]] = diff_dict(
                        dic_t[self.poly_vars[i - 1 + (der_order + 1) * k]],
                        dic_x,
                        self.frac_decomps,
                    )
                else:
                    dic_t[self.poly_vars[(der_order + 1) * k]] = self.pde_eq[k][1]
        count = last + 2
        for rel in self.frac_decomps.rels:
            frac_der_t = self.frac_decomps.diff_frac(rel, dic_t) # we save the result to use it later
            frac_ders.append(
                (sp.symbols(rel[0].name + self.first_indep.name), frac_der_t)
            )
            dic_t[self.poly_vars[count]] = frac_der_t
            count += der_order + 1

        return dic_t, dic_x, frac_ders

    def set_new_vars(self, new_vars: list[PolyElement]) -> None:
        """Sets with a new value the new_vars attribute

        Parameters
        ----------
        new_vars
            List of proposed new variables

        Returns
        -------
        None
        """
        self.new_vars["new_vars"] = new_vars

    def set_NS_list(self, NS_list: list[tuple[sp.Symbol, PolyElement]]) -> None:
        """Sets with a new value the NS_list attribute

        Parameters
        ----------
        NS_list
            List of non-quadratic expressions in the PDE

        Returns
        -------
        None
        """
        self.NS_list = NS_list
        
    def set_quad_sys(self, quad_sys):
        """Sets with a new value the quad_sys attribute

        Parameters
        ----------
        quad_sys
            List of quadratic expressions in the PDE using the auxiliary variables in the system

        Returns
        -------
        None
        """
        self.quad_sys = quad_sys
    
    def get_aux_vars(self) -> list[PolyElement]:
        """Gets the fraction variables introduced in the system

        Returns
        -------
        list[PolyElement]
            a list of the fraction variables introduced
        """
        return self.new_vars["new_vars"], self.new_vars["frac_vars"]

    def get_max_order(self) -> int:
        """Gets the max derivative order of the system

        Returns
        -------
        int
            the max derivative order of the system
        """
        return self.max_order
    
    def get_quad_sys(self):
        """Gets the quadratic PDE system

        Returns
        -------
        list
            a list with tuples that represent each quation
        """
        return self.quad_sys

    def differentiate_dict(
        self, named_new_vars: tuple[sp.Symbol, PolyElement]
    ) -> tuple[
        list[tuple[sp.Symbol, PolyElement]], list[tuple[sp.Symbol, PolyElement]]
    ]:
        """Builds two lists that map the new variables with their respective
        derivatives in the first and second independent variable

        Parameters
        ----------
        named_new_vars
            List of proposed new variables with their respective symbol

        Returns
        -------
        tuple
            a tuple with two lists, the first with the derivatives with respect to the first independent variable
            and the second with the derivatives with respect to the second independent variable
        """
        deriv_t = []
        deriv_x = []

        for name, expr in named_new_vars:
            deriv_t.append(
                (
                    sp.symbols(f"{name}{self.first_indep}"),
                    diff_dict(expr, self.dic_t, frac_decomp=self.frac_decomps),
                )
            )

        for name, expr in named_new_vars:
            var_ord = self.order - get_diff_order(expr)
            if var_ord<0: var_ord = 0
            for i in range(1, var_ord + 1):
                deriv_x.append(
                    (
                        sp.symbols(f"{name}{self.sec_indep}{i}"),
                        diff_dict(
                            expr, self.dic_x, order=i, frac_decomp=self.frac_decomps
                        ),
                    )
                )

        for rel in self.frac_decomps.rels:
            var_ord = self.order - get_diff_order(rel[1])
            if var_ord<0: var_ord = 0
            for i in range(1, var_ord + 1):
                deriv_x.append(
                    (
                        sp.symbols(f"{rel[0].name}{self.sec_indep}{i}"),
                        self.frac_decomps.diff_frac(rel, self.dic_x, n_diff=i),
                    )
                )

        return deriv_t, deriv_x

    def try_make_quadratic(self) -> tuple[bool, list[PolyElement]]:
        """Verifies if a set of new variables is a quadratization for the PDE system

        Returns
        -------
        tuple[bool, list[PolyElement]]
            a tuple with a bool and the transformation if it is a quadratization. If not,
            returns the residual expressions that are nonquadratic.
        """
        new_vars_named = [
            (sp.symbols(f"w_{i}"), pol)
            for i, pol in enumerate(self.new_vars["new_vars"])
        ]
        new_vars_t, new_vars_x = self.differentiate_dict(new_vars_named)
        deriv_t = new_vars_t + self.frac_der_t + self.pde_eq
        poly_vars = list(filter(lambda x: str(x)[0] != "q", self.poly_vars))
        V = (
            [(1, self.poly_vars[0].ring(1))]
            + [(sp.symbols(f"{sym}"), sym) for sym in poly_vars]
            + [(q, self.poly_vars[0].ring(q)) for q, _ in self.frac_decomps.rels]
            + new_vars_named
            + new_vars_x
        )

        result = is_quadratization(V, deriv_t, self.frac_decomps)
        if not result[0]:
            self.NS_list = result[1]
        return result

    def prop_new_vars(
        self,
        sort_fun: Callable,
    ) -> list[PolyElement]:
        """Proposes new variables for the quadratization of a PDE

        Parameters
        ----------
        sort_fun
            Function to sort the variables

        Returns
        -------
        list[PolyElement]
            the proposed new variables
        """
        list_vars = []

        min_monom = (0,)*len(self.NS_list[0][1].leading_expv())
        
        for ns_pol in self.NS_list:
            monoms = [m for m in ns_pol[1].itermonoms() if sum(m) > 2]
        if monoms:
            min_monom = min(monoms, key=lambda m: sum(m))
            
        list_vars += get_decompositions(min_monom)
        
        list_vars = list(
            map(
                lambda x: (
                    self.NS_list[0][1].ring({x[0]: 1}),
                    self.NS_list[0][1].ring({x[1]: 1}),
                ),
                list_vars,
            )
        )

        list_vars = remove_vars(list_vars, self.new_vars["new_vars"])
        
        new_vars = list_vars[:]

        for i in range(len(list_vars)):
            if not list_vars[i]:
                new_vars.remove(list_vars[i])

        sorted_vars = sort_vars(new_vars, sort_fun)

        return sorted_vars
