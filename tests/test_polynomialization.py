import math
import pytest
from sympy import symbols, simplify, expand, nsimplify, Function, exp, sin, cos, Pow, Symbol, Integer
from sympy import Derivative as D

from qupde.polynomialization import polynomialize 
from qupde.utils import get_sys_order
from qupde.quadratization import check_quadratization
from qupde.pde_sys import PDESys


class PDECase:
    def __init__(self, func_eq) -> None:
        self.func_eq = func_eq


class PolynomializationHelpers:
    def rewrite_new_vars(self, aux_vars):
        """
        Transform the new variables to sympy expressions.
        """
        refac = []
        new_vars = []
        for aux_var in aux_vars: 
            refac.append(aux_var) 
            aux_var_re = aux_var[0].subs(refac)
            new_vars.append((aux_var_re, aux_var[1]))
        return new_vars

    def differentiate_t(self, funcs_eqs, aux_vars):
        """
        Differentiate the new variables with respect to t.
        """
        deriv_t = []
        refac = [(D(eq[0], symbols("t")), eq[1]) for eq in funcs_eqs]
        for i in range(len(aux_vars)):
            pt = D(aux_vars[i][0], symbols("t")).doit().subs(refac)
            deriv_t.append((symbols(f"{aux_vars[i][0]}_t"), pt.doit()))
        return deriv_t

    def rewrite_expr(self, test_case, new_vars, frac_vars):
        """
        Rewrite the expressions with the new variables definitions.
        """
        refac = []
        for fun, _ in test_case.func_eq:
            refac += [(symbols(fun.name), fun)]

        quad_prop = [expr.subs(refac) for expr in new_vars]
        frac_vars = [(q, 1 / expr.subs(refac)) for q, expr in frac_vars]
        new_vars = [expr.subs(frac_vars) for expr in quad_prop]
        return new_vars, frac_vars, refac

    def convert_to_rational(self, expr):
        """
        Convert an expression coefficients to rational numbers.
        """
        result = nsimplify(expr, rational=True, tolerance=0.0001)
        return result

@pytest.fixture(scope="module")
def test_data():
    t, x = symbols("t x")
    u = Function("u")(t, x)
    v = Function("v")(t, x)
    z = Function("z")(t, x)
    omega = symbols("omega", constant=True)
    alpha = symbols("alpha", constant=True)

    helpers = PolynomializationHelpers()

    test_cases_pol = [
        # u_t = (1 - u) / (1 + v^\omega) - u
        # v_t = (1 - v) * u - v
        PDECase(
            [
                (u, (1 - u) / (1 + v**omega) - u),
                (v, (1 - v) * u - v),
            ]
        ),
        # u_t = k_1 * (s_1 - u) * (K_1**n_1 / (K_1**n_1 + z**n_1)) - k_2 * u
        # v_t = k_3 * (s_2 - v) * u * (1 + (alpha * z**n_2) / (K_2**n_2 + z**n_2)) - k_4 * v
        # z_t = k_5 * (s_3 - z) * v - k_6 * z
        PDECase(
            [
                (u, (1 - u) * omega / (1 + z**omega) - u),
                (v, (1 - v) * u * (1 + 2 * z**alpha) / (1 + z**alpha) - v),
                (z, (1 - z) * v - alpha * z),
            ]
        ),

        # u_t = u_xx - sin(u)
        PDECase([(u, D(u, x, 2) - sin(u))]),  
        # PDECase()
    ]
    
    test_cases_rat = [
        PDECase([(u, 1 / (1 + exp(omega*u - 1)))]),  
        # u_t = k_1 * (s_1 - u) * (K_1**n_1 / (K_1**n_1 + z**n_1)) - k_2 * u
        # v_t = k_3 * (s_2 - v) * u * (1 + (alpha * z**n_2) / (K_2**n_2 + z**n_2)) - k_4 * v
        # z_t = k_5 * (s_3 - z) * v - k_6 * z
        PDECase(
            [
                (u, (1 - u) * omega / (1 + z**omega) - u),
                (v, (1 - v) * u * (1 + 2 * z**alpha) / (1 + z**alpha) - v),
                (z, (1 - z) * v - alpha * z),
            ]
        )
    ]
    
    return {
        "t": t,
        "x": x,
        "helpers": helpers,
        "poly": test_cases_pol,
        "rat": test_cases_rat
    }

def polynomialization_test(test_cases, data, is_rat):
    """
    Main method to test the polynomialization algorithm.
    """
    helpers = data["helpers"]
    for test in test_cases:
        print("\nTest case: ")
        [print(f"Derivative({eq[0]}, t)", "=", eq[1]) for eq in test.func_eq]
        poly_syst, aux_vars = polynomialize(
            test.func_eq,
            is_rat=is_rat
        )
        assert (len(aux_vars) != 0), (
            f"Polynomialization not found for {test.func_eq}"
        )
        print(f"Polynomialization: {aux_vars}")

        poly_vars = helpers.rewrite_new_vars(aux_vars)
        exprs_orig = test.func_eq
        exprs_orig += helpers.differentiate_t(test.func_eq, poly_vars) 
        
        results = [(eq[0], eq[1].subs(aux_vars)) for eq in poly_syst]
        
        for i in range(len(exprs_orig)):
            result = results[i][1].evalf()
            assert (
                simplify(
                    helpers.convert_to_rational(exprs_orig[i][1])
                    - helpers.convert_to_rational(result)
                )
                == 0
            ), (
                f"Test failed: expressions are not equal for {exprs_orig[i]} \n"
                + f"Equation: {results[i][1]} \n"
                + f"Original expression: {expand(helpers.convert_to_rational(exprs_orig[i][1]))} \n"
                + f"Algorithm expression: {helpers.convert_to_rational(result)} \n"
                + f"Substraction: {simplify(helpers.convert_to_rational(exprs_orig[i][1]) - helpers.convert_to_rational(result))}"
            )


def test_to_polynomial(test_data):
    """
    Test the transformation to polynomials.
    """
    polynomialization_test(test_data["poly"], test_data, is_rat=False)


def test_to_rational(test_data):
    """
    Test the transformation to rational functions.
    """
    polynomialization_test(test_data["rat"], test_data, is_rat=True)

