import math
import pytest
from sympy import symbols, simplify, expand, nsimplify, Function
from sympy import Derivative as D

from qupde import quadratize
from qupde.utils import get_sys_order
from qupde.quadratization import check_quadratization
from qupde.pde_sys import PDESys


class PDECase:
    def __init__(self, func_eq, n_diff, max_der_order=None, nvars_bound=10) -> None:
        self.func_eq = func_eq
        self.n_diff = n_diff
        self.max_der_order = max_der_order
        self.n_vars_bound = nvars_bound


class QuadratizationHelpers:
    def transform_new_vars(self, new_vars, frac_vars):
        """
        Transform the new variables to sympy expressions.
        """
        quad_prop = list(map(lambda x: x.as_expr(), new_vars))
        frac_vars = list(map(lambda x: (x[0], x[1].as_expr()), frac_vars))
        return quad_prop, frac_vars

    def differentiate_t(self, funcs_eqs, new_vars):
        """
        Differentiate the new variables with respect to t.
        """
        deriv_t = []
        refac = [(D(deriv[0], symbols("t")), deriv[1]) for deriv in funcs_eqs]
        for i in range(len(new_vars)):
            wt = D(new_vars[i][1], symbols("t")).doit().subs(refac)
            deriv_t.append((symbols(f"{new_vars[i][0]}_t"), wt.doit()))
        return deriv_t

    def differentiate_x(self, new_vars, n_diff, x):
        """
        Differentiate the new variables with respect to x according to the differential order.
        """
        vars_prop, frac_vars = new_vars
        quad_vars = []
        for i in range(len(vars_prop)):
            var_ord = n_diff - get_sys_order([vars_prop[i]])
            if var_ord < 0:
                var_ord = 0
            quad_vars.extend(
                [
                    (symbols(f"w_{i}{x}{j}"), D(vars_prop[i], x, j).doit())
                    for j in range(1, var_ord + 1)
                ]
                + [(symbols(f"w_{i}"), vars_prop[i])]
            )
        for i in range(len(frac_vars)):
            var_ord = n_diff - get_sys_order([frac_vars[i]])
            if var_ord < 0:
                var_ord = 0
            quad_vars.extend(
                [
                    (symbols(f"q_{i}{x}{j}"), D(frac_vars[i], x, j).doit())
                    for j in range(1, var_ord + 1)
                ]
            )
        return quad_vars

    def rewrite_expr(self, test_case, new_vars, frac_vars, x):
        """
        Rewrite the expressions with the new variables definitions.
        """
        max_order = get_sys_order(list(zip(*test_case.func_eq))[1])
        refac = []
        for fun, _ in test_case.func_eq:
            refac += [
                (symbols(f"{fun.name}_{x}{i}"), D(fun, x, i))
                for i in range(test_case.n_diff + max_order + 1, 0, -1)
            ] + [(symbols(fun.name), fun)]

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

    def construct_quadratic_PDE(self, new_vars, frac_vars, test_case, refac, x):
        """
        Returns the new variables expressions and derivatives, and its differential equations
        to construct the quadratic PDE transformation.
        """
        var_dic = [(symbols(f"w_{i}"), new_vars[i]) for i in range(len(new_vars))]
        total_vars = (new_vars, [rel for _, rel in frac_vars])
        quad_vars = self.differentiate_x(total_vars, test_case.n_diff, x)
        deriv_t = self.differentiate_t(
            test_case.func_eq,
            [(var, expr.subs(frac_vars)) for var, expr in var_dic] + frac_vars,
        ) + [(symbols(eqs[0].name + "_t"), eqs[1]) for eqs in test_case.func_eq]
        refac += quad_vars + frac_vars
        exprs_orig = [expr for _, expr in deriv_t]
        return refac, exprs_orig


@pytest.fixture(scope="module")
def test_data():
    t, x = symbols("t x")
    u = Function("u")(t, x)
    v = Function("v")(t, x)
    omega = symbols("omega", constant=True)

    helpers = QuadratizationHelpers()

    test_cases_quad = [
        # u_t = - u_x * u**3 - 1/3 * u_x * u**2
        # v_t = v_x * u - 2 * v_x
        PDECase(
            [
                (
                    u,
                    -D(v, x) * u**3 - 1 / 3 * D(v, x) * u**2,
                ),
                (v, D(v, x) * u - 2 * D(v, x)),
            ],
            3,
        ),
        # u_t = u**3 * u_xxx
        PDECase([(u, u**3 * D(u, x, 3))], 9),
        # u_t = u_x**3 + u**3
        PDECase([(u, D(u, x) ** 3 + u**3)], 3),
        # u_t = u_x**4
        PDECase([(u, D(u, x) ** 4)], 3),
        # u_t = u_x**3 * u
        PDECase([(u, D(u, x) ** 3 * u)], 3),
        # u_t = u_x**3
        PDECase([(u, D(u, x) ** 3)], 3),
    ]

    test_cases_rat = [
        # u_t = 3.4 * u_x * v_x
        # v_t = v_x / v - pi * v_x
        PDECase(
            [
                (u, 3.4 * D(u, x) * D(v, x)),
                (
                    v,
                    D(v, x) / v - round((math.pi), 5) * D(v, x),
                ),
            ],
            1,
        ),
        # ut = 7.15666*D(u, x)/u - 5.677*D(u, x)
        PDECase(
            [
                (
                    u,
                    7.15666 * D(u, x) / u + 5.677 * D(u, x),
                )
            ],
            1,
        ),
        # u_t = u_xx * u**2 + 2
        # v_t = v_xx/u**3 + u
        PDECase(
            [
                (u, u**2 * D(u, x, 2) + 2),
                (v, D(v, x, 2) / u**3 + u),
            ],
            2,
        ),
        # u_t = 1/(5 * (u + 1))
        PDECase([(u, 1 / (5 * (u + 1)))], 1),
        # u_t = 1/(0.6 * u + 1.3)**2
        PDECase([(u, 1 / (0.6 * u + 0.5) ** 2)], 1),
        # u_t = - 1/(u + u**2) - u
        PDECase([(u, 1 / (u**2 + 1))], 1),
        # u_t = - u_x/(u + 1)
        PDECase([(u, D(u, x) / (u + 1))], 1),
        # u_t = 1/(u + 1)**2 + 1/(u - 1)
        PDECase([(u, 1 / (u + 1) ** 2 + 1 / (u - 1))], 1),
        # u_t = 1/(u**2) - 0.5 * u + 1
        PDECase([(u, 1 / (u**2) - 0.5 * u + 1)], 1),
        # u_t = -u_xx/u - u**2 - u + 5
        # v_t = u/v - v + 5
        PDECase(
            [
                (u, -D(u, x, 2) / u - u**2 - u + 5),
                (v, u / v - v + 5),
            ],
            4,
            nvars_bound=6,
        ),
        # u_t = 1/((u+1)(u+2))
        PDECase([(u, 1 / ((u + 1) * (u + 2)))], 0),
        # u_t = 1/((v+1)(u+1))
        # v_t = 1/u
        PDECase([(u, 1 / ((v + 1) * (u + 1))), (v, 1 / u)], 0),
    ]

    test_cases_coeff_sym = [
        # u_t = omega * u**3 * u_xxx
        PDECase([(u, omega * u**3 * D(u, x, 3))], 3),
        # u_t = 1/(omega * (u + 1))
        PDECase([(u, 1 / (omega * (u + 1)))], 1),
    ]
    return {
        "t": t,
        "x": x,
        "helpers": helpers,
        "quad": test_cases_quad,
        "rat": test_cases_rat,
        "coeff": test_cases_coeff_sym,
    }


def quadratization_test(search_alg, test_cases, data):
    """
    Main method to test the quadratization algorithm.
    """
    helpers = data["helpers"]
    x = data["x"]
    for test in test_cases:
        print("\nTest case: ")
        [print(f"Derivative({eq[0]}, t)", "=", eq[1]) for eq in test.func_eq]
        poly_syst = quadratize(
            test.func_eq,
            test.n_diff,
            search_alg=search_alg,
            max_der_order=test.max_der_order,
            nvars_bound=test.n_vars_bound,
        )
        assert isinstance(poly_syst, PDESys), (
            f"Quadratization not found for {test.func_eq}"
        )
        quad_prop, frac_vars = poly_syst.get_aux_vars()
        print(f"Quadratization: {quad_prop}")
        print(f"Rational variables: {frac_vars}")

        quad_prop_expr, frac_vars_expr = helpers.transform_new_vars(
            quad_prop, frac_vars
        )
        new_vars, frac_vars, refac = helpers.rewrite_expr(
            test, quad_prop_expr, frac_vars_expr, x
        )

        refac_new_vars, exprs_orig = helpers.construct_quadratic_PDE(
            new_vars, frac_vars, test, refac, x
        )
        results = check_quadratization(test.func_eq, quad_prop, test.n_diff)
        for i in range(len(exprs_orig)):
            rewritten_result = results[1][i].rhs.subs(refac_new_vars)
            assert (
                simplify(
                    helpers.convert_to_rational(exprs_orig[i])
                    - helpers.convert_to_rational(rewritten_result.evalf())
                )
                == 0
            ), (
                f"Test failed: expressions are not equal for {exprs_orig[i]} \n"
                + f"Equation: {results[1][i]} \n"
                + f"Original expression: {expand(helpers.convert_to_rational(exprs_orig[i]))} \n"
                + f"Quad expression: {helpers.convert_to_rational(rewritten_result.evalf())} \n"
                + f"Substraction: {simplify(helpers.convert_to_rational(exprs_orig[i]) - helpers.convert_to_rational(rewritten_result.evalf()))}"
            )


def test_branch_and_bound(test_data):
    """
    Test the branch and bound algorithm.
    """
    quadratization_test("bnb", test_data["quad"], test_data)


def test_nearest_neighbor(test_data):
    """
    Test the nearest neighbor algorithm.
    """
    quadratization_test("inn", test_data["quad"], test_data)


def test_rational_pdes(test_data):
    """
    Test PDEs quadratization for rational PDEs.
    """
    quadratization_test("bnb", test_data["rat"], test_data)


def test_symbolic_coeff(test_data):
    """
    Test PDEs quadratization for PDEs with symbolic coefficients.
    """
    quadratization_test("inn", test_data["coeff"], test_data)
