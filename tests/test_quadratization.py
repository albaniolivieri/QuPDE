import unittest
import math
from sympy import symbols, simplify, expand, nsimplify, Function
from sympy import Derivative as D

from qupde import quadratize
from qupde.utils import get_sys_order
from qupde.quadratization import check_quadratization
from qupde.pde_sys import PDESys


class TestCase:
    def __init__(self, func_eq, n_diff, max_der_order=None, nvars_bound=10) -> None:
        self.func_eq = func_eq
        self.n_diff = n_diff
        self.max_der_order = max_der_order
        self.n_vars_bound = nvars_bound


class TestQuadratization(unittest.TestCase):
    """
    Test cases for the PDEs quadratization algorithm.
    """

    def setUp(self):
        """
        Set up the test cases for the PDEs quadratization algorithm.
        """
        self.t, self.x = symbols("t x")
        self.u = Function("u")(self.t, self.x)
        self.v = Function("v")(self.t, self.x)
        self.omega = symbols("omega", constant=True)

        self.test_cases_quad = []

        # u_t = - u_x * u**3 - 1/3 * u_x * u**2
        # v_t = v_x * u - 2 * v_x
        self.test_cases_quad.append(
            TestCase(
                [
                    (
                        self.u,
                        -D(self.v, self.x) * self.u**3
                        - 1 / 3 * D(self.v, self.x) * self.u**2,
                    ),
                    (self.v, D(self.v, self.x) * self.u - 2 * D(self.v, self.x)),
                ],
                2,
                max_der_order=2,
            )
        )

        # u_t = u**3 * u_xxx
        self.test_cases_quad.append(
            TestCase([(self.u, self.u**3 * D(self.u, self.x, 3))], 3)
        )

        # u_t = u_x**3 + u**3
        self.test_cases_quad.append(
            TestCase([(self.u, D(self.u, self.x) ** 3 + self.u**3)], 2, max_der_order=3)
        )

        # u_t = u_x**4
        self.test_cases_quad.append(
            TestCase([(self.u, D(self.u, self.x) ** 4)], 3, max_der_order=2)
        )

        # u_t = u_x**3 * u
        self.test_cases_quad.append(
            TestCase([(self.u, D(self.u, self.x) ** 3 * self.u)], 2, max_der_order=3)
        )

        # u_t = u_x**3
        self.test_cases_quad.append(
            TestCase([(self.u, D(self.u, self.x) ** 3)], 2, max_der_order=3)
        )

        self.test_cases_rat = []

        # u_t = 3.4 * u_x * v_x
        # v_t = v_x / v - pi * v_x
        self.test_cases_rat.append(
            TestCase(
                [
                    (self.u, 3.4 * D(self.u, self.x) * D(self.v, self.x)),
                    (
                        self.v,
                        D(self.v, self.x) / self.v
                        - round((math.pi), 5) * D(self.v, self.x),
                    ),
                ],
                4,
            )
        )

        # ut = 7.15666*D(u, x)/u - 5.677*D(u, x)
        self.test_cases_rat.append(
            TestCase(
                [
                    (
                        self.u,
                        7.15666 * D(self.u, self.x) / self.u
                        + 5.677 * D(self.u, self.x),
                    )
                ],
                1,
            )
        )

        # u_t = u_xx * u**2 + 2
        # v_t = v_xx/u**3 + u
        self.test_cases_rat.append(
            TestCase(
                [
                    (self.u, self.u**2 * D(self.u, self.x, 2) + 2),
                    (self.v, D(self.v, self.x, 2) / self.u**3 + self.u),
                ],
                2,
                10,
            )
        )

        # u_t = 1/(5 * (u + 1))
        self.test_cases_rat.append(TestCase([(self.u, 1 / (5 * (self.u + 1)))], 1))

        # u_t = 1/(0.6 * u + 1.3)**2
        self.test_cases_rat.append(
            TestCase([(self.u, 1 / (0.6 * self.u + 0.5) ** 2)], 3)
        )

        # u_t = - 1/(u + u**2) - u
        self.test_cases_rat.append(TestCase([(self.u, 1 / (self.u**2 + 1))], 3))

        # u_t = - u_x/(u + 1)
        self.test_cases_rat.append(
            TestCase([(self.u, D(self.u, self.x) / (self.u + 1))], 3)
        )

        # u_t = 1/(u + 1)**2 + 1/(u - 1)
        self.test_cases_rat.append(
            TestCase(
                [(self.u, 1 / (self.u + 1) ** 2 + 1 / (self.u - 1))], 4, max_der_order=4
            )
        )

        # u_t = 1/(u**2) - 0.5 * u + 1
        self.test_cases_rat.append(
            TestCase([(self.u, 1 / (self.u**2) - 0.5 * self.u + 1)], 4, max_der_order=4)
        )

        # u_t = -u_xx/u - u**2 - u + 5
        # v_t = u/v - v + 5
        self.test_cases_rat.append(
            TestCase(
                [
                    (self.u, -D(self.u, self.x, 2) / self.u - self.u**2 - self.u + 5),
                    (self.v, self.u / self.v - self.v + 5),
                ],
                2,
                max_der_order=10,
                nvars_bound=6,
            )
        )

        # u_t = 1/((u+1)(u+2))
        self.test_cases_rat.append(
            TestCase([(self.u, 1 / ((self.u + 1) * (self.u + 2)))], 0, max_der_order=1)
        )

        # u_t = 1/((v+1)(u+1))
        # v_t = 1/u
        self.test_cases_rat.append(
            TestCase(
                [(self.u, 1 / ((self.v + 1) * (self.u + 1))), (self.v, 1 / self.u)],
                0,
                max_der_order=1,
            )
        )

        self.test_cases_coeff_sym = []

        # u_t = omega * u**3 * u_xxx
        self.test_cases_coeff_sym.append(
            TestCase([(self.u, self.omega * self.u**3 * D(self.u, self.x, 3))], 3)
        )

        # u_t = 1/(omega * (u + 1))
        self.test_cases_coeff_sym.append(
            TestCase([(self.u, 1 / (self.omega * (self.u + 1)))], 1)
        )

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

    def differentiate_x(self, new_vars, n_diff):
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
                    (symbols(f"w_{i}{self.x}{j}"), D(vars_prop[i], self.x, j).doit())
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
                    (symbols(f"q_{i}{self.x}{j}"), D(frac_vars[i], self.x, j).doit())
                    for j in range(1, var_ord + 1)
                ]
            )
        return quad_vars

    def rewrite_expr(self, test_case, new_vars, frac_vars):
        """
        Rewrite the expressions with the new variables definitions.
        """
        max_order = get_sys_order(list(zip(*test_case.func_eq))[1])
        refac = []
        for fun, _ in test_case.func_eq:
            refac += [
                (symbols(f"{fun.name}_{self.x}{i}"), D(fun, self.x, i))
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

    def construct_quadratic_PDE(self, new_vars, frac_vars, test_case, refac):
        """
        Returns the new variables expressions and derivatives, and its differential equations
        to construct the quadratic PDE transformation.
        """
        var_dic = [(symbols(f"w_{i}"), new_vars[i]) for i in range(len(new_vars))]
        total_vars = (new_vars, [rel for _, rel in frac_vars])
        quad_vars = self.differentiate_x(total_vars, test_case.n_diff)
        deriv_t = self.differentiate_t(
            test_case.func_eq,
            [(var, expr.subs(frac_vars)) for var, expr in var_dic] + frac_vars,
        ) + [(symbols(eqs[0].name + "_t"), eqs[1]) for eqs in test_case.func_eq]
        refac += quad_vars + frac_vars
        exprs_orig = [expr for _, expr in deriv_t]
        return refac, exprs_orig

    def quadratization_test(self, search_alg, test_cases, sort_heur="by_fun"):
        """
        Main method to test the quadratization algorithm.
        """
        for test in test_cases:
            print("\nTest case: ")
            [print(f"Derivative({eq[0]}, t)", "=", eq[1]) for eq in test.func_eq]
            poly_syst = quadratize(
                test.func_eq,
                test.n_diff,
                sort_fun=sort_heur,
                search_alg=search_alg,
                max_der_order=test.max_der_order,
                nvars_bound=test.n_vars_bound,
            )
            self.assertIsInstance(
                poly_syst, PDESys, f"Quadratization not found for {test.func_eq}"
            )
            quad_prop, frac_vars = poly_syst.get_aux_vars()
            print(f"Quadratization: {quad_prop}")
            print(f"Rational variables: {frac_vars}")

            quad_prop_expr, frac_vars_expr = self.transform_new_vars(
                quad_prop, frac_vars
            )
            new_vars, frac_vars, refac = self.rewrite_expr(
                test, quad_prop_expr, frac_vars_expr
            )

            refac_new_vars, exprs_orig = self.construct_quadratic_PDE(
                new_vars, frac_vars, test, refac
            )
            results = check_quadratization(test.func_eq, quad_prop, test.n_diff)
            for i in range(len(exprs_orig)):
                rewritten_result = results[1][i].rhs.subs(refac_new_vars)
                self.assertEqual(
                    simplify(
                        self.convert_to_rational(exprs_orig[i])
                        - self.convert_to_rational(rewritten_result.evalf())
                    ),
                    0,
                    f"Test failed: expressions are not equal for {exprs_orig[i]} \n"
                    + f"Equation: {results[1][i]} \n"
                    + f"Original expression: {expand(self.convert_to_rational(exprs_orig[i]))} \n"
                    + f"Quad expression: {self.convert_to_rational(rewritten_result.evalf())} \n"
                    + f"Substraction: {simplify(self.convert_to_rational(exprs_orig[i]) - self.convert_to_rational(rewritten_result.evalf()))}",
                )

    def test_branch_and_bound(self):
        """
        Test the branch and bound algorithm.
        """
        self.quadratization_test(search_alg="bnb", test_cases=self.test_cases_quad)

    def test_nearest_neighbor(self):
        """
        Test the nearest neighbor algorithm.
        """
        self.quadratization_test(search_alg="inn", test_cases=self.test_cases_quad)

    def test_rational_pdes(self):
        """
        Test PDEs quadratization for rational PDEs.
        """
        self.quadratization_test(search_alg="bnb", test_cases=self.test_cases_rat)

    def test_symbolic_coeff(self):
        """
        Test PDEs quadratization for PDEs with symbolic coefficients.
        """
        self.quadratization_test(search_alg="inn", test_cases=self.test_cases_coeff_sym)


if __name__ == "__main__":
    unittest.main()
