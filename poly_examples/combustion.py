import sympy as sp
from sympy import Derivative as D
from qupde.polynomialization import polynomialize, polynomialize_and_quadratize

t, x = sp.symbols("t x")
c_1 = sp.Function("c_1")(t, x)
c_2 = sp.Function("c_2")(t, x)
c_3 = sp.Function("c_3")(t, x)
c_4 = sp.Function("c_4")(t, x)
A = sp.symbols("A", constant=True)
E_a = sp.symbols("E_a", constant=True)
R_u = sp.symbols("R_a", constant=True)
T = sp.symbols("T", constant=True)

c_1_t = -A * sp.exp(E_a / (R_u * T)) * c_1**0.2 * c_2**1.3
c_2_t = 2 * c_1_t
c_3_t = -c_1_t
c_4_t = -c_2_t

# new_pde, new_vars = (polynomialize([(c_1, c_1_t), (c_2, c_2_t), (c_3, c_3_t), (c_4, c_4_t)], is_rat=True))
# print('New PDE\n', new_pde, '\nNew variables\n', new_vars)

new_quad_pde = polynomialize_and_quadratize(
    [(c_1, c_1_t), (c_2, c_2_t), (c_3, c_3_t), (c_4, c_4_t)],
    diff_ord=0,
    nvars_bound=6,
    printing="latex",
)
print(new_quad_pde.get_aux_vars())
