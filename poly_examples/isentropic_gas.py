from sympy import *
from sympy import Derivative as D
from qupde.polynomialization import polynomialize, polynomialize_and_quadratize

t, zetav = symbols('t zeta')
u = Function('u')(t, zetav)
v = Function('v')(t, zetav)
A = symbols('A', constant = True)
gammac = symbols('gammac', constant = True)

u_t = D(v, zetav)
v_t = -gammac * A * u**(gammac-1) * D(u, zetav)

new_pde, new_vars = (polynomialize([(u, u_t), (v, v_t)], second_indep=zetav, is_rat=True))
print('New PDE\n', new_pde, '\nNew variables\n', new_vars)

new_quad_pde = polynomialize_and_quadratize([(u, u_t), (v, v_t)], second_indep=zetav, diff_ord=2)
print(new_quad_pde.get_aux_vars())
