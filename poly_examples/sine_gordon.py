import sympy as sp
from sympy import Derivative as D
from qupde.polynomialization import polynomialize 

t, x = sp.symbols('t x')
u = sp.Function('u')(t, x)
v = sp.Function('v')(t, x)


u_t = v
v_t = D(u, x, 2) - sp.sin(u)

new_pde, new_vars = (polynomialize([(u, u_t), (v, v_t)]))
print('New PDE\n', new_pde, '\nNew variables\n', new_vars)