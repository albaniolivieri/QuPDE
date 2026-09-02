import sympy as sp
from qupde.polynomialization import polynomialize_and_quadratize

t, x = sp.symbols("t x")
u = sp.Function("u")(t, x)
a = sp.symbols("a", constant=True)
b = sp.symbols("b", constant=True)

u_t = 1 / (1 + sp.exp(-a * u - b))

# new_pde, new_vars = polynomialize([(u, u_t)], is_rat=True)
# print("New PDE\n", new_pde, "\nNew variables\n", new_vars)

new_quad_pde = polynomialize_and_quadratize([(u, u_t)], diff_ord=0, printing="latex")
print(new_quad_pde.get_aux_vars())
