import sympy as sp
from qupde.polynomialization import polynomialize_and_quadratize

t, x = sp.symbols("t x")
u = sp.Function("u")(t, x)
v = sp.Function("v")(t, x)
z = sp.Function("z")(t, x)
k_1 = sp.symbols("k_1", constant=True)
k_2 = sp.symbols("k_2", constant=True)
k_3 = sp.symbols("k_3", constant=True)
k_4 = sp.symbols("k_4", constant=True)
k_5 = sp.symbols("k_5", constant=True)
k_6 = sp.symbols("k_6", constant=True)
s_1 = sp.symbols("s_1", constant=True)
s_2 = sp.symbols("s_2", constant=True)
s_3 = sp.symbols("s_3", constant=True)
n_1 = sp.symbols("n_1", constant=True)
n_2 = sp.symbols("n_2", constant=True)
K_1, K_2 = sp.symbols("K_1 K_2", constant=True)
alpha = sp.symbols("alpha", constant=True)


u_t = k_1 * (s_1 - u) * (K_1**n_1 / (K_1**n_1 + z**n_1)) - k_2 * u
v_t = k_3 * (s_2 - v) * u * (1 + (alpha * z**n_2) / (K_2**n_2 + z**n_2)) - k_4 * v
z_t = k_5 * (s_3 - z) * v - k_6 * z

# u_t = (1 - u) * K_1 / (1 + z**n_1) - u
# v_t = (1 - v) * u * (1 + 2 * z**alpha) / (1 + z**alpha) - v
# z_t = (1 - z) * v - alpha * z

# new_pde, new_vars = (polynomialize([(u, u_t), (v, v_t), (z, z_t)]))
# print('New PDE\n', new_pde, '\nNew variables\n', new_vars)

new_quad_pde = polynomialize_and_quadratize(
    [(u, u_t), (v, v_t), (z, z_t)], diff_ord=0, nvars_bound=9
)
print(new_quad_pde.get_aux_vars())
