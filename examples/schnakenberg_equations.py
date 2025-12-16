import sympy as sp
from sympy import Derivative as D
import time
import statistics
from qupde import quadratize

"""
The Schnakenberg equations are evolution equations for reaction-diffusion systems with cross-diffusion:
    u_t = D_u u_xx + D_uv * v_{xx} + k_1 * a_1 - k_2 * u + k_3 * u^2 * v,
    v_t = D_v v_xx + D_vu * u_xx + k_4 * b_1 - k_3 * u^2v
References:
    Madzvamuse, A., Ndakwo, H. S., & Barreira, R. (2014). Cross-diffusion-driven instability for 
    reaction-diffusion systems: Analysis and simulations. Journal of Mathematical Biology, 70(4), 
    709–743. https://doi.org/10.1007/s00285-014-0779-6
"""

t, x = sp.symbols("t x")
a, b, gamma, d_uv, d_vu, d_u, d_v = sp.symbols(
    "a b gamma d_uv d_vu d_u d_v", constant=True
)
v = sp.Function("v")(t, x)
u = sp.Function("u")(t, x)

u_t = d_u * D(u, x, 2) + d_uv * D(v, x, 2) + gamma * (a - u + u**2 * v)
v_t = d_v * D(v, x, 2) + d_vu * D(u, x, 2) + gamma * (b - u**2 * v)

# we run QuPDE for the Schnakenberg equations
if __name__ == "__main__":
    times = []
    for i in range(10):
        ti = time.time()
        quadratize([(u, u_t), (v, v_t)], diff_ord=2)
        times.append(time.time() - ti)
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])

    quadratize([(u, u_t), (v, v_t)], diff_ord=2, printing="pprint")

    print("Average time", avg)
    print("Standard deviation", std)
