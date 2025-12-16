import sympy as sp
from sympy import Derivative as D
import time
import statistics
from qupde import quadratize

"""
The Brusselator system was developed to model morphogenesis and pattern formation in chemical reactions:
    u_t = d_1 u_x + lambda * (1 - (b + 1) * u + b * u^2 * v),
    v_t = d_2 v_x + lambda * a^2 * (u - u^2 * v)
References:
    Li, B., & Wang, M.-x. (2008). Diffusion-driven instability and Hopf bifurcation in Brusselator system. 
    Applied Mathematics and Mechanics, 29(6), 825–832. https://doi.org/10.1007/s10483-008-0614-y
"""
t, x = sp.symbols('t x')
d_1, d_2, a, b = sp.symbols('d_1 d_2 a b', constant=True)
l = sp.symbols('lambda', constant=True)
u = sp.Function('u')(t,x)
v = sp.Function('v')(t,x)

u_t = d_1 * D(u, x) + l * (1 - (b + 1) * u + b * u**2 * v)
v_t = d_2 * D(v, x) + l * a**2 * (u - u**2 * v)

# we run QuPDE for the Brusselator system
if __name__ == '__main__':   
    times= []
    for i in range(10):
        ti = time.time()
        quadratize([(u, u_t), (v, v_t)], diff_ord=1)
        times.append(time.time() - ti) 
    avg=statistics.mean(times[1:])
    std=statistics.stdev(times[1:])

    quadratize([(u, u_t), (v, v_t)], diff_ord=1, printing='pprint')
    
    print('Average time', avg)
    print('Standard deviation', std)