import sympy as sp
from sympy import Derivative as D
import time
import statistics
import sys
sys.path.append("..")
from qupde.quadratize import quadratize
import cProfile
import pstats

"""
The Euler equations are derived from tile physical principles of conservation of mass, momentum, and energy:
    rho_t = -u * rho_x - rho * u_x 
    u_t = -u_x * u - p_x/rho}
    p_t = -gamma * u_x * p - u * p_x.
References:
    Huynh, H. T. (1995). Accurate upwind methods for the Euler equations. SIAM Journal on Numerical Analysis, 
    32(5), 1565–1619. https://doi.org/10.1137/0732071
"""

t, x = sp.symbols('t x')
rho = sp.Function('rho')(t,x)
u = sp.Function('u')(t,x)
p = sp.Function('p')(t,x)
gamma = sp.symbols('gamma', constant=True)

rho_t = -u * D(rho, x) - rho * D(u, x)
u_t = -u * D(u, x) - D(p, x) / rho
p_t = -gamma * p * D(u, x) - u * D(p, x)

# we run QuPDE for the Euler equations
if __name__ == '__main__':
    times = []
    for i in range(10):
        ti = time.time()
        quadratize([(rho, rho_t), (u, u_t), (p, p_t)], diff_ord=2, search_alg='bnb')
        times.append(time.time() - ti) 
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])
    
    ti = time.time()
    
    print(quadratize([(rho, rho_t), (u, u_t), (p, p_t)], diff_ord=2, search_alg='bnb', printing='pprint'))

    print('Average time', avg)
    print('Standard deviation', std)
