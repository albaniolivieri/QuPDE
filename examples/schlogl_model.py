import sympy as sp
from sympy import Derivative as D
import time
import statistics
from qupde import quadratize

"""
The Schlögl model is a simple example of a chemical reaction system that exhibits bistability. 
We show a version of this model without input functions: 
    u_t = u_xx - k * (u - u_1) * (u - u_2) * (u - u_3)
References: 
    Buchholz, R., Engel, H., Kammann, E., & Tröltzsch, F. (2013). On the optimal control of the
    Schlögl-model. Computational Optimization and Applications, 56(1), 153–185. https://doi.org/10.1007/s10589-013-9550-y
"""

t, x = sp.symbols('t x')
u = sp.Function('u')(t,x)
v_1, v_2, v_3 = sp.symbols('v_1 v_2 v_3', constant=True)
k = sp.symbols('k', constant=True)

u_t = D(u, x, 2) - k * (u - v_1) * (u - v_2) * (u - v_3)

# we run QuPDE for the Schlögl model
if __name__ == '__main__':
    times= []
    for i in range(10):
        ti = time.time()
        quadratize([(u, u_t)], diff_ord=2)
        times.append(time.time() - ti) 
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])
    
    quadratize([(u, u_t)], diff_ord=2, printing='pprint')

    print('Average time', avg)
    print('Standard deviation', std)
