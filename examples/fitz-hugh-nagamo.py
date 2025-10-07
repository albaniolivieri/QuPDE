import sympy as sp
from sympy import Derivative as D
import time
import statistics
import sys
sys.path.append("..")
from qupde.quadratize import quadratize
from qupde.mon_heuristics import *

"""
The FitzHugh-Nagumo system is a simplified neuron model of the Hodgkin-Huxley model, 
which describes activation and deactivation dynamics of a spiking neuron:
    v_t = epsilon * v_xx + (1/epsilon) * v * (v - 0.1) * (1 - v) - (1/epsilon) * u + (1/epsilon) * q,
    y_t = h * v - gamma * y + q.
References:
    Chaturantabut, S., & Sorensen, D. C. (2010). Nonlinear model reduction via discrete empirical interpolation.
    SIAM Journal on Scientific Computing, 32(5), 2737-2764.
"""
t, x = sp.symbols('t x')
v = sp.Function('v')(t,x)
y = sp.Function('y')(t,x)
epsilon, h, gamma, r = sp.symbols('epsilon h gamma r', constant=True)

v_t = epsilon * D(v, x, 2) - (1/epsilon) * (v * (v - 0.1) * (1 - v)) - y/epsilon + r/epsilon
y_t = h * v - gamma * y + r

# we run QuPDE for the FitzHugh-Nagumo system
if __name__ == '__main__':
    times= []
    for i in range(10):
        ti = time.time()
        quadratize([(v, v_t), (y, y_t)], 3, search_alg='bnb')
        times.append(time.time() - ti) 
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])

    print(quadratize([(v, v_t), (y, y_t)], 3, search_alg='bnb', printing='pprint'))

    print('Average time', avg)
    print('Standard deviation', std)

