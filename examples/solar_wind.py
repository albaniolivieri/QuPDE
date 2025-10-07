import sympy as sp
from sympy import Derivative as D
import sys
import time
import statistics
sys.path.append("..")
from qupde.quadratize import quadratize
from qupde.mon_heuristics import *
import cProfile
import pstats

"""
The HUX (Heliospheric Upwinding eXtrapolation) model is a two-dimensional time-stationary 
model that predicts the heliospheric solar wind speed:
    u_r = Omega_rot * u_phi/u.
References:
    Issan, O., & Kramer, B. (2023). Predicting solar wind streams from the inner-heliosphere
    to Earth via shifted operator inference. Journal of Computational Physics, 473, 111689.
    https://doi.org/10.1016/j.jcp.2022.111689
"""

r, phi = sp.symbols('r phi')
omega = sp.symbols('omega', constant=True)
v = sp.Function('v')(r,phi)

v_r = (omega*D(v, phi)) / v

if __name__ == '__main__':
    times = []
    for i in range(10):
        ti = time.time()
        quadratize([(v, v_r)], diff_ord=1, first_indep=r, search_alg="bnb")
        times.append(time.time() - ti) 
    avg = statistics.mean(times[1:])
    std = statistics.stdev(times[1:])

    print(quadratize([(v, v_r)], diff_ord=1, first_indep=r, search_alg='bnb', printing='pprint'))
    
    print('Average time', avg)
    print('Standard deviation', std)