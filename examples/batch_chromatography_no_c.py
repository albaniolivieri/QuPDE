import sympy as sp
from sympy import Derivative as D
import time
import statistics
import sys
sys.path.append("..")
from qupde import quadratize
from qupde.mon_heuristics import *

t, x = sp.symbols('t x')
c_a = sp.Function('p_a')(t,x)
c_b = sp.Function('p_b')(t,x)
b_a = sp.Function('r_a')(t,x)
b_b = sp.Function('r_b')(t,x)

def q_eq(c_a, c_b, i):
    denom = 1 + c_a + c_b
    return (2*c_a / denom)

b_at = (q_eq(c_a, c_b, 0) - b_a)
b_bt = (q_eq(c_a, c_b, 1) - b_b)
c_at = (q_eq(c_a, c_b, 0) - b_a) - D(c_a, x) + D(c_a, x, 2)
c_bt = (q_eq(c_a, c_b, 1) - b_b) - D(c_b, x) + D(c_b, x, 2)

print(quadratize([(c_a, c_at), (c_b, c_bt), (b_a, b_at), (b_b, b_bt)], 3, sort_fun=by_fun, search_alg='bnb', max_der_order=4, nvars_bound=8))