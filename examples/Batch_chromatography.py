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
q_a = sp.Function('r_a')(t,x)
q_b = sp.Function('r_b')(t,x)
epsilon = sp.symbols('epsilon', constant=True)
Pe = sp.symbols('Pe', constant=True)
H = [[sp.symbols('H_a1', constant=True), sp.symbols('H_a2', constant=True)], [sp.symbols('H_b1', constant=True), sp.symbols('H_b2', constant=True)]]
K = [[sp.symbols('Ka1', constant=True), sp.symbols('Ka2', constant=True)], [sp.symbols('Kb1', constant=True), sp.symbols('Kb2', constant=True)]]
L = sp.symbols('L', constant=True)
Q = sp.symbols('Q', constant=True)
Ac = sp.symbols('Ac', constant=True)
k_a = sp.symbols('k_a', constant=True)
k_b = sp.symbols('k_b', constant=True)
c_af = sp.symbols('c_af', constant=True)
c_bf = sp.symbols('c_bf', constant=True)

def q_eq(c_a, c_b, i):
    denom1 = 1 + K[0][0] * c_af * c_a + K[1][0] * c_bf * c_b
    denom2 = 1 + K[0][1] * c_af * c_a + K[1][1] * c_bf * c_b
    return (H[i][0] * c_a / denom1) + (H[i][1] * c_a / denom2)

q_at = (L/(Q/(epsilon*Ac))) * k_a * (q_eq(c_a, c_b, 0) - q_a)
q_bt = (L/(Q/(epsilon*Ac))) * k_b * (q_eq(c_a, c_b, 1) - q_b)
c_at = ((epsilon - 1)/epsilon) * (L/(Q/(epsilon*Ac))) * k_a * (q_eq(c_a, c_b, 0) - q_a) - D(c_a, x) + (1/Pe) * D(c_a, x, 2)
c_bt = ((epsilon - 1)/epsilon) * (L/(Q/(epsilon*Ac))) * k_b * (q_eq(c_a, c_b, 1) - q_b)- D(c_b, x) + (1/Pe) * D(c_b, x, 2)

print(quadratize([(c_a, c_at), (c_b, c_bt), (q_a, q_at), (q_b, q_bt)], 3, sort_fun=by_fun, search_alg='bnb', max_der_order=5, nvars_bound=10))