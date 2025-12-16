#!/usr/bin/python
import time
import sys
import statistics

from sympy import *
from sympy import Derivative as D

sys.path.append("..")
from qupde import quadratize as sparse
from deprecated import check_quad_gauss as dense
from deprecated import check_quad_matrix as matrix

t, x = symbols("t x")
u = Function("u")(t, x)
u_x1 = symbols("u_x1")

u_sym = symbols("u")
ut5 = u**3 * D(u, x, 3)

times_avg, times_std, diff_orders = [], [], []

if len(sys.argv) > 1:
    if sys.argv[1] == "sparse":
        for i in range(1, 11):
            times = []
            for _ in range(10):
                ti = time.time()
                sparse.test_quadratization([(u, ut5)], [u_sym**3, u_sym * u_x1**2], i)
                times.append(time.time() - ti)
            times_avg.append(statistics.mean(times))
            times_std.append(statistics.stdev(times))
            diff_orders.append(i)
            print("Diff order:", i)
    elif sys.argv[1] == "dense":
        for i in range(1, 11):
            times = []
            for _ in range(10):
                ti = time.time()
                dense.get_quadratization([(u, ut5)], [u**3, u * D(u, x) ** 2], i)
                times.append(time.time() - ti)
            times_avg.append(statistics.mean(times))
            times_std.append(statistics.stdev(times))
            diff_orders.append(i)
            print("Diff order:", i)
    elif sys.argv[1] == "matrix":
        for i in range(1, 11):
            times = []
            for _ in range(10):
                ti = time.time()
                matrix.get_quadratization([(u, ut5)], [u**3, u * D(u, x) ** 2], i)
                times.append(time.time() - ti)
            times_avg.append(statistics.mean(times))
            times_std.append(statistics.stdev(times))
            diff_orders.append(i)
            print("Diff order:", i)

print("diff_order", diff_orders, "times_avg:", times_avg, "times_std:", times_std)
