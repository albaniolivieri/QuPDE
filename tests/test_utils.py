import pytest
import sympy as sp
from qupde.utils import get_pol_diff_order


@pytest.fixture(scope="module")
def test_data():
    poly_vars = [sp.symbols("u u_x1 u_x2 w_0 w_0x5 w_0x10")]
    R, u, u_x1, u_x2, w_0, w_0x5, w_0x10 = sp.xring(poly_vars, sp.QQ)
    polys = [
        (u + u_x1, 1),
        (u * u_x1, 1),
        (u_x2, 2),
        (w_0 + w_0x5, 5),
        (w_0x10 * w_0x5, 10),
    ]
    return {"polys": polys, "ring": R}


def test_get_pol_diff_order(test_cases):
    for test in test_cases:
        assert test[1] == get_pol_diff_order(test[0]), (
            f"The derivative order of {test[0]} does not match"
        )
