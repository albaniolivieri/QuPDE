import pytest
import sympy as sp
from qupde.utils import get_pol_diff_order


@pytest.fixture(scope="module")
def test_data():
    poly_syms = sp.symbols("u u_x1 u_x2 w_0 w_0x5 w_0x10 w_10x4")
    R, poly_vars = sp.xring(poly_syms, sp.QQ)
    u, u_x1, u_x2, w_0, w_0x5, w_0x10, w_10x4 = poly_vars
    polys = [
        (u + u_x1, 1),
        (u * u_x1, 1),
        (w_10x4, 4),
        (u_x2, 2),
        (w_0 + w_0x5, 5),
        (w_0x10 * w_0x5, 10),
        (w_10x4 + w_0x10, 10),
    ]
    return {"polys": polys}


def test_get_pol_diff_order(test_data):
    for test in test_data["polys"]:
        assert test[1] == get_pol_diff_order(test[0]), (
            f"The derivative order of {test[0]} does not match, got {get_pol_diff_order(test[0])}"
        )
