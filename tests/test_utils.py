import pytest
import sympy as sp
from qupde.utils import get_pol_diff_order, diff_dict


@pytest.fixture(scope="module")
def test_data():
    poly_syms = sp.symbols("u u_x1 u_x2 w_0 w_0x1 w_0x5 w_0x10 w_10x4")
    R, poly_vars = sp.xring(poly_syms, sp.QQ)
    u, u_x1, u_x2, w_0, w_0x1, w_0x5, w_0x10, w_10x4 = poly_vars
    polys_diff_ord = [
        (u * u_x1, 1),
        (w_10x4, 4),
        (u_x2, 2),
        (w_0 + w_0x5, 5),
        (w_0x10 * w_0x5, 10),
        (w_10x4 + w_0x10, 10),
    ]
    dic = {u: u_x1, u_x1: u_x2, w_0: w_0x1}
    polys_diff_dict = [
        (u * u_x1, u_x1**2 + u * u_x2),
        (w_0 + u, w_0x1 + u_x1),
        (w_0 * u_x1 * u, w_0x1 * u * u_x1 + w_0 * u_x2 * u + w_0 * u_x1**2),
    ]
    return {"diff_order": polys_diff_ord, "diff_dict": (dic, polys_diff_dict)}


def test_get_pol_diff_order(test_data):
    for test in test_data["diff_order"]:
        assert test[1] == get_pol_diff_order(test[0]), (
            f"The derivative order of {test[0]} does not match, got {get_pol_diff_order(test[0])}"
        )


def test_diff_dict(test_data):
    dic, test_polys = test_data["diff_dict"]
    for test in test_polys:
        assert test[1] == diff_dict(test[0], dic), (
            f"The derivative of {test[0]} does not match, got {diff_dict(test[0], dic)}"
        )
