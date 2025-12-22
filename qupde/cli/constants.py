from enum import StrEnum


class InputFormat(StrEnum):
    sympy = "sympy"
    mathematica = "mathematica"


class SortFun(StrEnum):
    by_fun = "by_fun"
    by_degree_order = "by_degree_order"
    by_order_degree = "by_order_degree"


class SearchAlg(StrEnum):
    bnb = "bnb"
    inn = "inn"


class Printing(StrEnum):
    pprint = "pprint"
    latex = "latex"
    none = "none"
