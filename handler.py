import numpy as np
import matplotlib.pyplot as plt
import math
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application
import argparse
import string


def parse_arg_func(arg_dict, n, var_list):
    for i in range(n):
        arg_dict[string.ascii_lowercase[i]] = var_list[i]
    return arg_dict



def parse_args():
    parser = argparse.ArgumentParser('Some description')
    parser.add_argument('-f', '--func-expr', required=True, help='function (expression) written using sympy syntax. Note that all arguments should follow pattern of english alphabet: x0: a, x1: b.., xn: z (take a look at few examples as readme.md)')
    parser.add_argument('-n', '--number_of_arguments', type=int, required=True, help='number of arguments of function')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    transformations = (standard_transformations +
        (implicit_multiplication_application,))

    # create function
    expr = parse_expr(args.func_expr, transformations=transformations)

    # create dictionary of variables
    arg_dict = dict()
    for i in range(args.number_of_arguments):
        arg_dict[string.ascii_lowercase[i]] = i + 1
    # check
    print("Some evaluation check")
    print(expr.evalf(16, subs=arg_dict))
    print("END")

