import numpy as np
import matplotlib.pyplot as plt
import math

from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application

import argparse
import string
# methods importing
from gradient_method import GradientMethod
from golden_section import GoldenSection
from dividing_method import DividingMethod
from gradient_projection import  GradientProjectionMethod
from broken_lines_method import BrokenLinesMethod
from tangent_method import TangentMethod
from gradient_conditional import GradientConditionalMethod


def parse_arg_func(arg_dict, n, var_list):
    for i in range(n):
        arg_dict[string.ascii_lowercase[i]] = var_list[i]
    return arg_dict



def parse_args():
    parser = argparse.ArgumentParser('Package for functions minimization')
    parser.add_argument('-f', '--func-expr', required=True, 
                help='function (expression) written using sympy syntax. Note that all arguments should follow pattern of english alphabet: x0: a, x1: b.., xn: z (take a look at few examples as readme.md)')

    parser.add_argument('-n', '--number_of_arguments', type=int, required=True, 
                help='number of arguments of function')

    parser.add_argument('-m', '--method_name', required=True, 
                choices=['gradient_descent', 'gd', 'dividing_segment', 'ds', 
                         'golden_section', 'gs', 'gradient_projection', 'gp',
                         'broken_lines', 'bl', 'tangent', 'tg', 
                         'gradient_conditional', 'gc'], 
                help='method name (check readme.md)')

    parser.add_argument('-mi', '--max_iterations', type=int, default=100,
                help='max number of iterations')

    parser.add_argument('-p','--init_point', nargs='*', type=float, help='Initial point (check readme.md)')

    parser.add_argument('-pp', '--print_points', type=bool, default=False, required=False,
                help='Boolean (print point update at each step)')

    parser.add_argument('-gu', '--gradient_updater', required=False, default='armicho',
                help='method of step updater (only in gradient method))')

    parser.add_argument('-v', '--verbose', type=bool, default=False, required=False, help='visualisation in space of arguments (only for 2d and 3d with gradient methods)')

    parser.add_argument('-fv', '--full_vis', type=bool, default=False, required=False, help='visualisation of point update (only for 1d, 2d and 3d with gradient methods)')

    parser.add_argument('-aname', '--area_name', default='sphere', required=False, help='area name in gradient_projection method. available: \'sphere\',')

    parser.add_argument('-R', '--radius', type=float, required=False, help='sphere radius for --area_name=\'sphere\' in gradient_projection method')

    parser.add_argument('-C', '--center', nargs='*', required=False, type=float, help='sphere center for --area_name=\'sphere\' in gradient_projection method')

    parser.add_argument('-d', '--delta', type=float,

                help='indentation parameter, should be greater than eps (only for dividing_segment method)',
                default=1e-10)
    parser.add_argument('-e', '--eps', type=float,
                help='minimal distance to minumum (only for 1d methods)',
                default=1e-9)
    
    parser.add_argument('-a', '--a', type=float, required=False,
                help='left edge of considered segment')

    parser.add_argument('-b', '--b', type=float, required=False,
                help='right edge of considered segment')

    parser.add_argument('-n_delta', '--n_delta', type=int, default=200, required=False,
                help='number of points in interval')

    

    args, leftovers = parser.parse_known_args()

    if (args.a is None or args.b is None) and \
            args.method_name in ['dividing_segment', 'ds']:
        print("Segment range was not provided")
    elif (args.a is None or args.b is None) and \
            args.method_name in ['golden_section', 'gs']:
        print("Segment range was not provided")
    
    return parser.parse_args()

def create_function(expr, args):
    def functor(x):
        # 1d variant hardcoded
        if type(x) is float or type(x) is np.float64 and args.number_of_arguments == 1:
            arg_dict = dict()
            arg_dict['a'] = x
            return float(expr.evalf(16, arg_dict))
        # n dim
        assert len(x) == args.number_of_arguments, 'wrong dim'
        arg_dict = dict()
        for i in range(args.number_of_arguments):
            arg_dict[string.ascii_lowercase[i]] = x[i]
        return float(expr.evalf(16, arg_dict))
    return functor


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
    # # check
    # print("Some evaluation check")
    # print(expr.evalf(16, subs=arg_dict))
    # print("END")
    f_to_min = create_function(expr, args)

    if args.method_name in ['gradient_descent', 'gd']:
        gr_method = GradientMethod(method_name=args.gradient_updater, 
                    n=args.max_iterations, print_points=args.print_points, 
                    verbose=args.verbose, full_vis=args.full_vis)
        point = gr_method.minimize(f_to_min, args.number_of_arguments, args.init_point)
        if not args.print_points:
            print(f'Founded point: {point}')
    
    elif args.method_name in ['gradient_projection', 'gp']:
        gr_method = GradientProjectionMethod(method_name=args.gradient_updater, 
                    n=args.max_iterations, print_points=args.print_points, 
                    area_name=args.area_name, R=args.radius, C=args.center,
                    verbose=args.verbose, full_vis=args.full_vis)
        point = gr_method.minimize(f_to_min, args.number_of_arguments, args.init_point)
        if not args.print_points:
            print(f'Founded point: {point}')

    elif args.method_name in ['gradient_conditional', 'gc']:
        gc_method = GradientConditionalMethod(method_name=args.gradient_updater, 
                    n=args.max_iterations, print_points=args.print_points, 
                    area_name=args.area_name, R=args.radius, C=args.center,
                    verbose=args.verbose, full_vis=args.full_vis)
        point = gc_method.minimize(f_to_min, args.number_of_arguments, args.init_point)
        if not args.print_points:
            print(f'Founded point: {point}')

    elif args.method_name in ['dividing_segment', 'ds']:
        ds_method = DividingMethod(delta=args.delta, eps=args.eps, n=args.max_iterations, full_vis=args.full_vis)
        point = ds_method.minimize(f_to_min, args.a, args.b)
        print(f'Founded point: {point}')
    
    elif args.method_name in ['golden_section', 'gs']:
        ds_method = GoldenSection(eps=args.eps, n=args.max_iterations, full_vis=args.full_vis)
        point = ds_method.minimize(f_to_min, args.a, args.b)
        print(f'Founded point: {point}')

    elif args.method_name in ['broken_lines', 'bl']:
        bl_method = BrokenLinesMethod(eps=args.eps, m=args.n_delta,  n=args.max_iterations, full_vis=args.full_vis)
        point = bl_method.minimize(f_to_min, a=args.a, b=args.b, x_0=args.init_point[0])
        print(f'Founded point: {point}')

    elif args.method_name in ['tangent', 'tg']:
        tg_method = TangentMethod(eps=args.eps, h=args.delta, m=args.n_delta, n=args.max_iterations, full_vis=args.full_vis)
        point = tg_method.minimize(f_to_min, a=args.a, b=args.b, x_0=args.init_point[0])
        print(f'Founded point: {point}')