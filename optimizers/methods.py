import numpy as np
import random


def broken_curve_method(f, a, b, L, min_f, n):
    '''
    Broken curve method for Lipschitz function

    Arguments
    ---------
    func: Lipschitz function
    a: left edge of considered segment
    b: right edge of considered segment
    L: Lipschitz's constant
    min_f: minimizer function
    '''
    g = lambda x, y: f(y) - L*abs(x-y)

    # chooose x0 with uniform distribution right now
    x0 = random.random()*(b-a) + a
    p0 = lambda x: f(x0) - L*abs(x-x0)
    x1 = min_f(p0, a, b)
    print(x1)

    p1 = lambda x: max(g(x, x1), p0(x)) 
    x2 = min_f(p1, a, b)

    return _broken_curve_method(f, a, b, g, min_f, p1, [x1, x2], 2, n)


def _broken_curve_method(f, a, b, g, min_f, pn_1, xlist, i, n):
    pn = lambda x: max(g(x, xlist[-1]), pn_1(x))
    xn1 = min_f(pn, a, b)

    i += 1
    if i == n:
        return [xn1], [pn(xn1)]

    return _broken_curve_method(f, a, b, g, min_f, pn, [*xlist, xn1], i, n)
