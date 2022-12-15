
from minimizer import Minimizer

class DividingMethod(Minimizer):
    def __init__(self, delta=1e-10, eps=1e-9, n=900):
        '''
        Segment dividing optimizer

        Arguments
        ---------
        delta: indentation parameter
        eps: minimal distance to minumum, should be less than delta
        n: maximum number of iterations
        '''
        self.delta = delta
        self.eps = eps
        self.n = n

    def _dividing_method(self, func, a, b, delta, eps, itera, n):
        if itera == n:
            print("Reached max number of iterations")
            return (a + b) / 2
        elif (b - a) < eps:
            return (a + b) / 2

        x1 = (a + b - delta) / 2
        x2 = (a + b + delta) / 2

        if func(x1) <= func(x2):
            a1 = a
            b1 = x2
        else:
            a1 = x1
            b1 = b
        return self._dividing_method(func, a1, b1, delta, eps, itera+1, n)

    def minimize(self, func, a, b):
        '''
        Segment dividing optimizer method

        Arguments
        ---------
        func: unimodal function to minimize
        a: left edge of considered segment
        b: right edge of considered segment
        '''
        eps = self.eps
        delta = self.delta
        n = self.n
        assert delta < eps, 'delta should be less than eps'
        x = self._dividing_method(func, a, b, delta, eps, 1, n)
        return x
