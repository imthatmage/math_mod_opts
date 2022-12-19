import numpy as np
import math

class BrokenLinesMethod():
    def __init__(self, eps=1e-9, n=100):
        '''
        Broken lines optimizer

        Arguments
        ---------
        eps: minimal distance to minumum
        n: maximum number of iterations
        '''
        self.eps = eps
        self.n = n

    def get_L(self, func, a, b, n):
        X = np.linspace(a, b, n)
        L_max = -1
        for i in range(len(X) - 1):
            L = math.fabs(func(X[i + 1]) - func(X[i])) / np.fabs(X[i + 1] - X[i])
            if L > L_max:
                L_max = L
        if L == -1:
            print("Not lipschitz function")
        return L_max

    def get_p_i(self, func, x, x_path, i, L):
        if i == 0:    
            return func(x_path[i]) - L * math.fabs(x - x_path[i])
        else:
            g = func(x_path[i]) - L * math.fabs(x - x_path[i])
            return max(g, self.get_p_i(func, x, x_path, i - 1, L))

    def get_x_new(self, func, a, b, n, x_path, L, eps):
        X = np.linspace(a, b, n)
        p_i_min = 1e8
        x_new = x_path[-1]
        for i in range(len(X)):
            curr_p_i = self.get_p_i(func, X[i], x_path, len(x_path) - 1, L)
            if (curr_p_i - p_i_min) < eps:
                p_i_min = curr_p_i
                x_new = X[i]
        return x_new

    def _broken_lines_method(self, func, a, b, n, x_0, eps):
        x_path = []
        x_path.append(x_0)
        L = self.get_L(func, a, b, n)
        print("L", round(L, 5))
        for i in range(n):
            x_path.append(self.get_x_new(func, a, b, n, x_path, L, eps))
            percent = (i + 1) / n * 100
            if int(percent) == percent:
                print(percent, '%', sep='', end='\r')
        return round(x_path[-1], 5)

    def minimize(self, func, a, b, x_0):
        '''
        Broken lines optimizer method

        Arguments
        ---------
        func: Lipschitz function to minimize
        a: left edge of considered segment
        b: right edge of considered segment
        '''
        eps = self.eps
        n = self.n
        x = self._broken_lines_method(func, a, b, n, x_0, eps)
        return x