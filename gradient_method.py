import numpy as np
import sys
import random
import warnings
from minimizer import Minimizer

class GradientMethod(Minimizer):
    def __init__(self, method_name='armicho', n=500, eps=1e-2, print_points=False):
        '''
        Gradient descent optimizer

        ---------
        Arguments
        method_name: name of method for gradient step update
        n: max_number of iterations
        eps: step for gradient descent

        for method_name available methods: monotone, armicho

        '''
        assert method_name in ['monotone', 'armicho', 'apriori'], "no such method"
        self.method_name = method_name
        self.n = n
        self.eps = eps
        self.points = []
        self.print_points = print_points
    
    def get_abs_grad(self, vector):
        return np.sqrt((vector**2).sum())


    def get_gradient(self, func, x, h=1e-6):
        grad_vector = np.zeros(x.shape).astype('float64')
        for i in range(len(x)):
            add_x = np.copy(x)
            add_x[i] = add_x[i] + h
            grad_vector[i] = (func(add_x) - func(x))/h
        return grad_vector


    def minimize(self, func, k, x0=None):
        '''
        Gradient descent method

        ---------
        Arguments
        func: function to minimize with gradient descent
        k: number of variables
        x0: first approximation point
        '''
        self.points.append(x0)
        if self.print_points:
            print(f"1. {x0}")
        
        method_name = self.method_name
        n = self.n
        eps = self.eps
        if n > 999:
            sys.setrecursionlimit(n+100)

        if x0 is None:
            x0 = np.zeros(k)
            for i in range(k):
                x0[i] = random.random()

        if not type(x0) == np.ndarray:
            x0 = np.array(x0).astype('float64')
        assert len(x0) == k, "x0 wrong size"

        gradient = self.get_gradient(func, x0)

        if method_name == 'monotone':
            x1 = x0 - eps*gradient

            count = 0
            max_count = 20
            while func(x0) < func(x1):
                if count == max_count:
                    warn_str = f"finished on iteration 1 (can not make monotone function decrease)"
                    warnings.warn(warn_str)
                    return x0

                eps /= 2
                x1 = x0 - eps*gradient
                count += 1
        elif method_name == 'armicho':
            eps = 1
            alpha = 1e-1
            lmbd = 1/2
            degree = 2
            while func(x0) - func(x0 - eps*gradient) < alpha*eps*self.get_abs_grad(gradient)**2:
                eps = lmbd**degree
                degree += 1

                if degree > 20:
                    warn_str = f"finished on iteration 1 (can not make monotone function decrease)"
                    warnings.warn(warn_str)
                    return x0

            x1 = x0 - eps*gradient
        elif method_name == 'apriori':
            c = 2/3 
            alpha = 1
            eps = c

            x1 = x0 - eps*gradient

        self.points.append(x1)
        if self.print_points:
            print(f"2. {x1}")

        return self._gradient_method(func, x1, eps, method_name, 1, n)

    def _gradient_method(self, func, xn, eps, method_name, itera, n):
        gradient = self.get_gradient(func, xn)

        if itera == n:
            return xn

        if method_name == 'monotone':
            xn1 = xn - eps*gradient

            count = 0
            max_count = 20
            while func(xn) < func(xn1):
                if count == max_count:
                    warn_str = f"finished on {itera} (can not make monotone function decrease)"
                    warnings.warn(warn_str)
                    return xn

                eps /= 2
                xn1 = xn - eps*gradient
                count += 1
        elif method_name == 'armicho':
            alpha = 1e-1
            lmbd = 1/2
            degree = 2
            while func(xn) - func(xn - eps*gradient) < alpha*eps*self.get_abs_grad(gradient)**2:
                eps = lmbd**degree
                degree += 1

                if degree > 20:
                    warn_str = f"finished on iteration {itera} (can not make monotone function decrease)"
                    warnings.warn(warn_str)
                    return xn

            xn1 = xn - eps*gradient
        elif method_name == 'apriori':
            c = 1/2
            alpha = 1
            eps = c*(itera+1)**(-alpha)

            xn1 = xn - eps*gradient
            
        self.points.append(xn1)
        if self.print_points:
            print(f"{itera+1}. {xn1}")

        return self._gradient_method(func, xn1, eps, method_name, itera + 1, n)

    def get_path(self):
        return self.points
