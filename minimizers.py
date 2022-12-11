import numpy as np
import math
import sys
import random
import warnings


class Minimizer:
    def __init__(self):
        pass


class GoldenSection(Minimizer):
    def __init__(self, eps=1e-10, n=900):
        '''
        Golden section optimizer

        Arguments
        ---------
        eps: minimal distance to minumum
        n: maximum iteration
        '''
        self.eps = eps
        self.n = n
    def _golden_section(self, func, a1, b1, itera, eps, n): 
        x1 = a1 + (3 - math.sqrt(5))/2*(b1-a1) 
        x2 = a1 + (math.sqrt(5) - 1)/2*(b1-a1)

        m_cond = func(x1) <= func(x2)

        if itera == n:
            print("Reached max number of iterations")
            return x1 if m_cond else x2
        elif (b1-a1) < eps:
            return x1 if m_cond else x2
        
        if m_cond:
            a2 = a1
            b2 = x2
            s_x2 = x1
        else:
            a2 = x1
            b2 = b1
            s_x2 = x2
        return self._golden_section(func, a2, b2, itera+1, eps, n)
    def minimize(self, func, a1, b1):
        '''
        Golden section optimizer method

        Arguments
        ---------
        func: unimodal function to minimize
        a1: left edge of considered segment
        b1: right edge of considered segment
        '''
        eps = self.eps
        n = self.n
        x1 = a1 + (3 - math.sqrt(5))/2*(b1-a1)
        x2 = a1 + (math.sqrt(5) - 1)/2*(b1-a1)
        
        if func(x1) <= func(x2):
            a2 = a1
            b2 = x2
            s_x2 = x1
        else:
            a2 = x1
            b2 = b1
            s_x2 = x2
        return self._golden_section(func, a2, b2, 1, eps, n)


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

class GradientMethod(Minimizer):
    def __init__(self, method_name='armicho', n=500, eps=1e-2):
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

        return self._gradient_method(func, xn1, eps, method_name, itera + 1, n)
