import numpy as np
from minimizer import Minimizer
import warnings
import sys
import random
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

class GradientProjectionMethod(Minimizer):
    def __init__(self, method_name='armicho', n=500, eps=1e-2, print_points=False, area_name = 'sphere', C=0, R=4, verbose=False):
        '''
        Gradient descent optimizer with projection

        ---------
        Arguments
        method_name: name of method for gradient step update
        n: max_number of iterations
        eps: step for gradient descent
        print_points: print points or not
        area_name: name of considered area
        C: center of sphere (for area_name: 'sphere')
        R: radius of sphere (for area_name: 'sphere')
        verbose: adds some visualisation of process

        for method_name available methods: monotone, armicho

        '''
        assert method_name in ['monotone', 'armicho', 'apriori'], "no such method"
        self.method_name = method_name
        self.n = n
        self.eps = eps
        self.points = []
        self.print_points = print_points
        self.area_name = 'sphere'
        self.R = R
        self.C = C
        self.verbose = verbose
    
    def get_abs_vec(self, vector):
        return np.sqrt((vector**2).sum())

    def is_inside_area(self, x):
        if self.area_name == 'sphere':
            return ((x - self.C)**2).sum() <= self.R**2

    def projection(self, x):
        # check if x is inside our area
        if self.is_inside_area(x):
            w = x
        elif self.area_name == 'sphere':
            w = self.C  + self.R*(x - self.C)/self.get_abs_vec(x - self.C)
        return w


    def get_gradient(self, func, x, h=1e-6):
        grad_vector = np.zeros(x.shape).astype('float64')
        for i in range(len(x)):
            add_x = np.copy(x)
            add_x[i] = add_x[i] + h
            grad_vector[i] = (func(add_x) - func(x))/h
        return grad_vector


    def minimize(self, func, k, x0=None):
        '''
        Gradient descent method with projection

        ---------
        Arguments
        func: function to minimize with gradient descent
        k: number of variables
        x0: first approximation point
        '''
        self.func = func
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
            print(x0 - eps*gradient)
            x1 = self.projection(x0 - eps*gradient)
            print(x1)

            count = 0
            max_count = 20
            while func(x0) < func(x1):
                if count == max_count:
                    warn_str = f"finished on iteration 1 (can not make monotone function decrease)"
                    warnings.warn(warn_str)
                    return x0

                eps /= 2
                x1 = self.projection(x0 - eps*gradient)
                count += 1
        elif method_name == 'armicho':
            eps = 1
            alpha = 1e-1
            lmbd = 1/2
            degree = 2
            while func(x0) - func(self.projection(x0 - eps*gradient)) < alpha*self.get_abs_vec(x0 - self.projection(x0 - eps*gradient))**2:
                eps = lmbd**degree
                degree += 1

                if degree > 20:
                    warn_str = f"finished on iteration 1 (can not satisfty needed criterion)"
                    warnings.warn(warn_str)
                    return x0
            x1 = x0 - eps*gradient
        elif method_name == 'apriori':
            c = 2/3 
            alpha = 1
            eps = c

            x1 = self.projection(x0 - eps*gradient)

        self.points.append(x1)
        if self.print_points:
            print(f"2. {x1}")

        res_point =  self._gradient_method(func, x1, eps, method_name, 1, n)
        if self.verbose:
            self.visualize_results()
        return res_point

    def _gradient_method(self, func, xn, eps, method_name, itera, n):
        gradient = self.get_gradient(func, xn)

        if itera == n:
            return xn

        if method_name == 'monotone':
            xn1 = self.projection(xn - eps*gradient)

            count = 0
            max_count = 20
            while func(xn) < func(xn1):
                if count == max_count:
                    warn_str = f"finished on {itera} (can not make monotone function decrease)"
                    warnings.warn(warn_str)
                    return xn

                eps /= 2
                xn1 = self.projection(xn - eps*gradient)
                count += 1
                
        elif method_name == 'armicho':
            alpha = 1e-1
            lmbd = 1/2
            degree = 2
            while func(xn) - func(self.projection(xn - eps*gradient)) < alpha*self.get_abs_vec(xn - self.projection(xn - eps*gradient))**2:
                eps = lmbd**degree
                degree += 1

                if degree > 20:
                    warn_str = f"finished on iteration {itera} (can not make monotone function decrease)"
                    warnings.warn(warn_str)
                    return xn
            xn1 = self.projection(xn - eps*gradient)
        elif method_name == 'apriori':
            c = 1/2
            alpha = 1
            eps = c*(itera+1)**(-alpha)

            xn1 = self.projection(xn - eps*gradient)
            
        self.points.append(xn1)
        if self.print_points:
            print(f"{itera+1}. {xn1}")

        return self._gradient_method(func, xn1, eps, method_name, itera + 1, n)

    def get_path(self):
        return self.points
    
    def visualize_results(self):
        if not self.verbose:
            return
        if self.area_name == 'sphere' and len(self.C) == 2:
            fig, ax = plt.subplots()
            print('fasfa')
            values = np.linspace(0, 2*np.pi, 500)
            xc = self.C[0]+ self.R*np.cos(values)
            yc = self.C[1] + self.R*np.sin(values)

            f_values = []

            for point in self.points:
                f_values.append(self.func(point))
            self.points = np.array(self.points)

            origin = np.zeros((len(self.C), len(self.points) - 1))

            origin = self.points[:-1, :]
            
            dir_vectors = np.zeros(((len(self.points) - 1), 2))

            for i in range(len(self.points) - 1):
                dir_vectors[i] = self.points[i + 1] - self.points[i]

            xs, ys = origin[:, 0], origin[:, 1]
            x, y = dir_vectors[:, 0], dir_vectors[:, 1]

            plt.plot(xc, yc, label='E0', color='black')
            plt.quiver(xs, ys, x, y, color='red', angles='xy', scale_units='xy', scale=1)
            
            plt.show()
