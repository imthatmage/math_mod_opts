import numpy as np
import sys
import random
import warnings
from minimizer import Minimizer
import matplotlib.pyplot as plt
from tqdm import tqdm

class GradientConditionalMethod(Minimizer):
    def __init__(self, method_name='armicho', n=500, eps=1e-2, print_points=False, 
                 area_name = 'sphere', C=0, R=4, verbose=False, full_vis=False):
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
        self.full_vis = full_vis
    
    def get_abs_vec(self, vector):
        return np.sqrt((vector**2).sum())


    def get_gradient(self, func, x, h=1e-6):
        grad_vector = np.zeros(x.shape).astype('float64')
        for i in range(len(x)):
            add_x = np.copy(x)
            add_x[i] = add_x[i] + h
            grad_vector[i] = (func(add_x) - func(x))/h
        return grad_vector

    
    def  get_x_int(self, gradient, xk):
        return self.C - self.R*gradient/self.get_abs_vec(gradient)


    def minimize(self, func, k, x0=None):
        '''
        Gradient descent method

        ---------
        Arguments
        func: function to minimize with gradient descent
        k: number of variables
        x0: first approximation point
        '''
        self.k = k
        self.func = func
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
        x_int = self.get_x_int(gradient, x0)

        if method_name == 'monotone':
            x1 = x0 + eps*(x_int - x0)

            count = 0
            max_count = 20
            while func(x0) < func(x1):
                if count == max_count:
                    warn_str = f"finished on 1 (can not make monotone function decrease)"
                    warnings.warn(warn_str)
                    return x1

                eps /= 2
                x1 = x0 + eps*(x_int - x0)
                count += 1
        elif method_name == 'armicho':
            lmbd = 0.5
            alpha = 0.5

            grad_scalar = (gradient*(x_int - x0)).sum()
            if grad_scalar > 0:
                eps = 0
                x1 = x0 + eps*(x_int - x0)
            else:
                for i in range(30 + 1):
                    if self.func(x0) - self.func(x0 + lmbd**i*(x_int - x0)) >= lmbd**i*alpha*abs(grad_scalar):
                        eps = lmbd**i
                        break
                    elif i == 30:
                        warn_str = f"finished on 1 (can not find appropriate condition for step updater)"
                        warnings.warn(warn_str)
                        return x0
                x1 = x0 + eps*(x_int - x0)

        elif method_name == 'apriori':
            c = 2/3 
            alpha = 1
            eps = c

            x1 = x0 + eps*(x_int - x0)

        self.points.append(x1)
        if self.print_points:
            print(f"2. {x1}")

        res_point = self._gradient_method(func, x1, eps, method_name, 1, n)

        if self.verbose:
            self.visualize_results()

        return res_point

    def _gradient_method(self, func, xn, eps, method_name, itera, n):
        gradient = self.get_gradient(func, xn)
        x_int = self.get_x_int(gradient, xn)

        if itera == n:
            return xn

        if method_name == 'monotone':
            xn1 = xn + eps*(x_int - xn)

            count = 0
            max_count = 20
            while func(xn) < func(xn1):
                if count == max_count:
                    warn_str = f"finished on {itera} (can not make monotone function decrease)"
                    warnings.warn(warn_str)
                    return xn

                eps /= 2
                xn1 = xn + eps*(x_int - xn)
                count += 1
        elif method_name == 'armicho':
            lmbd = 0.5
            alpha = 0.5

            grad_scalar = (gradient*(x_int - xn)).sum()
            if grad_scalar > 0:
                eps = 0
                xn1 = xn + eps*(x_int - xn)
            else:
                for i in range(1, 30 + 1):
                    if self.func(xn) - self.func(xn + lmbd**i*(x_int - xn)) >= lmbd**i*alpha*abs(grad_scalar):
                        eps = lmbd**i
                        break
                    elif i == 30:
                        warn_str = f"finished on {itera} (can not find appropriate condition for step updater)"
                        warnings.warn(warn_str)
                        return xn
                xn1 = xn + eps*(x_int - xn)
        elif method_name == 'apriori':
            c = 1/2
            alpha = 1
            eps = c*(itera+1)**(-alpha)

            xn1 = xn + eps*(x_int - xn)
            
        self.points.append(xn1)
        if self.print_points:
            print(f"{itera+1}. {xn1}")


        return self._gradient_method(func, xn1, eps, method_name, itera + 1, n)

    def get_path(self):
        return self.points
    
    def visualize_results(self):
        if not self.verbose:
            return
        if self.full_vis and self.k == 2:
            self.points = np.array(self.points)
            import plotly.graph_objects as go
            layout = go.Layout(width = 700, height =700, title_text='Chasing global Minima')
            # visualize function
            left_point = self.points[0] - (self.points[-1] - self.points[0])/2
            right_point = self.points[-1] + (self.points[-1] - self.points[0])/2

            n_dots = 100

            X = np.linspace(left_point[0], right_point[0], n_dots)
            Y = np.linspace(left_point[1], right_point[1], n_dots)

            Z = np.zeros((X.shape[0], Y.shape[0]))

            for i in tqdm(range(X.shape[0])):
                for j in range(Y.shape[0]):
                    Z[j, i] = self.func([X[i], Y[j]])
            # show path
            f_values = []

            for point in self.points:
                f_values.append(self.func(point))

            f_values = np.array(f_values)[:, None]

            or_vectors = np.concatenate((self.points[1:], f_values[1:]), axis=-1)
            dir_vectors = np.zeros((self.points.shape[0], 3))

            for i in range(len(or_vectors) - 1):
                dir_vectors[i] = 10*(or_vectors[i+1] - or_vectors[i])
            
            fig = go.Figure(data=[go.Cone(
                x=or_vectors[:, 0],
                y=or_vectors[:, 1],
                z=or_vectors[:, 2],
                u=dir_vectors[:, 0],
                v=dir_vectors[:, 1],
                w=dir_vectors[:, 2],
                sizemode="absolute", colorscale='Viridis'), 
                go.Surface(x = X, y = Y, z=Z, colorscale='Blues')], layout=layout)

            fig.update_layout(
                scene=dict(domain_x=[0, 1],
                            camera_eye=dict(x=-1.57, y=1.36, z=0.58)))

            fig.show()
        elif self.k == 2:
            fig, ax = plt.subplots()

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

            plt.quiver(xs, ys, x, y, color='red', angles='xy', scale_units='xy', scale=1)
            
            plt.show()
        

        if self.k == 3:
            self.points = np.array(self.points)
            import plotly.graph_objects as go
            layout = go.Layout(width = 700, height =700, title_text='Chasing global Minima')
            # show path
            or_vectors = self.points
            dir_vectors = np.zeros((self.points.shape[0], 3))

            for i in range(len(or_vectors) - 1):
                dir_vectors[i] = 10*(or_vectors[i+1] - or_vectors[i])

            f_values = []

            for point in self.points:
                f_values.append(self.func(point))
            
            fig = go.Figure(data=[go.Cone(
                x=or_vectors[:, 0],
                y=or_vectors[:, 1],
                z=or_vectors[:, 2],
                u=dir_vectors[:, 0],
                v=dir_vectors[:, 1],
                w=dir_vectors[:, 2],
                sizemode="absolute")], layout=layout)

            fig.update_layout(
                scene=dict(domain_x=[0, 1],
                            camera_eye=dict(x=-1.57, y=1.36, z=0.58)))

            fig.show()