import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

class TangentMethod():
    def __init__(self, eps=1e-9, h=1e-5):
        '''
        Tangent optimizer

        Arguments
        ---------
        eps: minimal distance to minumum
        h: offset value in the difference method
        '''
        self.eps = eps
        self.h = h

    def get_func_der(self, func, x, h):
        func_der = -(func(x + h) - func(x - h)) / (2 * h)
        return func_der 

    def get_p_i(self, func, x, x_path, i, h):
        if i == 0:    
            return func(x_path[i]) -  self.get_func_der(func, x_path[i], h) * (x - x_path[i])
        else:
            g = func(x_path[i]) - self.get_func_der(func, x_path[i], h) * (x - x_path[i])
            return max(g, self.get_p_i(func, x, x_path, i - 1, h))

    def get_x_new(self, func, a, b, m, x_path, eps, h):
        X = np.linspace(a, b, m)
        p_i_min = 1e8
        x_new = x_path[-1]
        for i in range(m):
            curr_p_i = self.get_p_i(func, X[i], x_path, len(x_path) - 1, h)
            if (curr_p_i - p_i_min) < eps:
                p_i_min = curr_p_i
                x_new = X[i]
        return x_new

    def _tangent_method(self, func, a, b, m, x_0, n, eps, h):
        X = np.linspace(a, b, m)
        x_path = []
        x_path.append(x_0)
        for i in range(n):
            x_path.append(self.get_x_new(func, a, b, m, x_path, eps, h))
            if(x_path[len(x_path) - 1] == x_path[len(x_path) - 2]):
                break
            percent = (i + 1) / n * 100
            if int(percent) == percent:
                print(percent, '%', sep='', end='\r')
        self.visualize_results(func, a, b, m, x_path, h)
        return round(x_path[-1], 5)

    def minimize(self, func, a, b, m, x_0, n):
        '''
        Tangent optimizer method

        Arguments
        ---------
        func: convex function to minimize
        a: left edge of considered segment
        b: right edge of considered segment
        m: number of points in interval
        x_0: starting point
        n: max number of iterations, n <= m
        '''
        eps = self.eps
        h = self.h
        x = self._tangent_method(func, a, b, m, x_0, n, eps, h)
        return x


    def visualize_results(self, func, a, b, m, x_path, h):
        xlist = np.linspace(a, b, m)
        ylist = [func(x) for x in xlist]
        
        fig = plt.figure()
        plt.xlim((min(xlist) - 1, max(xlist) + 1))
        plt.ylim((min(ylist) - 10, max(ylist)))
        l, = plt.plot([], [],  marker="o", markersize=4, c="red")

        metadata = dict(title="Movie")
        writer = PillowWriter(fps=2, metadata=metadata)

        with writer.saving(fig, "Tangent method.gif", dpi=300):
            plt.plot(xlist, ylist, c='b')
            for i in range(len(x_path)):
                plt.title(f"{i} iteration")
                l.set_data(x_path[i], func(x_path[i]))
                
                y_new = [self.get_p_i(func, x, x_path, i, h) for x in xlist]
                plt.plot(xlist, y_new, c="black", linewidth=1)
                
                writer.grab_frame()