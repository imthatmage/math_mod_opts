import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

class BrokenLinesMethod():
    def __init__(self, eps=1e-9):
        '''
        Broken lines optimizer

        Arguments
        ---------
        eps: minimal distance to minumum
        '''
        self.eps = eps

    def get_L(self, func, a, b, m):
        X = np.linspace(a, b, m)
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

    def get_x_new(self, func, a, b, m, x_path, L, eps):
        X = np.linspace(a, b, m)
        p_i_min = 1e8
        x_new = x_path[-1]
        for i in range(m):
            curr_p_i = self.get_p_i(func, X[i], x_path, len(x_path) - 1, L)
            if (curr_p_i - p_i_min) < eps:
                p_i_min = curr_p_i
                x_new = X[i]
        return x_new

    def _broken_lines_method(self, func, a, b, m, x_0, n, eps):
        X = np.linspace(a, b, m)
        x_path = []
        x_path.append(x_0)
        L = self.get_L(func, a, b, m)
        print("L", round(L, 5))
        for i in range(n):
            x_path.append(self.get_x_new(func, a, b, m, x_path, L, eps))
            if(x_path[len(x_path) - 1] == x_path[len(x_path) - 2]):
                break
            percent = (i + 1) / n * 100
            if int(percent) == percent:
                print(percent, '%', sep='', end='\r')
        print("Saveing visualization")
        self.visualize_results(func, a, b, m, x_path)
        return round(x_path[-1], 5)

    def minimize(self, func, a, b, m, x_0, n):
        '''
        Broken lines optimizer method

        Arguments
        ---------
        func: lipschitz function to minimize
        a: left edge of considered segment
        b: right edge of considered segment
        m: number of points in interval
        x_0: starting point
        n: number of iterations
        '''
        eps = self.eps
        x = self._broken_lines_method(func, a, b, m, x_0, n, eps)
        return x


    def visualize_results(self, func, a, b, m, x_path):
        xlist = np.linspace(a, b, m)
        ylist = [func(x) for x in xlist]
        
        fig = plt.figure()
        plt.xlim((min(xlist) - 1, max(xlist) + 1))
        plt.ylim((min(ylist) - 10, max(ylist)))
        l, = plt.plot([], [],  marker="o", markersize=4, c="red")

        metadata = dict(title="Movie")
        writer = PillowWriter(fps=2, metadata=metadata)

        with writer.saving(fig, "Broken lines method.gif", dpi=300):
            plt.plot(xlist, ylist, c='b')
            for i in range(len(x_path)):
                plt.title(f"{i} iteration")
                l.set_data(x_path[i], func(x_path[i]))

                L = self.get_L(func, a, b, m)
                y_new = [self.get_p_i(func, x, x_path, i, L) for x in xlist]
                plt.plot(xlist, y_new, c="black", linewidth=1)

                writer.grab_frame()

        