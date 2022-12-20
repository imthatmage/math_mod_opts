import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import numpy as np
import math
from minimizer import Minimizer

class GoldenSection(Minimizer):
    def __init__(self, eps=1e-10, n=900, x_path = []):
        '''
        Golden section optimizer

        Arguments
        ---------
        eps: minimal distance to minumum
        n: maximum iteration
        x_path: list of pairs [x1_i, x2_i], where x1_i and x2_i is a points of golden ratio on i-th iteration
        '''
        self.eps = eps
        self.n = n
        self.x_path = x_path

    def _golden_section(self, func, a1, b1, itera, eps, n, x_path): 
        x1 = a1 + (3 - math.sqrt(5))/2*(b1-a1) 
        x2 = a1 + (math.sqrt(5) - 1)/2*(b1-a1)
        x_path.append([x1, x2])
    
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
        return self._golden_section(func, a2, b2, itera+1, eps, n, x_path)

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
        x_path = self.x_path
        x1 = a1 + (3 - math.sqrt(5))/2*(b1-a1)
        x2 = a1 + (math.sqrt(5) - 1)/2*(b1-a1)
        x_path.append([x1, x2])

        if func(x1) <= func(x2):
            a2 = a1
            b2 = x2
            s_x2 = x1
        else:
            a2 = x1
            b2 = b1
            s_x2 = x2

        x = self._golden_section(func, a2, b2, 1, eps, n, x_path)
        self.visualize_results(func, a1, b1, x_path)
        return x 

    def visualize_results(self, func, a, b, x_path):
        xlist = np.linspace(a, b, 1000)
        ylist = [func(x) for x in xlist]
        
        fig = plt.figure()
        plt.xlim((min(xlist) - 1, max(xlist) + 1))
        plt.ylim((min(ylist) - 10, max(ylist)))
        l1, = plt.plot([], [],  marker="o", markersize=5, c="red")
        l2, = plt.plot([], [],  marker="o", markersize=5, c="green")

        metadata = dict(title="Movie")
        writer = PillowWriter(fps=2, metadata=metadata)

        with writer.saving(fig, "Golden section method.gif", dpi=300):
            plt.plot(xlist, ylist, c='b')
            for i in range(len(x_path)):
                plt.title(f"{i} iteration")
                l1.set_data(x_path[i][0], func(x_path[i][0]))
                l2.set_data(x_path[i][1], func(x_path[i][1]))
                writer.grab_frame()    
