import sys
import random
import warnings
from math import ceil

import numpy as np
from minimizer import Minimizer


class SwarmMethod(Minimizer):
    def __init__(self, n=100, a=0.01, b=0.01, iterations=500, tol=1e-4):
        '''
        Swarm Optimizer:

        Arguments
        ---------
        n: number of initial points
        '''
        self.n = n
        self.iterations = iterations
        self.a = a
        self.b = b
        self.tol = tol
        self.itera_thresh = ceil(self.iterations*0.05) + 1
    def minimize(self, func, x0):
        '''
        Swarm method

        ---------
        Arguments
        func: function to minimize with gradient descent
        x0: first approximation point
        '''

        self.xmin = x0[0] - 10*x0[0]
        self.xmax = x0[0] + 10*x0[0]

        self.ymin = x0[1] - 10*x0[1]
        self.ymax = x0[1] + 10*x0[1]
        
        xs = np.random.uniform([self.xmin, self.ymin], [self.xmax, self.ymax], (self.n, 2))


        pbest = xs
        fs = func(*xs.T)
        fpbest = fs

        gbest = xs[np.argmin(fs)]
        fgbest = func(*gbest)

        vs = np.random.uniform(0, 1, (self.n, 2))

        itera = 0
        dntch_count = 0

        yield xs, gbest, 'OK'
        
        while itera != self.iterations:
            vs = vs + self.a*random.random() * (pbest-xs) \
                    + self.b*random.random() * (gbest-xs)

            xs = xs + vs

            fs = func(*xs.T)
            best_indices = np.argmin(np.vstack((fpbest, fs)), axis=0)
            pbest = np.concatenate((pbest[None], xs[None]), axis=0) \
                    [(best_indices, np.arange(len(xs)))]

            fpbest = func(*pbest.T)

            best_index = np.argmin(fpbest)
            gbest = pbest[best_index]
            fgbest_new = func(*gbest)
            itera += 1

            if abs(fgbest - fgbest_new) < self.tol:
                dntch_count += 1
            else:
                dntch_count = 0
            fgbest = fgbest_new

            if dntch_count > self.itera_thresh:
                print(f"Global minimum does not change for {self.itera_thresh} iterations")
                print(f"Stopped at {itera} epoch")
                break

            yield xs, gbest, "OK"

        yield xs, gbest, "END"
