import sys
import random
import warnings
from math import ceil

import numpy as np
from minimizer import Minimizer


class SwarmMethod(Minimizer):
    def __init__(self, n=100, n_args=2, a=0.9, b=0.1, iterations=500, 
                 itera_thresh=None, tol=1e-4, 
                 optim='classic', shift_x=45, shift_y=45):
        '''
        Swarm Optimizer:

        Arguments
        ---------
        n: number of initial points
        '''
        self.n = n
        self.n_args = n_args
        self.iterations = iterations
        self.a = a
        self.b = b
        self.tol = tol
        self.itera_thresh = ceil(self.iterations*0.05)+1 if itera_thresh is None else itera_thresh
        self.mutation_thresh = self.itera_thresh//5
        self.optim = optim
        self.shift_x = shift_x
        self.shift_y = shift_y
    def minimize(self, func, x0):
        '''
        Swarm method

        ---------
        Arguments
        func: function to minimize with gradient descent
        x0: first approximation point
        '''
        self.xmin = x0[0] - self.shift_x
        self.xmax = x0[0] + self.shift_x

        self.ymin = x0[1] - self.shift_y
        self.ymax = x0[1] + self.shift_y
        
        xs = np.random.uniform([self.xmin, self.ymin], [self.xmax, self.ymax], (self.n, self.n_args))


        pbest = xs
        fs = func(*xs.T)
        fpbest = fs

        gbest = xs[np.argmin(fs)]
        fgbest = func(*gbest)

        vs = np.random.uniform(-1, 1, (self.n, self.n_args))

        itera = 0
        dntch_count = 0

        yield xs, gbest, 'OK'

        weights = np.flip(np.linspace(0, 1, self.iterations))
        
        while itera != self.iterations:
            if self.optim == 'classic':
                vs = vs + self.a*random.random() * (pbest-xs) \
                        + self.b*random.random() * (gbest-xs)
            elif self.optim == 'inertia':
                vs = weights[itera]*vs + self.a*random.random() * (pbest-xs) \
                        + self.b*random.random() * (gbest-xs) 
            elif self.optim == 'annealing':
                vs = vs + self.a*random.random() * (pbest-xs) \
                        + self.b*random.random() * (gbest-xs) 
            elif self.optim == 'extinction':
                vs = vs + self.a*random.random() * (pbest-xs) \
                        + self.b*random.random() * (gbest-xs) 
            elif self.optim == 'evolution':
                vs = vs + self.a*random.random() * (pbest-xs) \
                        + self.b*random.random() * (gbest-xs) 

            xs = xs + vs

            fs = func(*xs.T)
            best_indices = np.argmin(np.vstack((fpbest, fs)), axis=0)
            pbest_new = np.concatenate((pbest[None], xs[None]), axis=0) \
                    [(best_indices, np.arange(len(xs)))]

            fpbest_new = func(*pbest_new.T)

            best_index = np.argmin(fpbest_new)
            gbest_new = pbest_new[best_index]
            fgbest_new = func(*gbest_new)

            itera += 1

            if self.optim == 'annealing':
                E = np.exp(-(fpbest_new-fgbest) / fpbest_new)
                tau = random.random()*((self.iterations - itera) / self.iterations)
                best_indices = (E > tau).astype('int32')
                pbest = np.concatenate((pbest[None], pbest_new[None]), axis=0) \
                        [(best_indices, np.arange(len(xs)))]
                fpbest = func(*pbest.T)
                best_index = np.argmin(fpbest)
                gbest = pbest[best_index]
                fgbest = func(*gbest)
            elif self.optim == 'extinction':
                if itera % self.mutation_thresh == 0:
                    f_thresh = np.percentile(fpbest, 90)
                    mask = fpbest_new <= f_thresh
                    self.n = mask.sum()
                    pbest = pbest_new[np.repeat(mask[:, None], self.n_args, axis=1)] \
                            .reshape(self.n, self.n_args)
                    xs = xs[np.repeat(mask[:, None], self.n_args, axis=1)] \
                            .reshape(self.n, self.n_args)
                    vs = vs[np.repeat(mask[:, None], self.n_args, axis=1)] \
                            .reshape(self.n, self.n_args)
                    fpbest = func(*pbest.T)
                else:
                    pbest = pbest_new
                    fpbest = fpbest_new
                gbest = gbest_new
                fgbest = fgbest_new
            elif self.optim == 'evolution':
                if itera % self.mutation_thresh == 0:
                    f_thresh = np.percentile(fpbest, 90)
                    mask = fpbest_new <= f_thresh
                    tmp_n = mask.sum()
                    top_pbest = pbest_new[np.repeat(mask[:, None], self.n_args, axis=1)] \
                            .reshape(tmp_n, self.n_args)
                    rand_indices = np.random.randint(tmp_n, size=self.n-tmp_n)
                    pbest[np.repeat(np.logical_not(mask[:, None]), \
                            self.n_args, axis=1)] = top_pbest[rand_indices].reshape(-1)
                    xs[np.repeat(np.logical_not(mask[:, None]), \
                            self.n_args, axis=1)] = top_pbest[rand_indices].reshape(-1)
                    vs[np.repeat(np.logical_not(mask[:, None]), \
                                    self.n_args, axis=1)] = np.random.uniform(-1, 1, ((self.n-tmp_n)*self.n_args))
                    fpbest = func(*pbest.T)
                else:
                    pbest = pbest_new
                    fpbest = fpbest_new
                gbest = gbest_new
                fgbest = fgbest_new
            else:
                pbest = pbest_new
                fpbest = fpbest_new
                gbest = gbest_new
                fgbest = fgbest_new

            if abs(fgbest - fgbest_new) < self.tol:
                dntch_count += 1
            else:
                dntch_count = 0
            if dntch_count > self.itera_thresh:
                print(f"Global minimum does not change for {self.itera_thresh} iterations")
                print(f"Stopped at {itera} epoch")
                break

            fgbest = fgbest_new

            yield xs, gbest, "OK"

        yield xs, gbest, "END"
