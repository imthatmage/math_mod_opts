import sys
import random
import warnings
from math import ceil

import numpy as np
from minimizer import Minimizer


class SwarmMethod(Minimizer):
    def __init__(self, n=100, n_args=2, a=0.1, b=0.1, iterations=500, 
                 itera_thresh=None, tol=1e-4, 
                 optim='classic', options=[], shift_x=45, shift_y=45):
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
        self.options = options
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.itera = 0
        self.eps_x = 2
        self.eps_y = 2
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

        dntch_count = 0

        # genetic
        reel = 1500

        yield xs, gbest, 'OK'

        weights = np.flip(np.linspace(0, 1, self.iterations))
        self.fgbest_list = []
        
        while self.itera != self.iterations:
            if "inertia" in self.options:
                w = weights[self.itera]
            elif "leader" in self.options:
                percentile = random.randint(75, 90)
                f_thresh = np.percentile(fpbest, percentile)
                best_indices = np.where(fpbest_new <= f_thresh)[0]
                vs[best_indices] *= 1.2
            elif "fading" in self.options:
                percentile = random.randint(75, 90)
                f_thresh = np.percentile(fpbest, percentile)
                worst_indices = np.where(fpbest_new > f_thresh)[0]
                vs[worst_indices] *= 0.9 
            else:
                w = 1
            
            if self.optim == 'classic':
                vs = w*vs + self.a*random.random() * (pbest-xs) \
                          + self.b*random.random() * (gbest-xs)
            elif self.optim == 'annealing':
                vs = w*vs + self.a*random.random() * (pbest-xs) \
                          + self.b*random.random() * (gbest-xs) 
            elif self.optim == 'extinction':
                vs = w*vs + self.a*random.random() * (pbest-xs) \
                          + self.b*random.random() * (gbest-xs)
            elif self.optim == 'evolution':
                vs = w*vs + self.a*random.random() * (pbest-xs) \
                          + self.b*random.random() * (gbest-xs)
            elif self.optim == 'genetic':
                vs = w*vs + self.a*random.random() * (pbest-xs) \
                          + self.b*random.random() * (gbest-xs)

            if 'reflection' in self.options:
                xs = xs + vs
                extend_indices = np.where((xs[:, 0] < self.xmin+self.eps_x) | (xs[:, 0] > self.xmax-self.eps_x))[0]
                vs[extend_indices, 0] = -vs[extend_indices, 0]

                extend_indices = np.where((xs[:, 1] < self.ymin+self.eps_y) | (xs[:, 1] > self.ymax-self.eps_y))[0]
                vs[extend_indices, 1] = -vs[extend_indices, 1] 

                xs = xs + vs
            else:
                xs = xs + vs

            fs = func(*xs.T)
            best_indices = np.argmin(np.vstack((fpbest, fs)), axis=0)
            pbest_new = np.concatenate((pbest[None], xs[None]), axis=0) \
                    [(best_indices, np.arange(len(xs)))]

            fpbest_new = func(*pbest_new.T)

            best_index = np.argmin(fpbest_new)
            gbest_new = pbest_new[best_index]
            fgbest_new = func(*gbest_new)

            self.itera += 1

            if self.optim == 'annealing':
                E = np.exp(-(fpbest_new-fgbest) / fpbest_new)
                tau = random.random()*((self.iterations - self.itera) / self.iterations)
                best_indices = (E > tau).astype('int32')
                pbest = np.concatenate((pbest[None], pbest_new[None]), axis=0) \
                        [(best_indices, np.arange(len(xs)))]
                fpbest = func(*pbest.T)
                best_index = np.argmin(fpbest)
                gbest_new = pbest[best_index]
                fgbest_new = func(*gbest_new)
                
            elif self.optim == 'extinction':
                if self.itera % self.mutation_thresh == 0:
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
                # gbest = gbest_new
                # fgbest = fgbest_new
                # 
            elif self.optim == 'evolution':
                if self.itera % self.mutation_thresh == 0:
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
                # gbest = gbest_new
                # fgbest = fgbest_new
            elif self.optim == 'genetic':
                print(self.itera, reel)
                if self.itera == 1:
                    reel = random.randint(1, self.mutation_thresh)
                elif self.itera % reel == 0:
                    percentile = random.randint(75, 90)
                    f_thresh = np.percentile(fpbest, percentile)
                    best_indices = np.where(fpbest_new <= f_thresh)[0]
                    worst_indices = np.where(fpbest_new > f_thresh)[0]
                    n_pairs = len(worst_indices)
                    n_new = random.randint(n_pairs, 2*n_pairs)
                    n = len(best_indices)
                    # 1d fictitious 2d array
                    pairs_indices = np.random.randint(0, n, 2*n_new)
                    gen_pairs = pbest_new[pairs_indices].reshape(2, n_new, self.n_args)
                    gen_points = gen_pairs.sum(axis=0)/2

                    # delete worst points
                    pbest_new = np.delete(pbest_new, worst_indices, axis=0)
                    xs = np.delete(xs, worst_indices, axis=0)
                    vs = np.delete(vs, worst_indices, axis=0)
                    # add new points
                    pbest_new = np.concatenate((pbest_new, gen_points), axis=0)
                    xs = np.concatenate((xs, gen_points), axis=0)
                    vs = np.concatenate((vs, np.random.uniform(-1, 1, gen_points.shape)), axis=0)

                    self.n = len(xs)

                    pbest = pbest_new
                    fpbest = func(*pbest.T)
                    best_index = np.argmin(fpbest)
                    gbest_new = pbest[best_index]
                    fgbest_new = func(*gbest_new)
                    reel = random.randint(1, self.mutation_thresh)
                else:
                    pbest = pbest_new
                    fpbest = fpbest_new
            else:
                pbest = pbest_new
                fpbest = fpbest_new

            if abs(fgbest - fgbest_new) < self.tol:
                dntch_count += 1
            else:
                dntch_count = 0
            if dntch_count > self.itera_thresh:
                print(f"Global minimum does not change for {dntch_count} iterations")
                print(f"Stopped at {self.itera} epoch")
                break
            gbest = gbest_new
            fgbest = fgbest_new
            self.fgbest_list.append(fgbest)
            

            yield xs, gbest, "OK"

        yield xs, gbest, "END"
