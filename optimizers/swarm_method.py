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
        self.eps_x = 0.001
        self.eps_y = 0.001

        self.m_cat = int(self.n*0.1)
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
        
        self.gbest_list = []
        self.gbest_list.append(gbest)
        self.fgbest_list = []
        self.fgbest_list.append(fgbest)
        self.xs_list = []
        self.xs_list.append(xs)
        
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
                if self.optim != 'cat':
                    xs_tmp = xs + vs
                else:
                    xs_tmp = xs
                extend_indices_x = np.where((xs_tmp[:, 0] < self.xmin+self.eps_x) | (xs_tmp[:, 0] > self.xmax-self.eps_x))[0]
                extend_indices_init_x = extend_indices_x.copy()
                extend_indices_y = np.where((xs_tmp[:, 1] < self.ymin+self.eps_y) | (xs_tmp[:, 1] > self.ymax-self.eps_y))[0]
                extend_indices_init_y = extend_indices_y.copy()
                extend_indices = np.hstack((extend_indices_x, extend_indices_y))
                extend_indices_init = extend_indices.copy()

                step = 0.5
                print("Reflect start")
                while extend_indices.sum() > 0:
                    xs_tmp = xs + step*vs
                    vs *= step
                    extend_indices_x = np.where((xs_tmp[:, 0] < self.xmin+self.eps_x) | (xs_tmp[:, 0] > self.xmax-self.eps_x))[0]
                    extend_indices_y = np.where((xs_tmp[:, 1] < self.ymin+self.eps_y) | (xs_tmp[:, 1] > self.ymax-self.eps_y))[0]
                    extend_indices = np.hstack((extend_indices_x, extend_indices_y))
                    step *= 0.5

                    if step < 1e-6:
                        break

                print("Reflect end")
                vs[extend_indices_init] = -vs[extend_indices_init]

                xs = xs_tmp
            else:
                if self.optim != 'cat':
                    xs = xs + vs
                else:
                    xs = xs

            fs = func(*xs.T)
            best_indices = np.argmin(np.vstack((fpbest, fs)), axis=0)
            pbest_new = np.concatenate((pbest[None], xs[None]), axis=0) \
                    [(best_indices, np.arange(len(xs)))]

            fpbest_new = func(*pbest_new.T)

            best_index = np.argmin(fpbest_new)
            gbest_new = pbest_new[best_index]
            fgbest_new = func(*gbest_new)

            self.itera += 1

            if self.optim == 'cat':
                worst_index = np.argmax(fpbest_new)
                worst_new = pbest_new[worst_index]
                fgworst_new = func(*worst_new)
                if self.itera == 1:
                    seeking_mask = np.zeros(self.n)
                    trace_indices = np.where(seeking_mask == 0)[0]
                    seek_trace_indices = np.random.choice(trace_indices, 
                                                     int(0.1*len(trace_indices)))
                    seeking_mask[seek_trace_indices] = 1
                    # NxMxN_ARGS
                    seek_points = np.repeat(xs[:, None], self.m_cat, axis=1)
                    pbest_seek = np.repeat(pbest_new[:, None], self.m_cat, axis=1)
                    # NxM
                    fpbest_seek = np.repeat(fpbest_new[:, None], self.m_cat, axis=1)


                seeking_indices = np.where(seeking_mask > 0)[0]
                print(f"Seeking_indices len: {len(seeking_indices)}")

                tmp_seek_points = seek_points.copy()
                tmp_seek_points[seeking_indices] += np.random.normal \
                        (0, 1, (len(seeking_indices), self.m_cat, self.n_args))
                if len(seeking_indices) != 0:
                    # acception rejection (if we outside of our bounding box)
                    if 'reflection' in self.options:
                        extend_indices_x = np.where((tmp_seek_points[seeking_indices][:, 0] < self.xmin+self.eps_x) | (tmp_seek_points[seeking_indices][:, 0] > self.xmax-self.eps_x))[0]
                        extend_indices_y = np.where((tmp_seek_points[seeking_indices][:, 1] < self.ymin+self.eps_y) | (tmp_seek_points[seeking_indices][:, 1] > self.ymax-self.eps_y))[0]
                        extend_indices = np.hstack((extend_indices_x, extend_indices_y))

                        print("Start: ")
                        counts = 0
                        while extend_indices.sum() > 0:
                            tmp_seek_points[seeking_indices][extend_indices] = seek_points[seeking_indices][extend_indices] + np.random.normal \
                                    (0, 1, (len(extend_indices), self.m_cat, self.n_args))
                            extend_indices_x = np.where((tmp_seek_points[seeking_indices][:, 0] < self.xmin+self.eps_x) | (tmp_seek_points[seeking_indices][:, 0] > self.xmax-self.eps_x))[0]
                            extend_indices_y = np.where((tmp_seek_points[seeking_indices][:, 1] < self.ymin+self.eps_y) | (tmp_seek_points[seeking_indices][:, 1] > self.ymax-self.eps_y))[0]
                            extend_indices = np.hstack((extend_indices_x, extend_indices_y))
                            counts += 1
                            if counts == 30:
                                break
                        print("End: ")

                    seek_points = tmp_seek_points
                    # N_SEEK_INDICESxM
                    fseek = func(*seek_points[seeking_indices].T).T

                    seek_check_both = np.concatenate((fpbest_seek[seeking_indices][..., None], fseek[..., None]), axis=-1)
                    seek_point_both = np.concatenate((pbest_seek[seeking_indices][:, None], seek_points[seeking_indices][:, None]), axis=1)
                    seek_best_indices = np.argmin(seek_check_both, axis=-1).reshape(-1)
                    tmp_seek_n = len(fseek.reshape(-1))
                    fpbest_seek[seeking_indices] = seek_check_both.reshape(-1, 2)[(np.arange(tmp_seek_n), seek_best_indices)] \
                            .reshape(fseek.shape[0], fseek.shape[1])
                    pbest_seek[seeking_indices] = seek_point_both.reshape(-1, 2, self.n_args)[(np.arange(tmp_seek_n), seek_best_indices)] \
                            .reshape(fseek.shape[0], fseek.shape[1], self.n_args)

                    seek_probs = np.clip(((fgworst_new - fpbest_seek[seeking_indices]))/(fgworst_new - fgbest_new), 0, 1)

                    real_probs = np.random.uniform(0, 1, (len(seeking_indices), self.m_cat))

                    update_seek_mask = (seek_probs - real_probs)
                    update_seek_indices = np.argmax(update_seek_mask, axis=1)
                    pos_mask = update_seek_mask.max(axis=1) > 0
                    update_seek_indices = update_seek_indices[pos_mask]
                    x_indices = np.arange(len(seeking_indices))[pos_mask]

                    seek_update_points = pbest_seek[(x_indices, update_seek_indices)]

                    xs[seeking_indices[pos_mask]] = seek_update_points
                else:
                    pos_mask = np.zeros(len(seeking_indices)).astype('bool')


                trace_indices = np.where(seeking_mask == 0)[0]

                vs[trace_indices] = w*vs[trace_indices] \
                    + self.a*random.random() * (pbest_new[trace_indices]-xs[trace_indices]) \
                    + self.b*random.random() * (gbest_new-xs[trace_indices])
                xs[trace_indices] += vs[trace_indices]

                exclude_indices = seeking_indices[pos_mask]
                seeking_mask[exclude_indices] = 0
                trace_indices = np.where(seeking_mask == 0)[0]

                seek_trace_indices = np.random.choice(trace_indices, 
                                                 int(0.1*len(trace_indices)))
                # update previous points
                seeking_mask[seeking_mask > 0] += 1
                # kick stacked points
                seeking_mask[seeking_mask > 6] = 0

                seeking_mask[seek_trace_indices] = 1

                seeking_indices = np.where(seeking_mask > 0)[0]
                seek_points[seeking_indices] = np.repeat(xs[seeking_indices][:, None], self.m_cat, axis=1)

                pbest = pbest_new
                fpbest = fpbest_new


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
            
            self.gbest_list.append(gbest)
            self.fgbest_list.append(fgbest)
            self.xs_list.append(xs)
            
            yield xs, gbest, "OK"

        yield xs, gbest, "END"
