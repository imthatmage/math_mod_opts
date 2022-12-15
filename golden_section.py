import math
from minimizer import Minimizer

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
