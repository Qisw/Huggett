# -*- coding: utf-8 -*-
"""
Jul. 7, 2015, Hyun Chang Yi
Huggett (1996) "Wealth distribution in life-cycle economies," Journal of Monetary
Economics, 38(3), 469-494.
"""

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize_scalar
from numpy import linspace, mean, array, zeros, absolute, loadtxt, dot, prod, \
                    genfromtxt, sum, argmax, tile, concatenate
from matplotlib import pyplot as plt
from datetime import datetime
import time
import pickle
from multiprocessing import Process, Lock, Manager


class cohort:
    """ This class is just a "struct" to hold the collection of primitives defining
    a generation """
    def __init__(self, beta=0.994, sigma=1.5, aH=50.0, aL=0.0, y=-1,
        aN=51, tol=0.01, neg=-1e10, W=45, R=34, a0 = 0):
        self.beta, self.sigma = beta, sigma
        self.R, self.W, self.y = R, W, y
        # self.mls = mls = (y+1 if (y >= 0) and (y <= W+R-2) else W+R) # mls is maximum life span
        self.aH, self.aL, self.aN, self.aa = aH, aL, aN, aL+aH*linspace(0,1,aN)
        self.tol, self.neg = tol, neg
        self.sp = loadtxt('sp.txt', delimiter='\n')  # survival probability
        self.muz = genfromtxt('muz.csv', delimiter=',')  # initial distribution of productivity
        self.pi = genfromtxt('pi.csv', delimiter=',')  # productivity transition probability
        self.ef = genfromtxt('ef.csv', delimiter=',')
        self.zN = zN = self.pi.shape[0]
        self.mls = mls = self.ef.shape[0]
        """ container for value function and expected value function """
        # v[y,j,i] is the value of an y-yrs-old agent with asset i and productity j
        self.v = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(mls)], dtype=float)
        # ev[y,j,ni] is the expected value when the agent's next period asset is ni
        self.ev = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(mls)], dtype=float)
        """ container for policy functions """
        self.a = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(mls)], dtype=float)
        self.c = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(mls)], dtype=float)
        """ distribution of agents w.r.t. age, productivity and asset
        for each age, distribution over all productivities and assets add up to 1 """
        self.mu = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(mls)], dtype=float)


    def add(self):
        self.R = 10


#병렬처리를 위한 for loop 내 로직 분리
def transition_sub1(c):
    c.add()
    print c.R


if __name__ == '__main__':
    cohorts = [cohort(y=t) for t in range(20)]
    for n in range(1):
        start_time = datetime.now()
        jobs = []
        for i in range(20):
            # transition_sub1(c,k_tp.prices,c_t.mu)
            p = Process(target=transition_sub1, args=(cohorts[i],))
            p.start()
            jobs.append(p)
            #병렬처리 개수 지정 20이면 20개 루프를 동시에 병렬로 처리
            if len(jobs) % 4 == 0:
                for p in jobs:
                    p.join()
                print 'transition('+str(cohorts[i].y)+') is in progress : {}'.format(datetime.now())
                jobs=[]
        if len(jobs) > 0:
            for p in jobs:
                p.join()
        for c in cohorts:
            print c.R
        end_time = datetime.now()
        print 'transition ('+str(n)+') loop: {}'.format(end_time - start_time)


# if __name__ == '__main__':
    # initialize()
    cs = transition()
    print 'nnn'
    transition_sub1(cs[0])
    print cs[0].R
