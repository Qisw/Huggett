# -*- coding: utf-8 -*-
"""
Jul. 7, 2015, Hyun Chang Yi
Huggett (1996) "Wealth distribution in life-cycle economies," Journal of Monetary
Economics, 38(3), 469-494.
"""

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize_scalar
from numpy import linspace, mean, array, zeros, absolute, loadtxt, dot, prod, \
                    genfromtxt, sum, argmax, tile, concatenate, ones, log
from matplotlib import pyplot as plt
from datetime import datetime
import time
import pickle
from multiprocessing import Process, Lock, Manager
from multiprocessing.sharedctypes import Value, Array, RawArray
from ctypes import Structure, c_double


class params:
    """ This class is just a "struct" to hold the collection of PARAMETER values """
    def __init__(self, T=1, alpha=0.36, delta=0.06, tau=0.2378, theta=0.1, zeta=0.3,
        beta=0.994, sigma=1.5, W=45, R=34, a0 = 0, ng_init=1.012, ng_term=1.0-0.012,
        aH=50.0, aL=0.0, aN=50, psi=0.1, phi=0.5, dti=0.5, ltv=0.7, tcost=0.02,
        tol=1e-1, neg=-1e10):
        if T==1:
            ng_term = ng_init
        self.alpha, self.zeta, self.delta, self.tau = alpha, zeta, delta, tau
        self.theta = theta
        self.tcost = tcost
        self.beta, self.sigma = beta, sigma
        self.R, self.W, self.T = R, W, T
        self.aH, self.aL, self.aN, self.aa = aH, aL, aN, aL+aH*linspace(0,1,aN)
        self.phi, self.tol, self.neg = phi, tol, neg
        self.hh = linspace(0,4,5)   # hh = [0, 1, 2, 3, 4]
        self.hN = hN = len(self.hh)
        """ LOAD PARAMETERS : SURVIVAL PROB., PRODUCTIVITY TRANSITION PROB. AND ... """
        self.sp = sp = loadtxt('sp.txt', delimiter='\n')  # survival probability
        self.muz = genfromtxt('muz.csv', delimiter=',')  # initial distribution of productivity
        self.pi = genfromtxt('pi.csv', delimiter=',')  # productivity transition probability
        self.ef = genfromtxt('ef.csv', delimiter=',')
        self.zN = self.pi.shape[0]
        self.mls = self.ef.shape[0]
        """ CALCULATE POPULATIONS OVER THE TRANSITION PATH """
        m0 = array([prod(sp[:y+1])/ng_init**y for y in range(self.mls)], dtype=float)
        m1 = array([prod(sp[:y+1])/ng_term**y for y in range(self.mls)], dtype=float)
        self.pop = array([m1*ng_term**t for t in range(T)], dtype=float)
        for t in range(min(T,self.mls-1)):
            self.pop[t,t+1:] = m0[t+1:]*ng_init**t


class state:
    """ This class is just a "struct" to hold the collection of primitives defining
    an economy in which one or multiple generations live """
    def __init__(self, params, r_init=0.03, r_term=0.03, Bq_init=0, Bq_term=0):
        # tr = 0.429, tw = 0.248, zeta=0.5, gy = 0.195, in Section 9.3. in Heer/Maussner
        """tr, tw and tb are tax rates on capital return, wage and tax for pension.
        tb is determined by replacement ratio, zeta, and other endogenous variables.
        gy is ratio of government spending over output.
        Transfer from government to households, Tr, is determined endogenously"""
        self.zeta = params.zeta
        self.delta = delta = params.delta
        self.alpha = alpha = params.alpha
        self.T = T = params.T
        self.phi, self.tol = params.phi, params.tol
        self.aN = aN = params.aN
        self.aa = aa = params.aa
        self.hh = hh = params.hh
        self.hN = hN = params.hN
        """ SURVIVAL PROB., PRODUCTIVITY TRANSITION PROB. AND ... """
        self.sp = sp = params.sp
        self.pi = pi = params.pi
        # muz[y] : distribution of productivity of y-yrs agents
        self.muz = muz = params.muz
        self.ef = ef = params.ef
        self.zN = zN = params.zN
        self.mls = mls = params.mls
        """ CALCULATE POPULATIONS OVER THE TRANSITION PATH """
        self.pop = params.pop
        """Construct containers for market prices, tax rates, pension, bequest"""
        self.theta = params.theta*ones(T)
        self.tau = params.tau*ones(T)
        self.r = r_term*ones(T)
        self.Bq = Bq_term*ones(T)
        self.r[0:T-mls] = linspace(r_init,r_term,T-mls)
        self.Bq[0:T-mls] = linspace(Bq_init,Bq_term,T-mls)
        self.pr, self.L, self.K, self.w, self.b = [zeros(T) for i in range(5)]
        for t in range(T):
            # pr = population of retired agents
            self.pr[t] = sum(self.pop[t,45:])
            # L = labor supply in efficiency unit
            self.L[t] = sum([muz[y].dot(ef[y])*self.pop[t,y] for y in range(mls)])
            self.K[t] = ((self.r[t]+delta)/alpha)**(1.0/(alpha-1.0))*self.L[t]
            self.w[t] = ((self.r[t]+delta)/alpha)**(alpha/(alpha-1.0))*(1.0-alpha)
            self.b[t] = self.theta[t]*self.w[t]*self.L[t]/self.pr[t]
        self.r1 = zeros(T)
        self.K1 = zeros(T)
        self.Bq1 = zeros(T)
        """ PRICES, PENSION BENEFITS, BEQUESTS AND TAXES
        that are observed by households """
        self.prices = array([self.r, self.w, self.b, self.Bq, self.theta, self.tau])


    def aggregate(self, vmu):
        """Aggregate Capital, Labor in Efficiency unit and Bequest over all cohorts"""
        T, mls, alpha, delta, zeta = self.T, self.mls, self.alpha, self.delta, self.zeta
        aa, pop, sp, zN, aN = self.aa, self.pop, self.sp, self.zN, self.aN
        spr = (1-sp)/sp
        my = lambda x: x if x < T-1 else -1
        mu = [array(vmu[t]).reshape(mls,zN,aN) for t in range(len(vmu))]
        """Aggregate all cohorts' capital and labor supply at each year"""
        for t in range(T):
            self.K1[t] = sum([sum(mu[my(t+y)][-(y+1)],0).dot(aa)*pop[t,-(y+1)]
                                        for y in range(mls)])
            self.Bq1[t] = sum([sum(mu[my(t+y)][-(y+1)],0).dot(aa)*pop[t,-(y+1)]*spr[-(y+1)]
                                        for y in range(mls)])*(1-zeta)/sum(pop[t])
        self.r1 = alpha*(self.K1/self.L)**(alpha-1.0)-delta


    def update_all(self):
        """ Update market prices, w and r, and many others according to new
        aggregate capital and labor paths for years 0,...,T from last iteration """
        alpha, delta = self.alpha, self.delta
        self.r = self.phi*self.r + (1-self.phi)*self.r1
        self.K = ((self.r+delta)/alpha)**(1.0/(alpha-1.0))*self.L
        self.w = ((self.r+delta)/alpha)**(alpha/(alpha-1.0))*(1.0-alpha)
        self.b = self.theta*self.w*self.L/self.pr
        self.prices = array([self.r, self.w, self.b, self.Bq, self.theta, self.tau])


    def update_Bq(self):
        """ Update the amount of bequest given to households """
        self.Bq = self.phi*self.Bq + (1-self.phi)*self.Bq1
        self.prices = array([self.r, self.w, self.b, self.Bq, self.theta, self.tau])


class cohort:
    """ This class is just a "struct" to hold the collection of primitives defining
    a generation """
    def __init__(self, params, y=-1, a0 = 0):
        self.beta, self.sigma, self.psi = params.beta, params.sigma, params.psi
        self.R, self.W, self.y = params.R, params.W, y
        # self.mls = mls = (y+1 if (y >= 0) and (y <= W+R-2) else W+R) # mls is maximum life span
        self.aN = aN = params.aN
        self.aa = aa = params.aa
        self.hh = hh = params.hh
        self.hN = hN = params.hN
        self.tol, self.neg = params.tol, params.neg
        self.tcost = params.tcost
        """ SURVIVAL PROB., PRODUCTIVITY TRANSITION PROB. AND ... """
        self.sp = sp = params.sp
        self.pi = pi = params.pi
        # muz[y] : distribution of productivity of y-yrs agents
        self.muz = muz = params.muz
        self.ef = ef = params.ef
        self.zN = zN = params.zN
        self.mls = mls = params.mls
        """ container for value function and expected value function """
        # v[y,h,j,i] is the value of an y-yrs-old agent with asset i and productity j, house h
        self.v = zeros(mls*zN*hN*aN).reshape(mls,hN,zN,aN)
        # ev[y,nh,j,ni] is the expected value when next period asset ni and house hi
        self.ev = zeros(mls*zN*hN*aN).reshape(mls,hN,zN,aN)
        """ container for policy functions """
        self.a = zeros(mls*zN*hN*aN).reshape(mls,hN,zN,aN)
        self.h = zeros(mls*zN*hN*aN).reshape(mls,hN,zN,aN)
        self.c = zeros(mls*zN*hN*aN).reshape(mls,hN,zN,aN)
        """ distribution of agents w.r.t. age, productivity and asset
        for each age, distribution over all productivities and assets add up to 1 """
        self.vmu = zeros(mls*hN*zN*aN)


    def optimalpolicy(self, prices):
        """ Given prices, transfers, benefits and tax rates over one's life-cycle,
        value and decision functions are calculated ***BACKWARD*** """
        aa, hh = self.aa, self.hh
        t = prices.shape[1]
        if t < self.mls:
            d = self.mls - t
            prices = concatenate((tile(array([prices[:,0]]).T,(1,d)),prices), axis=1)
        [r, w, b, Bq, theta, tau] = prices
        ef, mls, aN, zN, hN = self.ef, self.mls, self.aN, self.zN, self.hN
        sigma, theta, psi = self.sigma, self.theta, self.psi
        pi, tcost = self.pi, self.tcost
        """ inline functions: utility and income adjustment by trading house """
        util = lambda c, h: log(x) + psi*log(h+1)
        hinc = lambda h, nh, q: (h-nh)*q - tcost*h*q*(h!=nh)
        """ y = -1 : just before the agent dies """
        for h in range(hN):
            for z in range(zN):
                c = aa*(1+(1-tau[-1])*r[-1]) + hinc(hh[h],hh[0],q[-1])\
                        + w[-1]*ef[-1,z]*(1-theta[-1]-tau[-1]) + b[-1] + Bq[-1]
                c[c<=0.0] = 1e-10
                self.c[-1,h,z] = c
                self.v[-1,h,z] = util(c)
            self.ev[-1,h] = pi.dot(self.v[-1,h])
        """ y = -2, -3,..., -60 """
        for y in range(-2, -(mls+1), -1):
            for h in range(hN):
                for z in range(zN):
                    vt = zeros(hN*aN*aN).reshape(hN,aN,aN)
                    for nh in range(hN):
                        c = tile(self.aa,(aN,1)).T*(1+(1-tau[y])*r[y]) + b[y]*(y>=-self.R) \
                                + w[y]*ef[y,j]*(1-theta[y]-tau[y]) + Bq[y] + hinc(hh[h],hh[nh],q[y]) \
                                - tile(self.aa,(aN,1))
                        c[c<=0.0] = 1e-10
                        vt[nh] = util(c) + self.beta*self.sp[y+1]*tile(self.ev[y+1,nh,j],(aN,1))
                    for a in range(aN):
                        self.h[y,h,z,a], self.a[y,h,z,a] = unravel_index(vt[:,a,:].argmax(),vt.shape)
                        self.v[y,h,z,a] = vt[self.h[y,h,z,a],a,self.a[y,h,z,a]]
                self.ev[y,h] = self.pi.dot(self.v[y,h])
        """ find distribution of agents w.r.t. age, productivity and asset """
        self.vmu = zeros(mls*zN*aN)
        mu = self.vmu.reshape(mls,zN,aN)
        mu[0,:,0] = self.muz[0]
        for y in range(1,mls):
            for j in range(zN):
                for i in range(aN):
                    mu[y,:,self.a[y-1,j,i]] += mu[y-1,j,i]*self.pi[j]


"""The following are procedures to get steady state of the economy using direct
age-profile iteration and projection method"""


def findsteadystate(ng=1.012,N=40):
    """Find Old and New Steady States with population growth rates ng and ng1"""
    start_time = datetime.now()
    params0 = params(T=1,ng_init=ng)
    c = cohort(params0)
    k = state(params0)
    for n in range(N):
        c.optimalpolicy(k.prices)
        k.aggregate([c.vmu])
        while True:
            k.update_Bq()
            if max(absolute(k.Bq - k.Bq1)) < k.tol:
                break
            c.optimalpolicy(k.prices)
            k.aggregate([c.vmu])
        k.update_all()
        print "n=%i" %(n+1),"r=%2.3f" %(k.r),"r1=%2.3f" %(k.r1),\
                "L=%2.3f," %(k.L),"K=%2.3f," %(k.K),"K1=%2.3f," %(k.K1),"Bq1=%2.3f," %(k.Bq1)
        if max(absolute(k.K - k.K1)) < k.tol:
            print 'Economy Converged to SS! in',n+1,'iterations with', k.tol
            break
        if n >= N-1:
            print 'Economy Not Converged in',n+1,'iterations with', k.tol
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return k, c


#병렬처리를 위한 for loop 내 로직 분리
def transition_sub1(t,mu,prices,mu_t,params):
    c = cohort(params)
    T = params.T
    mls = params.mls
    if t < T-1:
        c.optimalpolicy(prices.T[max(t-mls+1,0):t+1].T)
    else:
        c.vmu = mu_t
    for i in range(c.mls*c.zN*c.aN):
        mu[i] = c.vmu[i]


def transition(N=20, TP=320, ng_i=1.012, ng_t=1.0-0.012):
    k_i, c_i = findsteadystate(ng=ng_i)
    k_t, c_t = findsteadystate(ng=ng_t)
    params_tp = params(T=TP, ng_init=ng_i, ng_term=ng_t)
    k_tp = state(params_tp, r_init=k_i.r, r_term=k_t.r, Bq_init=k_i.Bq, Bq_term=k_t.Bq)
    mu_len = c_t.mls*c_t.zN*c_t.aN
    """Generate mu of TP cohorts who die in t = 0,...,T-1 with initial asset g0.apath[-t-1]"""
    mu_tp = [RawArray(c_double, mu_len) for t in range(TP)]
    for n in range(N):
        start_time = datetime.now()
        print 'multiprocessing :'+str(n)+' is in progress : {} \n'.format(start_time)
        jobs = []
        for t, mu in enumerate(mu_tp):
            # transition_sub1(c,k_tp.prices,c_t.mu)
            p = Process(target=transition_sub1, args=(t,mu,k_tp.prices,c_t.vmu,params_tp))
            p.start()
            jobs.append(p)
            #병렬처리 개수 지정 20이면 20개 루프를 동시에 병렬로 처리
            # if len(jobs) % 40 == 0:
        for p in jobs:
            p.join()
            # print 'year '+str(t)+' is in progress : {}'.format(datetime.now())
            jobs=[]
        # if len(jobs) > 0:
            # for p in jobs:
                # p.join()
        k_tp.aggregate(mu_tp)
        k_tp.update_all()
        k_tp.update_Bq()
        # print 'transition('+str(n)+') is done : {}'.format(end_time)
        for t in [0, int(TP/4), int(TP/2), int(3*TP/4), TP-1]:
            print "r=%2.3f" %(k_tp.r[t]),"r1=%2.3f" %(k_tp.r1[t]),"L=%2.3f," %(k_tp.L[t]),\
                "K=%2.3f," %(k_tp.K[t]),"K1=%2.3f," %(k_tp.K1[t]),"Bq1=%2.3f," %(k_tp.Bq1[t])
        end_time = datetime.now()
        print 'transition ('+str(n)+') loop: {}'.format(end_time - start_time)
        if max(absolute(k_tp.K - k_tp.K1)) < k_tp.tol:
            print 'Transition Path Converged! in', n+1,'iterations with', k_tp.tol
            break
        if n >= N-1:
            print 'Transition Path Not Converged! in', n+1,'iterations with', k_tp.tol
            break
    return k_tp, mu_tp


if __name__ == '__main__':
    start_time = datetime.now()
    k, mu = transition()
    end_time = datetime.now()
    print 'Total Duration: {}'.format(end_time - start_time)
    plt.plot(k.r)
    plt.show()
