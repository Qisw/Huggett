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


class state:
    """ This class is just a "struct" to hold the collection of primitives defining
    an economy in which one or multiple generations live """
    def __init__(self, cohort, alpha=0.36, delta=0.06, tau=0.2378, theta=0.1, zeta=0.3,
        phi=0.7, tol=0.01, r_init=0.03, r_term=0.03, Bq_init=0, Bq_term=0,
        T=1, ng_init = 1.012, ng_term = 1.0-.012):
        # tr = 0.429, tw = 0.248, zeta=0.5, gy = 0.195, in Section 9.3. in Heer/Maussner
        """tr, tw and tb are tax rates on capital return, wage and tax for pension.
        tb is determined by replacement ratio, zeta, and other endogenous variables.
        gy is ratio of government spending over output.
        Transfer from government to households, Tr, is determined endogenously"""
        self.alpha, self.zeta, self.delta, self.tau = alpha, zeta, delta, tau
        self.theta = theta
        self.T = T
        self.phi, self.tol = phi, tol
        self.aa = cohort.aa
        """ LOAD PARAMETERS : SURVIVAL PROB., PRODUCTIVITY TRANSITION PROB. AND ... """
        self.sp = sp = loadtxt('sp.txt', delimiter='\n')  # survival probability
        self.pi = genfromtxt('pi.csv', delimiter=',')  # productivity transition probability
        # muz[y] : distribution of productivity of y-yrs agents
        self.muz = muz = genfromtxt('muz.csv', delimiter=',')
        self.ef = ef = genfromtxt('ef.csv',delimiter=',')  # efficiency units
        self.mls = mls = ef.shape[0]
        """ CALCULATE POPULATIONS OVER THE TRANSITION PATH """
        m0 = array([prod(sp[:y+1])/ng_init**y for y in range(mls)], dtype=float)
        m1 = array([prod(sp[:y+1])/ng_term**y for y in range(mls)], dtype=float)
        self.pop = array([m1*ng_term**t for t in range(T)], dtype=float)
        for t in range(min(T,mls-1)):
            self.pop[t,t+1:] = m0[t+1:]*ng_init**t
        """Construct containers for market prices, tax rates, pension, bequest"""
        self.theta = array([theta for t in range(T)], dtype=float)
        self.tau = array([tau for t in range(T)], dtype=float)
        self.r0 = r0 = linspace(r_init,r_term,T)
        self.r1 = r1 = linspace(0,0,T)
        self.pr = array([sum(self.pop[t,45:]) for t in range(T)], dtype=float)
        self.L = array([sum([muz[y].dot(ef[y])*self.pop[t,y] for y in range(mls)])
                                                for t in range(T)], dtype=float)
        self.K0 = array([((r0[t]+delta)/alpha)**(1.0/(alpha-1.0))*self.L[t]
                                                for t in range(T)], dtype=float)
        self.K1 = array([0 for t in range(T)], dtype=float)
        self.w = array([((r0[t]+delta)/alpha)**(alpha/(alpha-1.0))*(1.0-alpha)
                                                for t in range(T)], dtype=float)
        self.b = array([self.theta[t]*self.w[t]*self.L[t]/self.pr[t]
                                                for t in range(T)], dtype=float)
        self.Bq0 = linspace(Bq_init,Bq_term,T)
        self.Bq1 = linspace(0,0,T)
        """ PRICES, PENSION BENEFITS, BEQUESTS AND TAXES
        that are observed by households """
        self.prices = array([self.r0, self.w, self.b, self.Bq0, self.theta, self.tau])


    def aggregate(self, cs):
        """Aggregate Capital, Labor in Efficiency unit and Bequest over all cohorts"""
        T, mls, alpha, delta, zeta = self.T, self.mls, self.alpha, self.delta, self.zeta
        aa, pop, sp = self.aa, self.pop, self.sp
        spr = (1-sp)/sp
        """Aggregate all cohorts' capital and labor supply at each year"""
        for t in range(T):
            if t <= T-mls-1:
                self.K1[t] = sum([sum(cs[t+y].mu[-(y+1)],0).dot(aa)*pop[t,-(y+1)]
                                        for y in range(mls)])
                self.Bq1[t] = sum([sum(cs[t+y].mu[-(y+1)],0).dot(aa)*pop[t,-(y+1)]*spr[-(y+1)]
                                        for y in range(mls)])*(1-zeta)/sum(pop[t])
            else:
                self.K1[t] = sum([sum(cs[-1].mu[-(y+1)],0).dot(aa)*pop[t,-(y+1)]
                                        for y in range(mls)])
                self.Bq1[t] = sum([sum(cs[-1].mu[-(y+1)],0).dot(aa)*pop[t,-(y+1)]*spr[-(y+1)]
                                        for y in range(mls)])*(1-zeta)/sum(pop[t])
        self.r1 = alpha*(self.K1/self.L)**(alpha-1.0)-delta


    def update_all(self):
        """ Update market prices, w and r, and many others according to new
        aggregate capital and labor paths for years 0,...,T from last iteration """
        alpha, delta = self.alpha, self.delta
        self.r0 = self.phi*self.r0 + (1-self.phi)*self.r1
        self.K0 = ((self.r0+delta)/alpha)**(1.0/(alpha-1.0))*self.L
        self.w = ((self.r0+delta)/alpha)**(alpha/(alpha-1.0))*(1.0-alpha)
        self.b = self.theta*self.w*self.L/self.pr
        self.prices = array([self.r0, self.w, self.b, self.Bq0, self.theta, self.tau])


    def update_Bq(self):
        """ Update the amount of bequest given to households """
        self.Bq0 = self.phi*self.Bq0 + (1-self.phi)*self.Bq1
        self.prices = array([self.r0, self.w, self.b, self.Bq0, self.theta, self.tau])


class cohort:
    """ This class is just a "struct" to hold the collection of primitives defining
    a generation """
    def __init__(self, beta=0.994, sigma=1.5, aH=50.0, aL=0.0, y=-1,
        aN=51, tol=0.01, neg=-1e10, W=45, R=34, a0 = 0):
        self.beta, self.sigma = beta, sigma
        self.R, self.W, self.y = R, W, y
        self.mls = mls = (y+1 if (y >= 0) and (y <= W+R-2) else W+R) # mls is maximum life span
        self.aH, self.aL, self.aN, self.aa = aH, aL, aN, aL+aH*linspace(0,1,aN)
        self.tol, self.neg = tol, neg
        self.sp = loadtxt('sp.txt', delimiter='\n')  # survival probability
        self.muz = genfromtxt('muz.csv', delimiter=',')  # initial distribution of productivity
        self.pi = genfromtxt('pi.csv', delimiter=',')  # productivity transition probability
        self.ef = genfromtxt('ef.csv', delimiter=',')
        self.zN = zN = self.pi.shape[0]
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


    def optimalpolicy(self, prices):
        """ Given prices, transfers, benefits and tax rates over one's life-cycle,
        value and decision functions are calculated ***BACKWARD*** """
        t = prices.shape[1]
        if t < self.mls:
            d = self.mls - t
            prices = concatenate((tile(array([prices[:,0]]).T,(1,d)),prices), axis=1)
        [r, w, b, Bq, theta, tau] = prices
        ef, mls, aN, zN = self.ef, self.mls, self.aN, self.zN
        # y = -1 : just before the agent dies
        for j in range(self.zN):
            c = self.aa*(1+(1-tau[-1])*r[-1]) \
                    + w[-1]*ef[-1,j]*(1-theta[-1]-tau[-1]) + b[-1] + Bq[-1]
            c[c<=0.0] = 1e-10
            self.c[-1,j] = c
            self.v[-1,j] = self.util(c)
        self.ev[-1] = self.pi.dot(self.v[-1])
        # y = -2, -3,..., -60
        for y in range(-2, -(mls+1), -1):
            for j in range(zN):
                na = tile(self.aa,(aN,1)).T*(1+(1-tau[y])*r[y]) \
                        + w[y]*ef[y,j]*(1-theta[y]-tau[y]) + b[y] + Bq[y]
                c = na - tile(self.aa,(aN,1))
                c[c<=0.0] = 1e-10
                v = self.util(c) + self.beta*self.sp[y+1]*tile(self.ev[y+1,j],(aN,1))
                self.a[y,j] = argmax(v,1)
                for i in range(aN):
                    self.v[y,j,i] = v[i,self.a[y,j,i]]
                    self.c[y,j,i] = c[i,self.a[y,j,i]]
            self.ev[y] = self.pi.dot(self.v[y])


    def calculate_mu(self):
        """ find distribution of agents w.r.t. age, productivity and asset """
        self.mu = self.mu*0
        self.mu[0,:,0] = self.muz[0]
        for y in range(1,self.mls):
            for j in range(self.zN):
                for i in range(self.aN):
                    self.mu[y,:,self.a[y-1,j,i]] += self.mu[y-1,j,i]*self.pi[j]


    def util(self, c): # period utility
        return c**(1.0-self.sigma)/(1.0-self.sigma)


"""The following are procedures to get steady state of the economy using direct
age-profile iteration and projection method"""


def findinitial(ng0=1.01, ng1=1.00, W=45, R=30, TG=4, alpha=0.3, beta=0.96, delta=0.08):
    start_time = datetime.now()
    """Find Old and New Steady States with population growth rates ng and ng1"""
    E0, g0 = value(state(TG=1,W=W,R=R,ng=ng0,alpha=alpha,delta=delta),
                    cohort(beta=beta,W=W,R=R))
    E1, g1 = value(state(TG=1,W=W,R=R,ng=ng1,alpha=alpha,delta=delta),
                    cohort(beta=beta,W=W,R=R))
    mls = W + R
    TS = mls*TG
    """Initialize Transition Path for t = 0,...,TS-1"""
    Et= state(TG=TG,W=W,R=R,ng=ng0,dng=(ng1-ng0),k=E1.k[0],alpha=alpha,delta=delta)
    Et.k[:TS-mls] = linspace(E0.k[-1],E1.k[0],TS-mls)
    Et.update()
    with open('E.pickle','wb') as f:
        pickle.dump([E0, E1, Et, 0], f)
    with open('G.pickle','wb') as f:
        pickle.dump([g0.apath, g0.epath, g0.lpath, g1.apath, g1.epath, g1.lpath], f)
    """http://stackoverflow.com/questions/2204155/
    why-am-i-getting-an-error-about-my-class-defining-slots-when-trying-to-pickl"""
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))


def spath(g):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    fig.subplots_adjust(hspace=.5, wspace=.3, left=None, right=None, top=None, bottom=None)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax1.plot(g.apath)
    ax2.plot(g.lpath)
    ax3.plot(g.cpath)
    ax4.plot(g.upath)
    ax.set_xlabel('generation')
    ax1.set_title('Asset')
    ax2.set_title('Labor')
    ax3.set_title('Consumption')
    ax4.set_title('Utility')
    plt.show()
    # time.sleep(1)
    # plt.close() # plt.close("all")


def tpath():
    with open('E.pickle','rb') as f:
        [E0, E1, Et, it] = pickle.load(f)
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    fig.subplots_adjust(hspace=.5, wspace=.3, left=None, right=None, top=None, bottom=None)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax1.plot(Et.k)
    ax2.plot(Et.r)
    ax3.plot(Et.L)
    ax4.plot(Et.w)
    ax5.plot(Et.K)
    ax6.plot(Et.C)
    ax.set_xlabel('generation')
    ax.set_title('R:' + str(Et.R) + 'W:' + str(Et.W) + 'TS:' + str(Et.TS), y=1.08)
    ax1.set_title('Capital/Labor')
    ax2.set_title('Interest Rate')
    ax3.set_title('Labor')
    ax4.set_title('Wage')
    ax5.set_title('Capital')
    ax6.set_title('Consumption')
    plt.show()
    # time.sleep(1)
    # plt.close() # plt.close("all")


def findsteadystate(ng=1.012,N=20):
    """Find Old and New Steady States with population growth rates ng and ng1"""
    start_time = datetime.now()
    c = cohort()
    k = state(c,ng_init=ng)
    for n in range(N):
        c.optimalpolicy(k.prices)
        c.calculate_mu()
        k.aggregate([c])
        while True:
            k.update_Bq()
            if max(absolute(k.Bq0 - k.Bq1)) < k.tol:
                break
            c.optimalpolicy(k.prices)
            c.calculate_mu()
            k.aggregate([c])
        k.update_all()
        print "n=%i" %(n+1),"r0=%2.3f" %(k.r0),"r1=%2.3f" %(k.r1),\
                "L=%2.3f," %(k.L),"K0=%2.3f," %(k.K0),"K1=%2.3f," %(k.K1),"Bq1=%2.3f," %(k.Bq1)
        if max(absolute(k.K0 - k.K1)) < k.tol:
            print 'Economy Converged to SS! in',n+1,'iterations with', k.tol
            break
        if n >= N-1:
            print 'Economy Not Converged in',n+1,'iterations with', k.tol
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return k, c


# if __name__ == '__main__':
    start_time = datetime.now()
    """Find Old and New Steady States with population growth rates ng and ng1"""
    c = cohort()
    k = state(c,r_init=0.11,T=1)
    N = 20
    for n in range(N):
        c.optimalpolicy(k.prices)
        c.calculate_mu()
        k.aggregate([c])
        while True:
            k.update_Bq()
            if max(absolute(k.Bq0 - k.Bq1)) < k.tol:
                break
            c.optimalpolicy(k.prices)
            c.calculate_mu()
            k.aggregate([c])
        k.update_all()
        print "n=%i" %(n+1),"r0=%2.3f" %(k.r0),"r1=%2.3f" %(k.r1),\
                "L=%2.3f," %(k.L),"K0=%2.3f," %(k.K0),"K1=%2.3f," %(k.K1),"Bq1=%2.3f," %(k.Bq1)
        if max(absolute(k.K0 - k.K1)) < k.tol:
            print 'Economy Converged to SS! in',n+1,'iterations with', k.tol
            break
        if n >= N-1:
            print 'Economy Not Converged in',n+1,'iterations with', k.tol
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))


#병렬처리를 위한 for loop 내 로직 분리
def transition_sub1(c,prices,mu_t):
    mls = c.mls
    T = prices.shape[1]
    if (c.y >= mls-1) and (c.y <= T-(mls+1)):
        c.optimalpolicy(prices.T[c.y-mls+1:c.y+1].T)
        c.calculate_mu()
    elif (c.y < mls-1):
        c.optimalpolicy(prices.T[:,:c.y+1].T)
        c.calculate_mu()
    else:
        c.mu = mu_t


def transition(N=15,T=300,ng_i=1.012,ng_t=1.0):
    k_i, c_i = findsteadystate(ng=ng_i)
    k_t, c_t = findsteadystate(ng=ng_t)
    k_tp = state(c_i, T=T, r_init=k_i.r0, r_term=k_t.r0, Bq_init=k_i.Bq0, Bq_term=k_t.Bq0,
                      ng_init=ng_i, ng_term=ng_t)
    for t in range(0,300,50):
        print "r0=%2.3f" %(k_tp.r0[t]),"r1=%2.3f" %(k_tp.r1[t]),"L=%2.3f," %(k_tp.L[t]),\
                "K0=%2.3f," %(k_tp.K0[t]),"K1=%2.3f," %(k_tp.K1[t]),"Bq1=%2.3f," %(k_tp.Bq1[t])
    """Generate T cohorts who die in t = 0,...,T-1 with initial asset g0.apath[-t-1]"""
    cohorts = [cohort(y=t) for t in range(T)]
    """Iteratively Calculate all generations optimal consumption and labour supply"""
    for n in range(N):
        start_time = datetime.now()
        print 'transition('+str(n)+') is in progress : {}'.format(start_time)
        jobs = []
        for c in cohorts:
            # transition_sub1(c,k_tp.prices,c_t.mu)
            p = Process(target=transition_sub1, args=(c,k_tp.prices,c_t.mu))
            p.start()
            jobs.append(p)
            #병렬처리 개수 지정 20이면 20개 루프를 동시에 병렬로 처리
            if len(jobs) % 4 == 0:
                for p in jobs:
                    p.join()
                print 'transition('+str(n)+') is in progress : {}'.format(datetime.now())
                jobs=[]
        if len(jobs) > 0:
            for p in jobs:
                p.join()
        k_tp.aggregate(cohorts)
        k_tp.update_all()
        k_tp.update_Bq()
        end_time = datetime.now()
        print 'transition('+str(n)+') is done : {}'.format(end_time)
        print 'transition ('+str(n)+') loop: {}'.format(end_time - start_time)
        for t in range(0,300,50):
            print "r0=%2.3f" %(k_tp.r0[t]),"r1=%2.3f" %(k_tp.r1[t]),"L=%2.3f," %(k_tp.L[t]),\
                "K0=%2.3f," %(k_tp.K0[t]),"K1=%2.3f," %(k_tp.K1[t]),"Bq1=%2.3f," %(k_tp.Bq1[t])
        if max(absolute(k_tp.K0 - k_tp.K1)) < k_tp.tol:
            print 'Transition Path Converged! in', n+1,'iterations with', k_tp.tol
            break
        if n >= N-1:
            print 'Transition Path Not Converged! in', n+1,'iterations with', k_tp.tol
            break
    return k_tp, cohorts
