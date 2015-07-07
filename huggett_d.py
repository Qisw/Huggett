# -*- coding: utf-8 -*-
"""
Jul. 7, 2015, Hyun Chang Yi
Huggett (1996) "Wealth distribution in life-cycle economies," Journal of Monetary
Economics, 38(3), 469-494.
"""

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize_scalar
from numpy import linspace, mean, array, zeros, absolute, loadtxt, dot, prod, genfromtxt, sum, argmax
from matplotlib import pyplot as plt
from datetime import datetime
import time
import pickle

from multiprocessing import Process, Lock, Manager


class state:
    """ This class is just a "struct" to hold  the collection of primitives defining
    an economy in which one or multiple generations live """
    def __init__(self, alpha=0.36, delta=0.06, tau=0.2378, theta=0.1, zeta=0.3,
        phi=0.5, tol=0.01, r_init=0.03, Bq_init=0,
        T=1, ng = 1.012, dng = 0.0):
        # tr = 0.429, tw = 0.248, zeta=0.5, gy = 0.195, in Section 9.3. in Heer/Maussner
        """tr, tw and tb are tax rates on capital return, wage and tax for pension.
        tb is determined by replacement ratio, zeta, and other endogenous variables.
        gy is ratio of government spending over output.
        Transfer from government to households, Tr, is determined endogenously"""
        self.alpha, self.zeta, self.delta, self.tau = alpha, zeta, delta, tau
        self.theta = theta
        self.T = T
        self.phi, self.tol = phi, tol
        ng0, ng1 = ng, ng + dng
        self.sp = sp = loadtxt('sp.txt', delimiter='\n')  # survival probability
        self.pi = genfromtxt('pi.csv', delimiter=',')  # productivity transition probability
        # muz[y] : distribution of productivity of y-yrs agents
        self.muz = muz = genfromtxt('muz.csv', delimiter=',')
        self.ef = ef = genfromtxt('ef.csv',delimiter=',')  # efficiency of i-productivity and j-age
        self.ly = ly = ef.shape[0]
        m0 = array([prod(sp[:y+1])/ng0**y for y in range(ly)], dtype=float)
        m1 = array([prod(sp[:y+1])/ng1**y for y in range(ly)], dtype=float)
        self.pop = array([m1*ng1**t for t in range(T)], dtype=float)
        for t in range(min(T,ly-1)):
            self.pop[t,t+1:] = m0[t+1:]*ng0**t
        """Construct containers for market prices, tax rates, transfers, other aggregate variables"""
        self.theta = array([theta for t in range(T)], dtype=float)
        self.tau = array([tau for t in range(T)], dtype=float)
        self.r0 = r0 =array([r_init for t in range(T)], dtype=float)
        self.r1 = r1 =array([r_init for t in range(T)], dtype=float)
        self.pr = array([sum(self.pop[t,45:]) for t in range(T)], dtype=float)
        self.L = array([sum([muz[y].dot(ef[y])*self.pop[t,y] for y in range(ly)]) for t in range(T)], dtype=float)
        self.K0 = array([((r0[t]+delta)/alpha)**(1.0/(alpha-1.0))*self.L[t] for t in range(T)], dtype=float)
        self.K1 = array([0 for t in range(T)], dtype=float)
        self.w = array([((r0[t]+delta)/alpha)**(alpha/(alpha-1.0))*(1.0-alpha) for t in range(T)], dtype=float)
        self.b = array([self.theta[t]*self.w[t]*self.L[t]/self.pr[t] for t in range(T)], dtype=float)
        self.Bq0 = array([Bq_init for t in range(T)], dtype=float)
        self.Bq1 = array([0 for t in range(T)], dtype=float)
        self.rdiff = 10
        self.Bqdiff = 10
        self.Kdiff = 10
        # container for r, w, b, Bq, theta, tau
        self.rwbq = array([self.r0, self.w, self.b, self.Bq0, self.theta, self.tau])


    def aggregate(self, cohorts):
        """Aggregate Capital, Labor in Efficiency unit and Bequest over all cohorts"""
        T, ly, alpha, delta = self.T, self.ly, self.alpha, self.delta
        aa = cohorts[-1].aa
        pop, sp = self.pop, self.sp
        """Aggregate all cohorts' capital and labor supply at each year"""
        for t in range(T):
            tt = t+y if t <= T-ly-1 else -1
            self.K1[t] = sum([sum(cohorts[tt].mu[y,:,:],0).dot(aa)*pop[t,y] for y in range(ly)])
            self.Bq1[t] = (1-self.zeta)*sum([sum(cohorts[tt].mu[y,:,:],0).dot(aa)*pop[t,y]\
                                                      /sp[y]*(1-sp[y]) for y in range(ly)])
            self.r1[t] = max(0.001,alpha*(self.K1[t]/self.L[t])**(alpha-1.0)-delta)
        self.rdiff = max(absolute(self.r0 - self.r1))
        self.Kdiff = max(absolute(self.K0 - self.K1))
        self.Bqdiff = max(absolute(self.Bq0 - self.Bq1))
        """ Update the economy's aggregate K and N with weight phi on the old """
        # print "K=%2.2f," %(self.K[0]),"L=%2.2f," %(self.L[0]),"K/L=%2.2f" %(self.k[0])


    def update(self):
        """ Update market prices, w and r, and many others according to new
        aggregate capital and labor paths for years 0,...,T from last iteration """
        alpha, delta = self.alpha, self.delta
        for t in range(self.T):
            self.r0[t] = self.phi*self.r0[t] + (1-self.phi)*self.r1[t]
            self.K0[t] = ((self.r0[t]+delta)/alpha)**(1.0/(alpha-1.0))*self.L[t]
            self.w[t] = ((self.r0[t]+delta)/alpha)**(alpha/(alpha-1.0))*(alpha-1.0)
            self.b[t] = self.theta[t]*self.w[t]*self.L[t]/self.pr[t]
            self.Bq0[t] = self.Bq1[t]
        # container for r, w, b, Bq, theta, tau
        self.rwbq = array([self.r0, self.w, self.b, self.Bq0, self.theta, self.tau])
        # print "for r=%2.2f," %(self.r[0]*100), "w=%2.2f," %(self.w[0]), \
        #         "Tr=%2.2f," %(self.Tr[0]), "b=%2.2f," %(self.b[0]), "Beq.=%2.2f," %(self.Beq[0])


    def update_Bq(self):
        """ Update market prices, w and r, and many others according to new
        aggregate capital and labor paths for years 0,...,T from last iteration """
        alpha, delta = self.alpha, self.delta
        for t in range(self.T):
            self.Bq0[t] = self.phi*self.Bq0[t] + (1-self.phi)*self.Bq1[t]
        # container for r, w, b, Bq, theta, tau
        self.rwbq = array([self.r0, self.w, self.b, self.Bq0, self.theta, self.tau])
        # print "for r=%2.2f," %(self.r[0]*100), "w=%2.2f," %(self.w[0]), \
        #         "Tr=%2.2f," %(self.Tr[0]), "b=%2.2f," %(self.b[0]), "Beq.=%2.2f," %(self.Beq[0])



class cohort:
    """ This class is just a "struct" to hold the collection of primitives defining
    a generation """
    def __init__(self, beta=0.994, sigma=1.5, aH=50.0, aL=0.0, y=-1,
        aN=51, tol=0.01, neg=-1e10, W=45, R=34, a0 = 0):
        self.beta, self.sigma = beta, sigma
        self.R, self.W, self.y = R, W, y
        self.T = T = (y+1 if (y >= 0) and (y <= W+R-2) else W+R)
        self.aH, self.aL, self.aN, self.aa = aH, aL, aN, aL+aH*linspace(0,1,aN)
        self.tol, self.neg = tol, neg
        self.sp = loadtxt('sp.txt', delimiter='\n')  # survival probability
        self.muz = genfromtxt('muz.csv', delimiter=',')  # initial distribution of productivity
        self.pi = genfromtxt('pi.csv', delimiter=',')  # productivity transition probability
        self.ef = genfromtxt('ef.csv', delimiter=',')
        self.zN = zN = self.pi.shape[0]
        """ value function and its interpolation """
        self.v = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(T)], dtype=float)
        self.ev = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(T)], dtype=float)
        #self.vtilde = [[] for y in range(T)]
        """ policy functions used in value function method """
        self.a = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(T)], dtype=float)
        self.c = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(T)], dtype=float)
        """ distribution of agents w.r.t. age, productivity and asset """
        self.mu = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(T)], dtype=float)


    def optimalpolicy(self, rwbq):
        """ Given prices, transfers, benefits and tax rates over one's life-cycle,
        value and decision functions are calculated ***BACKWARD*** """
        [r, w, b, Bq, theta, tau] = rwbq
        ef = self.ef
        T = self.T
        aN, zN = self.aN, self.zN
        aa = self.aa
        # self.v = self.v*0
        # self.ev = self.ev*0
        # y = -1 : the oldest generation
        for j in range(zN):
            for i in range(aN):
                self.c[-1,j,i] = max(self.neg, self.aa[i]*(1+(1-tau[-1])*r[-1])
                                 + w[-1]*ef[-1,j]*(1-theta[-1]-tau[-1]) + b[-1] + Bq[-1])
                self.v[-1,j,i] = self.util(self.c[-1,j,i])
        self.ev[-1] = self.pi.dot(self.v[-1])
                #self.vtilde[-1] = interp1d(self.aa, self.v[-1], kind='cubic')
        # y = -2, -3,..., -60
        for y in range(-2, -(T+1), -1):
            for j in range(zN):
                for i in range(aN):    # l = 0, 1, ..., 50
                    v = array([0 for i in range(aN)], dtype=float)
                    for ni in range(aN):
                        c = aa[i]*(1+(1-tau[y])*r[y]) + w[y]*ef[y,j]*(1-theta[y]-tau[y]) \
                            + b[y] + Bq[y] - aa[ni]
                        v[i] = self.util(c) + self.beta*self.sp[y+1]*self.ev[y+1,j,ni]
                    self.a[y,j,i] = argmax(v)
                    self.v[y,j,i] = v[self.a[y,j,i]]
            self.ev[y] = self.pi.dot(self.v[y])


    def calculate_mu(self):
        """ find distribution of agents w.r.t. age, productivity and asset """
        a, pi = self.a, self.pi
        self.mu = self.mu*0
        self.mu[0,:,0] = self.muz[0]
        for y in range(1,self.T):
            for j in range(self.zN):
                for i in range(self.aN):
                    self.mu[y,:,a[y-1,j,i]] = self.mu[y,:,a[y-1,j,i]] \
                                                + self.mu[y-1,j,i]*pi[j]


    def GetBracket(self, y, j, i, m, rwbq):
        """ Find a bracket (a,b,c) such that policy function for next period asset level,
        a[x;asset[l],y] lies in the interval (a,b) """
        aa = self.aa
        a, b, c = aa[0], 2*aa[0]-aa[1], 2*aa[0]-aa[2]
        minit = m
        m0 = m
        v0 = self.neg
        """ The slow part of if slope != float("inf") is no doubt converting
        the string to a float. """
        while (a > b) or (b > c):
            v1 = self.value(y, j, i, m, rwbq)
            if v1 > v0:
                a, b, = ([aa[m], aa[m]] if m == minit else [aa[m-1], aa[m]])
                v0, m0 = v1, m
            else:
                c = aa[m]
            if m == self.aN - 1:
                a, b, c = aa[m-1], aa[m], aa[m]
            m = m + 1
        return m0, a, b, c


    def value(self, y, j, i, ni, rwbq):
        """ Return the value at the given age y, productivity z[j] and asset aa[i]
        when the agent chooses his next period asset aa[ni] and current consumption c
        a1 is always within aL and aH """
        [r, w, b, Bq, theta, tau] = rwbq
        c = max(self.neg, self.aa[i]*(1+(1-tau[y])*r[y])
                + w[y]*self.ef[y,j]*(1-theta[y]-tau[y]) + b[y] + Bq[y] - self.aa[ni])
        ev = sum([self.v[y+1,nj,ni]*self.pi[j,nj] for nj in range(self.zN)])
        return self.util(c) + self.beta*self.sp[y+1]*ev


    def clip(self, a):
        return self.aL if a <= self.aL else self.aH if a >= self.aH else a


    def util(self, c):
        # calculate utility value with given consumption
        return c**(1.0-self.sigma)/(1.0-self.sigma)


    def uc(self, c):
        # marginal utility w.r.t. consumption
        return c**(-self.sigma)


"""The following are procedures to get steady state of the economy using direct
age-profile iteration and projection method"""


def findinitial(ng0=1.01, ng1=1.00, W=45, R=30, TG=4, alpha=0.3, beta=0.96, delta=0.08):
    start_time = datetime.now()
    """Find Old and New Steady States with population growth rates ng and ng1"""
    E0, g0 = value(state(TG=1,W=W,R=R,ng=ng0,alpha=alpha,delta=delta),
                    cohort(beta=beta,W=W,R=R))
    E1, g1 = value(state(TG=1,W=W,R=R,ng=ng1,alpha=alpha,delta=delta),
                    cohort(beta=beta,W=W,R=R))
    T = W + R
    TS = T*TG
    """Initialize Transition Path for t = 0,...,TS-1"""
    Et= state(TG=TG,W=W,R=R,ng=ng0,dng=(ng1-ng0),k=E1.k[0],alpha=alpha,delta=delta)
    Et.k[:TS-T] = linspace(E0.k[-1],E1.k[0],TS-T)
    Et.update()
    with open('E.pickle','wb') as f:
        pickle.dump([E0, E1, Et, 0], f)
    with open('G.pickle','wb') as f:
        pickle.dump([g0.apath, g0.epath, g0.lpath, g1.apath, g1.epath, g1.lpath], f)
    """http://stackoverflow.com/questions/2204155/
    why-am-i-getting-an-error-about-my-class-defining-slots-when-trying-to-pickl"""
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

#병렬처리를 위한 for loop 내 로직 분리
def transition_sub1(g,T,TS,Et,a1,e1):
    if (g.y >= T-1) and (g.y <= TS-(T+1)):
        g.findvpath(Et.p[:,g.y-T+1:g.y+1])
    elif (g.y < T-1):
        g.findvpath(Et.p[:,:g.y+1])
    else:
        g.apath, g.epath = a1, e1


def transition(N=15,beta=0.96):
    with open('E.pickle','rb') as f:
        [E0, E1, Et, it] = pickle.load(f)
    with open('G.pickle','rb') as f:
        [a0, e0, l0, a1, e1, l1] = pickle.load(f)
    T = Et.T
    TS = Et.TS
    """Generate TS cohorts who die in t = 0,...,TS-1 with initial asset g0.apath[-t-1]"""
    gs = [cohort(beta=beta,W=Et.W,R=Et.R,y=t,a0=(a0[-t-1] if t <= T-2 else 0))
            for t in range(TS)]
    """Iteratively Calculate all generations optimal consumption and labour supply"""
    for n in range(N):
        start_time = datetime.now()
        print 'transition('+str(n)+') is start : {}'.format(start_time)
        jobs = []
        for g in gs:
            p = Process(target=transition_sub1, args=(g,T,TS,Et,a1,e1))
            p.start()
            jobs.append(p)
            #병렬처리 개수 지정 20이면 20개 루프를 동시에 병렬로 처리
            if len(jobs) % 20 == 0:
                for p in jobs:
                    p.join()
                print 'transition('+str(n)+') is progressing : {}'.format(datetime.now())
                jobs=[]
                #            start_time_gs = datetime.now()
                #            if (g.y >= T-1) and (g.y <= TS-(T+1)):
                #                g.findvpath(Et.p[:,g.y-T+1:g.y+1])
                #            elif (g.y < T-1):
                #                g.findvpath(Et.p[:,:g.y+1])
                #            else:
                #                g.apath, g.epath = a1, e1
                #            print('transition gs loop: {}'.format(datetime.now() - start_time_gs))
        if len(jobs) > 0:
            for p in jobs:
                p.join()
        Et.aggregate(gs)
        Et.update()
        print 'after',n+1,'iterations over all cohorts,','r:', E0.r[0], Et.r[0::30]
        end_time = datetime.now()
        print 'transition('+str(n)+') is end : {}'.format(end_time)
        print 'transition n loop: {}'.format(end_time - start_time)
        with open('E.pickle','wb') as f:
            pickle.dump([E0, E1, Et, n+1], f)
        with open('GS.pickle','wb') as f:
            pickle.dump([[gs[t].apath for t in range(TS)], [gs[t].cpath for t in range(TS)],
                            [gs[t].lpath for t in range(TS)], n+1], f)
        if Et.Converged:
            print 'Transition Path Converged! in', n+1,'iterations with', Et.tol
            break
        if n >= N-1:
            print 'Transition Path Not Converged! in', n+1,'iterations with', Et.tol
            break


def value(e, g, N=15):
    start_time = datetime.now()
    for n in range(N):
        e.update()
        g.findvpath(e.p)
        e.aggregate([g])
        if e.Converged:
            print 'Economy Converged to SS! in',n+1,'iterations with', e.tol
            break
        if n >= N-1:
            print 'Economy Not Converged in',n+1,'iterations with', e.tol
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return e, g


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


if __name__ == '__main__':
    start_time = datetime.now()
    """Find Old and New Steady States with population growth rates ng and ng1"""
    k = state(r_init=0.11,T=1)
    c = cohort()
    N = 1
    #rgrid = linspace(0.0,0.2,6)

    for n in range(N):
        while (k.Bqdiff > 0.01):
            start_time1 = datetime.now()
            c.optimalpolicy(array([k.rwbq for y in range(k.ly)]).T[0])
            end_time1 = datetime.now()
            print('Duration of optimalpolicy: {}'.format(end_time1 - start_time1))
            start_time2 = datetime.now()
            c.calculate_mu()
            end_time2 = datetime.now()
            print('Duration of calculate_mu: {}'.format(end_time2 - start_time2))
            start_time3 = datetime.now()
            k.aggregate([c])
            k.update_Bq()
            end_time3 = datetime.now()
            print('Duration of aggregate and update: {}'.format(end_time3 - start_time3))
            print "Bqdiff=%2.2f" %(k.Bqdiff),"r0=%2.2f" %(k.r0[0]),\
                    "L=%2.2f," %(k.L),"K1=%2.2f," %(k.K1),"Bq1=%2.2f," %(k.Bq1)
        c.optimalpolicy(array([k.rwbq for y in range(k.ly)]).T[0])
        c.calculate_mu()
        k.aggregate([c])
        if k.rdiff < 0.001:
            print 'Economy Converged to SS! in',n+1,'iterations with', k.tol
            break
        if n >= N-1:
            print 'Economy Not Converged in',n+1,'iterations with', k.tol
            break
        print "n=%2.2f" %(n),"r0=%2.2f" %(k.r0[0]),"r1=%2.2f" %(k.r1[0]),\
                "L=%2.2f," %(k.L),"K1=%2.2f," %(k.K1),"Bq1=%2.2f," %(k.Bq1),\
                "Kdiff=%2.2f," %(k.Kdiff),"Bqdiff=%2.2f," %(k.Bqdiff)
        n = n + 1
        k.update()
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
