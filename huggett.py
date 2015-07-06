# -*- coding: utf-8 -*-
"""
Jul. 6, 2015, Hyun Chang Yi
Huggett (1996) "Wealth distribution in life-cycle economies," Journal of Monetary
Economics, 38(3), 469-494.
"""

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize_scalar
from numpy import linspace, mean, array, zeros, absolute, loadtxt, dot, prod, genfromtxt
from matplotlib import pyplot as plt
from datetime import datetime
import time
import pickle

from multiprocessing import Process, Lock, Manager


class state:
    """ This class is just a "struct" to hold  the collection of primitives defining
    an economy in which one or multiple generations live """
    def __init__(self, alpha=0.36, delta=0.06, tau=0.2378, theta=0.1, zeta=0.3,
        phi=0.8, tol=0.01, r=0.03,
        T=1, ng = 1.012, dng = 0.0):
        # tr = 0.429, tw = 0.248, zeta=0.5, gy = 0.195, in Section 9.3. in Heer/Maussner
        """tr, tw and tb are tax rates on capital return, wage and tax for pension.
        tb is determined by replacement ratio, zeta, and other endogenous variables.
        gy is ratio of government spending over output.
        Transfer from government to households, Tr, is determined endogenously"""
        self.alpha, self.zeta, self.delta = alpha, zeta, delta
        self.phi, self.tol = phi, tol
        ng0, ng1 = ng, ng + dng
        self.sp = sp = loadtxt('s.txt', delimiter='\n')  # survival probability
        self.pi = genfromtxt('pi.csv', delimiter=',')  # productivity transition probability
        # muyz[y] : distribution of productivity of y-yrs agents
        self.muyz = genfromtxt('muyz.csv', delimiter=',')
        self.ef = genfromtxt('ef.csv',delimiter=',')  # efficiency of i-productivity and j-age
        self.ly = ly = ef.shape[0]
        m0 = array([prod(sp[:y+1])/ng0**y for y in range(ly)], dtype=float)
        m1 = array([prod(sp[:y+1])/ng1**y for y in range(ly)], dtype=float)
        self.pop = array([m1*ng1**t for t in range(T)], dtype=float)
        for t in range(min(T,ly-1)):
            self.pop[t,t+1:] = m0[t+1:]*ng0**t
        """Construct containers for market prices, tax rates, transfers, other aggregate variables"""
        self.Ls = array([sum([muyz[y].dot(ef[y])*self.pop[t,y]]) for t in range(T)], dtype=float)
        self.Kd = array([((r+delta)/alpha)**(1.0/(alpha-1.0))*Ls[t] for t in range(T)], dtype=float)
        self.w = array([((r+delta)/alpha)**(alpha/(alpha-1.0))*(alpha-1.0) for t in range(T)], dtype=float)

        self.Pt = Pt = array([sum(self.pop[t]) for t in range(TS)], dtype=float)
        self.Pr = Pr = array([sum([self.pop[t,y] for y in range(W,T)]) for t in range(TS)], dtype=float)
        self.tr = array([tr for t in range(TS)], dtype=float)
        self.tw = array([tw for t in range(TS)], dtype=float)
        self.gy = array([gy for t in range(TS)], dtype=float)
        self.k = array([k for t in range(TS)], dtype=float)

        self.K = K = array([k*L[t] for t in range(TS)], dtype=float)
        self.C = C = array([0 for t in range(TS)], dtype=float)
        self.Beq = Beq = array([K[t]/Pt[t]*sum((1-self.sp[t])*self.pop[t]) for t in range(TS)], dtype=float)
        self.y = y = array([k**alpha for t in range(TS)], dtype=float)
        self.r = r = array([(alpha)*k**(alpha-1) - delta for t in range(TS)], dtype=float)
        self.w = w = array([(1-alpha)*k**alpha for t in range(TS)], dtype=float)
        self.tb = tb = array([zeta*(1-tw)*Pr[t]/(L[t]+zeta*Pr[t]) for t in range(TS)], dtype=float)
        self.b = array([zeta*(1-tw-tb[t])*w[t] for t in range(TS)], dtype=float)
        self.Tax = Tax = array([tw*w[t]*L[t]+tr*r[t]*K[t] for t in range(TS)], dtype=float)
        self.G = G = array([gy*y[t]*L[t] for t in range(TS)], dtype=float)
        self.Tr = array([(Tax[t]+Beq[t]-G[t])/Pt[t] for t in range(TS)], dtype=float)
        # container for r, w, b, tr, tw, tb, Tr
        self.p = array([self.r, self.w, self.b, self.tr, self.tw, self.tb, self.Tr])
        # whether the capital stock has converged
        self.Converged = False


    def aggregate(self, gs):
        """Aggregate Capital, Labor in Efficiency unit and Bequest over all cohorts"""
        W, T, TS = self.W, self.T, self.TS
        """Aggregate all cohorts' capital and labor supply at each year"""
        K1, L1 = array([[0 for t in range(TS)] for i in range(2)], dtype=float)
        for t in range(TS):
            if t <= TS-T-1:
                K1[t] = sum([gs[t+y].apath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                L1[t] = sum([gs[t+y].epath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
                self.Beq[t] = sum([gs[t+y].apath[-(y+1)]*self.pop[t,-(y+1)]
                                    /self.sp[t,-(y+1)]*(1-self.sp[t,-(y+1)]) for y in range(T)])
                self.C[t] = sum([gs[t+y].cpath[-(y+1)]*self.pop[t,-(y+1)] for y in range(T)])
            else:
                K1[t] = sum([gs[-1].apath[-(y+1)]*self.pop[t][-(y+1)] for y in range(T)])
                L1[t] = sum([gs[-1].epath[-(y+1)]*self.pop[t][-(y+1)] for y in range(T)])
                self.Beq[t] = sum([gs[-1].apath[-(y+1)]*self.pop[t,-(y+1)]
                                    /self.sp[t,-(y+1)]*(1-self.sp[t,-(y+1)]) for y in range(T)])
                self.C[t] = sum([gs[-1].cpath[-(y+1)]*self.pop[t][-(y+1)] for y in range(T)])
        self.Converged = (max(absolute(K1-self.K)) < self.tol*max(absolute(self.K)))
        """ Update the economy's aggregate K and N with weight phi on the old """
        self.K = self.phi*self.K + (1-self.phi)*K1
        self.L = self.phi*self.L + (1-self.phi)*L1
        self.k = self.K/self.L
        # print "K=%2.2f," %(self.K[0]),"L=%2.2f," %(self.L[0]),"K/L=%2.2f" %(self.k[0])
        for i in range(self.TS/self.T):
            print "K=%2.2f," %(self.K[i*self.T]),"L=%2.2f," %(self.L[i*self.T]),"K/L=%2.2f" %(self.k[i*self.T])


    def update(self):
        """ Update market prices, w and r, and many others according to new
        aggregate capital and labor paths for years 0,...,TS from last iteration """
        for t in range(self.TS):
            self.Pt[t] = sum(self.pop[t])
            self.Pr[t] = sum([self.pop[t,y] for y in range(self.W,self.T)])
            self.y[t] = self.k[t]**self.alpha
            self.r[t] = max(0.01,self.alpha*self.k[t]**(self.alpha-1)-self.delta)
            self.w[t] = (1-self.alpha)*self.k[t]**self.alpha
            self.Tax[t] = self.tw[t]*self.w[t]*self.L[t] + self.tr[t]*self.r[t]*self.k[t]*self.L[t]
            self.G[t] = self.gy[t]*self.y[t]*self.L[t]
            self.Tr[t] = (self.Tax[t] + self.Beq[t] - self.G[t])/self.Pt[t]
            self.tb[t] = self.zeta*(1-self.tw[t])*self.Pr[t]/(self.L[t]+self.zeta*self.Pr[t])
            self.b[t] = self.zeta*(1-self.tw[t]-self.tb[t])*self.w[t]
        # print "for r=%2.2f," %(self.r[0]*100), "w=%2.2f," %(self.w[0]), \
        #         "Tr=%2.2f," %(self.Tr[0]), "b=%2.2f," %(self.b[0]), "Beq.=%2.2f," %(self.Beq[0])
        for i in range(self.TS/self.T):
            print "r=%2.2f," %(self.r[i*self.T]*100),"w=%2.2f," %(self.w[i*self.T]),\
                    "Tr=%2.2f" %(self.Tr[i*self.T]), "b=%2.2f," %(self.b[i*self.T])
        self.p = array([self.r, self.w, self.b, self.tr, self.tw, self.tb, self.Tr])


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
        self.s = loadtxt('s.txt', delimiter='\n')  # survival probability
        self.mu0 = loadtxt('mu0.txt', delimiter='\n')  # initial distribution of productivity
        self.pi = genfromtxt('pi.csv', delimiter=',')  # productivity transition probability
        self.ef = loadtxt('ef.txt', delimiter='\n')
        self.zN = self.pi.shape[0]
        """ value function and its interpolation """
        self.v = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(T)], dtype=float)
        #self.vtilde = [[] for y in range(T)]
        """ policy functions used in value function method """
        self.a = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(T)], dtype=float)
        self.c = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(T)], dtype=float)
        """ distribution of agents w.r.t. age, productivity and asset """
        self.mu = array([[[0 for a in range(aN)] for z in range(zN)] for y in range(T)], dtype=float)


    def yza2a(self, p):
        """ Given prices, transfers, benefits and tax rates over one's life-cycle,
        value and decision functions are calculated ***BACKWARD*** """
        [r, w, b, Bq, theta, tau] = p
        ef = self.ef
        T = self.T
        # y = -1 : the oldest generation
        for j in range(self.zN):
            for i in range(self.aN)
                self.c[-1,j,i] = max(self.neg, self.aa[i]*(1+(1-tau)*r[-1])
                                 + w[-1]*ef[-1,j]*(1-theta-tau) + b[-1] + Bq[-1])
                self.v[-1,j,i] = self.util(self.c[-1,j,i])
                #self.vtilde[-1] = interp1d(self.aa, self.v[-1], kind='cubic')
        # y = -2, -3,..., -60
        for y in range(-2, -(T+1), -1):
            for j in range(zN):
                # print self.apath
                m0 = 0
                for i in range(self.aN):    # l = 0, 1, ..., 50
                    # Find a bracket within which optimal a' lies
                    m = max(0, m0)  # Rch91v.g uses m = max(0, m0-1)
                    m0, a0, b0, c0 = self.GetBracket(y, j, i, m, p)
                    # Find optimal a' using Golden Section Search
                    if a0 == b0:
                        self.a[y,j,i] = 0
                    elif b0 == c0:
                        self.a[y,j,i] = self.aN
                    else:
                        self.a[y,j,i] = m0
                    # Find c and v given current and next period asset aa[i], aa[ni]
                    self.c[y,j,i] = self.aa[i]*(1+(1-tau)*r[y]) \
                                    + w[y]*ef[y,j]*(1-theta-tau) + b[y] + Bq[y] \
                                    - self.aa[self.a[y,j,i]]
                    nv = sum([self.v[y+1,nj,self.a[y,j,i]]*self.pi[j,nj]
                                                    for nj in range(self.zN)])
                    self.v[y,j,i] = self.util(self.c[y,j,i]) + self.beta*self.s[y+1]*nv


    def mu(self):
        """ find distribution of agents w.r.t. age, productivity and asset """
        a, pi = self.a, self.pi
        self.mu[0,:,0] = self.mu0
        for y in range(1,self.T):
            for j in range(self.zN):
                for i in range(self.aN):
                    self.mu[y,:,a[y-1,j,i]] = self.mu[y,:,a[y-1,j,i]] \
                                                + self.mu[y-1,j,i]*pi[j]


    def GetBracket(self, y, j, i, m, p):
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
            v1 = self.yzaa2v(y, j, i, m, p)
            if v1 > v0:
                a, b, = ([aa[m], aa[m]] if m == minit else [aa[m-1], aa[m]])
                v0, m0 = v1, m
            else:
                c = aa[m]
            if m == self.aN - 1:
                a, b, c = aa[m-1], aa[m], aa[m]
            m = m + 1
        return m0, a, b, c


    def yzaa2v(self, y, j, i, ni, p):
        """ Return the value at the given age y, productivity z[j] and asset aa[i]
        when the agent chooses his next period asset aa[ni] and current consumption c
        a1 is always within aL and aH """
        [r, w, b, Bq, theta, tau] = p
        c = max(self.neg, self.aa[i]*(1+(1-tau)*r[y])
                + w[y]*ef[y,j]*(1-theta-tau) + b[y] + Bq[y] - self.aa[ni])
        nv = sum([self.v[y+1,nj,ni]*self.pi[j,nj] for nj in range(self.zN)])
        return self.util(c) + self.beta*self.s[y+1]*nv


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
    findinitial()
    transition()
