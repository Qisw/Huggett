# -*- coding: utf-8 -*-
"""
Jul. 7, 2015, Hyun Chang Yi
Huggett (1996) "Wealth distribution in life-cycle economies," Journal of Monetary
Economics, 38(3), 469-494.
"""

from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize_scalar, broyden1, broyden2
from scipy.fftpack import rfft, rfftfreq, irfft
from scipy.signal import savgol_filter
import statsmodels.api as sm
from numpy import linspace, mean, array, zeros, absolute, loadtxt, dot, prod, int,\
                    genfromtxt, sum, argmax, tile, concatenate, ones, log, \
                    unravel_index, cumsum, meshgrid, atleast_2d, where, newaxis,\
                    maximum, minimum, repeat
from matplotlib import pyplot as plt
from datetime import datetime
import time
import pickle
import os
from platform import system
from multiprocessing import Process, Lock, Manager
from multiprocessing.sharedctypes import Value, Array, RawArray
from ctypes import Structure, c_double


class params:
    """ This class is just a "struct" to hold the collection of PARAMETER values """
    def __init__(self, T=1, pg=1.012, pg_change=0.0,
        alpha=0.36, delta=0.08, tau=0.2378, theta=0.1, zeta=0.3,
        beta=0.994, sigma=1.5, W=45, R=34, a0=0,
        aH=50.0, aL=0.0, aN=200, hN=5, hL=0.1, hH=1.0,
        psi=0.1, phi=0.5, dti=0.5, ltv=0.7, sp_multiplier=0.1,
        savgol_windows=71, savgol_order=1, filter_on=1, lowess_frac=0.05,
        tcost=0.02, Hs=7, tol=1e-2, eps=0.2, neg=-1e10, gs=1.5):
        self.savgol_windows, self.savgol_order = savgol_windows, savgol_order
        self.lowess_frac = lowess_frac
        self.filter_on = filter_on
        self.alpha, self.zeta, self.delta, self.tau = alpha, zeta, delta, tau
        self.theta, self.psi = theta, psi
        self.tcost, self.dti, self.ltv = tcost, dti, ltv
        self.beta, self.sigma = beta, sigma
        self.R, self.W = R, W
        self.aH, self.aL = aH, aL
        am = [-aH*linspace(0,1,aN)**gs][0][int(-aN*aL/(aH-aL)):0:-1]
        ap = aH*linspace(0,1,aN)**gs
        self.aa = concatenate((am,ap))
        self.aN = len(self.aa)
        # self.aa = aL+aH*linspace(0,1,aN)
        self.phi, self.tol, self.neg, self.eps = phi, tol, neg, eps
        self.hh = linspace(hL,hH,hN)   # hh = [0, 1, 2, 3, 4]
        self.hN = hN
        self.pg, self.pg_change = pg, pg_change
        self.Hs = Hs
        """ LOAD PARAMETERS : SURVIVAL PROB., INITIAL DIST. OF PRODUCTIVITY,
        PRODUCTIVITY TRANSITION PROB. AND PRODUCTIVITY """
        if system() == 'Windows':
            self.sp0 = loadtxt('parameters\sp.txt', delimiter='\n')
            self.muz = genfromtxt('parameters\muz.csv', delimiter=',')
            self.pi = genfromtxt('parameters\pi.csv', delimiter=',')
            self.ef = genfromtxt('parameters\ef.csv', delimiter=',')
        else:
            self.sp0 = loadtxt('parameters/sp.txt', delimiter='\n')
            self.muz = genfromtxt('parameters/muz.csv', delimiter=',')
            self.pi = genfromtxt('parameters/pi.csv', delimiter=',')
            self.ef = genfromtxt('parameters/ef.csv', delimiter=',')
        self.sp1 = self.sp0 + sp_multiplier*(1-self.sp0)
        self.zN = self.pi.shape[0]
        self.yN = self.ef.shape[0]
        self.T = T


    def print_params(self):
        print '\n===================== Parameters ====================='
        if self.T==1:
            print 'Finding Steady State witn tol. %2.2f%%... \n'%(self.tol*100)
            print 'Birth Rate is %2.2f%%'%((sum(self.pg)-1)*100),\
                    'and survival prob. is %2.2f%%'%(prod(self.sp1)*100)
        else:
            print 'Finding Transition Path over %i periods ... \n'%(self.T)
            if self.pg_change != 0:
                print 'Birth Rate Changes from %2.2f%%'%((sum(self.pg)-1)*100),\
                        'to %2.2f%%'%((sum(self.pg+self.pg_change)-1)*100)
            if prod(self.sp0) != prod(self.sp1):
                print 'survival prob. rises from %2.2f%%'%(prod(self.sp0)*100),\
                        'to %2.2f%%'%(prod(self.sp1)*100)
        print 'Liquid Asset: From %i'%(self.aL),'to %i'%(self.aH),\
                ' with Grid Size %i'%(self.aN)
        print '       House: From %2.2f'%(self.hh[0]),'to %2.2f'%(self.hh[-1]),\
                ' with Grid Size %i'%(self.hN)
        print 'House Supply per Capita: %2.2f'%(self.Hs)
        print '\n','psi  :%2.2f'%(self.psi), ' eps  :%2.2f'%(self.eps)\
                , ' phi:%2.2f'%(self.phi), ' htc:%2.0f%%'%(self.tcost*100)\
                , ' sp:%2.2f%%'%(prod(self.sp1)*100), '\n'\
                , 'delta:%2.0f%%'%(self.delta*100), ' alpha:%2.2f'%(self.alpha)\
                , ' dti:%2.0f%%'%(self.dti*100), ' ltv:%2.0f%%'%(self.ltv*100)\
                , ' beta:%2.2f'%(self.beta)
        print '====================================================== \n'






class state:
    """ This class is just a "struct" to hold the collection of primitives defining
    an economy in which one or multiple generations live """
    def __init__(self, params, r_init=0.05, r_term=0.05, Bq_init=0, Bq_term=0,
        q_init=4.6, q_term=3.3):
        # tr = 0.429, tw = 0.248, zeta=0.5, gy = 0.195, in Section 9.3. in Heer/Maussner
        """tr, tw and tb are tax rates on capital return, wage and tax for pension.
        tb is determined by replacement ratio, zeta, and other endogenous variables.
        gy is ratio of government spending over output.
        Transfer from government to households, Tr, is determined endogenously"""
        self.zeta = params.zeta
        self.psi = params.psi # this parameter is used for plot_lp
        self.delta = delta = params.delta
        self.alpha = alpha = params.alpha
        self.T = T = params.T
        self.phi, self.tol, self.eps = params.phi, params.tol, params.eps
        self.aN = aN = params.aN
        self.aa = aa = params.aa
        self.hh = hh = params.hh
        self.hN = hN = params.hN
        """ PRODUCTIVITY TRANSITION PROB. AND ... """
        self.pi = pi = params.pi
        # muz[y] : distribution of productivity of y-yrs agents
        self.muz = muz = params.muz
        self.ef = ef = params.ef
        self.zN = zN = params.zN
        self.yN = yN = params.yN
        self.ng_init = ng_init = params.pg
        self.ng_term = ng_term = params.pg + params.pg_change
        self.savgol_windows, self.savgol_order = params.savgol_windows, params.savgol_order
        self.lowess_frac = params.lowess_frac
        self.filter_on = params.filter_on
        self.sp0 = params.sp0
        self.sp1 = params.sp1
        if T==1:
            ng_term, r_term, q_term, Bq_term = ng_init, r_init, q_init, Bq_init
        self.r_init = r_init
        self.q_init = q_init
        self.r_term = r_term
        self.q_term = q_term
        """ CALCULATE POPULATIONS OVER THE TRANSITION PATH
        In period 0, survival probability and birth rate change are announced
        from ng_init and sp0 to ng_term and sp1 so that
        p(p(1,0)  """
        m0 = array([prod(self.sp0[:y+1])/ng_init**y for y in range(self.yN)], dtype=float)
        m1 = array([prod(self.sp1[:y+1])/ng_term**y for y in range(self.yN)], dtype=float)
        self.pop = array([m1*ng_term**t for t in range(T)], dtype=float)
        pm = m0[1:]
        sp = self.sp1[1:]
        for t in range(min(T,self.yN-1)):
            self.pop[t,t+1:] = pm
            sp = sp[1:]
            pm = pm[:-1]*sp
            # If sp_multiplier = 0, the following line works the same way
            # self.pop[t,t+1:] = m0[t+1:]*ng_init**t
        """Construct containers for market prices, tax rates, pension, bequest"""
        self.Hs = [params.Hs*sum(self.pop[t]) for t in range(T)]
        self.theta = params.theta*ones(T)
        self.tau = params.tau*ones(T)
        self.r = r_term*ones(T)
        self.q = q_term*ones(T)
        self.Bq = Bq_term*ones(T)
        self.r[0:T-yN] = linspace(r_init,r_term,T-yN)
        self.q[0:T-yN] = linspace(q_init,q_term,T-yN)
        self.Bq[0:T-yN] = linspace(Bq_init,Bq_term,T-yN)
        self.pr, self.L, self.K, self.w, self.b = [zeros(T) for i in range(5)]
        for t in range(T):
            # pr = population of retired agents
            self.pr[t] = sum(self.pop[t,45:])
            # L = labor supply in efficiency unit
            self.L[t] = sum([muz[y].dot(ef[y])*self.pop[t,y] for y in range(yN)])
            self.K[t] = ((self.r[t]+delta)/alpha)**(1.0/(alpha-1.0))*self.L[t]
            self.w[t] = ((self.r[t]+delta)/alpha)**(alpha/(alpha-1.0))*(1.0-alpha)
            self.b[t] = self.theta[t]*self.w[t]*self.L[t]/self.pr[t]
        self.Hd = zeros(T)
        self.r1 = zeros(T)
        self.K1 = zeros(T)
        self.Bq1 = zeros(T)
        self.Prod = zeros(T)
        self.Cons = zeros(T)
        self.Debt = zeros(T)
        self.DI = zeros(T)
        """ PRICES, PENSION BENEFITS, BEQUESTS AND TAXES
        that are observed by households """
        self.prices = array([self.r, self.w, self.q, self.b, self.Bq, self.theta, self.tau])
        self.mu = [zeros((yN,hN,zN,aN)) for t in range(T)]
        self.c = [zeros((yN,hN,zN,aN)) for t in range(T)]


    def aggregate(self, vmu, vc):
        """Aggregate Capital, Labor in Efficiency unit and Bequest over all cohorts"""
        T, yN, alpha, delta, zeta = self.T, self.yN, self.alpha, self.delta, self.zeta
        aa, pop, sp1, zN, tau = self.aa, self.pop, self.sp1, self.zN, self.tau
        aN, hN, hh = self.aN, self.hN, self.hh
        spr = (1-sp1)/sp1
        my = lambda x: x if x < T-1 else -1
        self.mu = [array(vmu[t]).reshape(yN,hN,zN,aN) for t in range(len(vmu))]
        self.c = [array(vc[t]).reshape(yN,hN,zN,aN) for t in range(len(vc))]
        self.Hd = zeros(T)
        self.K1 = zeros(T)
        self.Bq1 = zeros(T)
        self.Prod = zeros(T)
        self.Cons = zeros(T)
        self.Debt = zeros(T)
        """Aggregate all cohorts' capital and labor supply at each year"""
        for t in range(T):
            for y in range(yN):
                k1 = sum(self.mu[my(t+y)][-(y+1)],(0,1)).dot(aa)*pop[t,-(y+1)]
                hd = sum(self.mu[my(t+y)][-(y+1)],(1,2)).dot(hh)*pop[t,-(y+1)]
                bq1 = (k1 + hd*self.q[t])*spr[-(y+1)]*(1-zeta)/sum(pop[t])
                cons = sum(self.mu[my(t+y)][-(y+1)]*self.c[my(t+y)][-(y+1)])*pop[t,-(y+1)]
                debt = sum(self.mu[my(t+y)][-(y+1)],(0,1)).dot(minimum(aa,0))*pop[t,-(y+1)]
                self.K1[t] += k1
                self.Hd[t] += hd
                self.Bq1[t] += bq1
                self.Cons[t] += cons
                self.Debt[t] += debt
        self.r1 = alpha*(self.K1/self.L)**(alpha-1.0)-delta
        self.Prod = self.K1**alpha*self.L**(1.0-alpha)
        self.DI = self.K1*(1-tau)*self.r + self.Bq1*sum(pop,axis=1) + self.w*self.L*(1-tau)


    def update_prices(self, n=0):
        """ Update market prices, w and r, and many others according to new
        aggregate capital and labor paths for years 0,...,T from last iteration """
        alpha, delta = self.alpha, self.delta
        self.r = self.phi*self.r + (1-self.phi)*self.r1
        self.K = ((self.r+delta)/alpha)**(1.0/(alpha-1.0))*self.L
        self.w = ((self.r+delta)/alpha)**(alpha/(alpha-1.0))*(1.0-alpha)
        self.b = self.theta*self.w*self.L/self.pr
        self.Bq = self.phi*self.Bq + (1-self.phi)*self.Bq1
        self.q = self.q*(1+self.eps*(self.Hd-self.Hs))
        rmin = min(self.r_init,self.r_term)-0.04
        rmax = max(self.r_init,self.r_term)+0.04
        qmin = min(self.q_init,self.q_term)-4
        qmax = max(self.q_init,self.q_term)+4
        if self.T > 1:
            r0 = self.r
            q0 = self.q
            if self.filter_on == 1:
                r1 = concatenate((self.r_init*ones(30),r0))
                q1 = concatenate((self.q_init*ones(30),q0))
                self.r = savgol_filter(r1, self.savgol_windows, self.savgol_order)[30:]
                self.q = savgol_filter(q1, self.savgol_windows, self.savgol_order)[30:]
            title = "Transition Paths after %i iterations"%(n)
            filename = title + '.png'
            fig = plt.figure(facecolor='white')
            plt.rcParams.update({'font.size': 8})
            ax = fig.add_subplot(111)
            ax1 = fig.add_subplot(231)
            ax2 = fig.add_subplot(232)
            ax3 = fig.add_subplot(234)
            ax4 = fig.add_subplot(235)
            ax5 = fig.add_subplot(233)
            ax6 = fig.add_subplot(236)
            fig.subplots_adjust(hspace=.5, wspace=.3, left=None, right=None,
                                    top=None, bottom=None)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top='off', bottom='off', left='off',
                            right='off')
            ax1.plot(self.K1/sum(self.pop,axis=1))
            ax2.plot(self.Hd/sum(self.pop,axis=1),label='Demand')
            ax2.plot(self.Hs/sum(self.pop,axis=1),label='Supply')
            ax3.plot(self.r,label='smoothed')
            ax3.plot(r0,label='updated')
            ax3.plot(0,self.r_init,'o',label='initial')
            ax4.plot(self.q,label='smoothed')
            ax4.plot(q0,label='updated')
            ax4.plot(0,self.q_init,'o',label='initial')
            ax5.plot(self.Prod/sum(self.pop,axis=1),label='per capita Production')
            ax5.plot(self.Cons/sum(self.pop,axis=1),label='per capita Consumption')
            ax6.plot((self.DI-self.Cons)/self.DI)
            ax2.legend(prop={'size':7})
            ax3.legend(prop={'size':7})
            ax4.legend(prop={'size':7})
            ax5.legend(prop={'size':7})
            # ax6.legend(prop={'size':7})
            ax1.axis([0, self.T, 0, 2])
            ax2.axis([0, self.T, 0, 1])
            ax3.axis([0, self.T, rmin, rmax])
            ax4.axis([0, self.T, qmin, qmax])
            ax5.axis([0, self.T, 0, 1.0])
            ax6.axis([0, self.T, 0.0, 0.4])
            ax.set_title('Transition over %i periods'%(self.T), y=1.08)
            ax1.set_title('per capita Liquid Asset')
            ax2.set_title('per capita House Demand')
            ax3.set_title('Interest Rate')
            ax4.set_title('House Price')
            ax5.set_title('Production and Consumption')
            ax6.set_title('Saving Rate')
            if system() == 'Windows':
                path = 'D:\Huggett\Figs'
            else:
                path = '/Users/hyunchangyi/GitHub/Huggett/Figs'
            fullpath = os.path.join(path, filename)
            fig.savefig(fullpath)
            plt.close()
        self.prices[0] = self.r
        self.prices[1] = self.w
        self.prices[3] = self.b
        self.prices[4] = self.Bq
        self.prices[2] = self.q


    def converged(self):
        return max(absolute(self.K - self.K1))/max(self.K) < self.tol \
                and max(absolute(self.Hd - self.Hs))/max(self.Hd) < self.tol


    def print_prices(self, n=0, t=0):
        print "n=%2i"%(n)," t=%3i"%(t),"r=%2.2f%%"%(self.r[t]*100),\
              "Pop.=%3.1f"%(sum(self.pop[t])),\
              "Ks=%3.1f,"%(self.K1[t]),"q=%2.2f,"%(self.q[t]),\
              "Hd=%3.1f%%,"%((self.Hd[t]-self.Hs[t])/self.Hs[t]*100),\
              "Bq=%2.2f," %(self.Bq1[t])


    def plot(self, t=0, yi=0, yt=78, ny=7):
        """plot life-path of aggregate capital accumulation and house demand"""
        yN = self.yN
        pop, aa, hh, aN, hN = self.pop, self.aa, self.hh, self.aN, self.hN
        mu = self.mu[t]
        a = zeros(yN)
        h = zeros(yN)
        ap = zeros(yN)
        hp = zeros(yN)
        al = zeros(aN)
        hl = zeros(hN)
        ah = zeros((hN,aN))
        """Aggregate all cohorts' capital and labor supply at each year"""
        for y in range(yN):
            ap[y] = sum(mu[y],(0,1)).dot(aa)*pop[t,y]
            hp[y] = sum(mu[y],(1,2)).dot(hh)*pop[t,y]
            a[y] = sum(mu[y],(0,1)).dot(aa)
            h[y] = sum(mu[y],(1,2)).dot(hh)
            al += sum(mu[y],(0,1))*pop[t,y]
            hl += sum(mu[y],(1,2))*pop[t,y]
            """ ah: hN by aN matrix that represents populations of each pairs
            of house and asset holders """
            ah += sum(mu[y],1)*pop[t,y]
        w = hh[:,newaxis]*self.q[t] + aa[newaxis,:]
        unsorted = array((ah.ravel(),w.ravel())).T
        ah, w = unsorted[unsorted[:,1].argsort()].T
        title = 'psi=%2.2f'%(self.psi) + ' aN=%2.2f'%(self.aN) \
                + ' hN=%2.2f'%(self.hN) + ' r=%2.2f%%'%(self.r[t]*100) \
                + ' q=%2.2f'%(self.q[t]) + ' K=%2.1f'%(self.K[t]) \
                + ' Hd=%2.1f'%(self.Hd[t])
        if self.T == 1:
            title = 'In SS, ' + title
        else:
            title = 'In Trans., at %i '%(t) + title
        filename = title + '.png'
        fig = plt.figure(facecolor='white')
        plt.rcParams.update({'font.size': 8})
        # matplotlib.rcParams.update({'font.size': 22})
        ax = fig.add_subplot(111)
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(234)
        ax3 = fig.add_subplot(232)
        ax4 = fig.add_subplot(235)
        ax5 = fig.add_subplot(233)
        ax6 = fig.add_subplot(236)
        fig.subplots_adjust(hspace=.5, wspace=.3, left=None, right=None, \
                                                        top=None, bottom=None)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', \
                                                        left='off', right='off')
        ax1.plot(range(yN),ap,label='aggregate')
        ax1.plot(range(yN),a,label='per capita')
        ax2.plot(range(yN),hp,label='aggregate')
        ax2.plot(range(yN),h,label='per capita')
        for y in linspace(yi,yt,ny).astype(int):
            ax3.plot(aa,sum(mu[y],(0,1)),label='age %i'%(y))
        for y in linspace(yi,yt,ny).astype(int):
            ax4.plot(hh,sum(mu[y],(1,2)),label='age %i'%(y))
        ax5.plot(cumsum(al)/sum(al),cumsum(aa*al)/sum(aa*al),".")
        # ax6.plot(cumsum(hl)/sum(hl),cumsum(hh*hl)/sum(hh*hl),".")
        ax6.plot(cumsum(ah)/sum(ah),cumsum(ah*w)/sum(ah*w),".")
        # ax1.legend(bbox_to_anchor=(0.9,1.0),loc='center',prop={'size':8})
        ax1.legend(prop={'size':7})
        ax2.legend(prop={'size':7})
        ax3.legend(prop={'size':7})
        ax4.legend(prop={'size':7})
        # ax3.axis([0, 15, 0, 0.1])
        ax5.axis([0, 1, 0, 1])
        ax6.axis([0, 1, 0, 1])
        # ax4.axis([0, 80, 0, 1.0])
        ax1.set_xlabel('Age')
        ax2.set_xlabel('Age')
        ax3.set_xlabel('Asset Size')
        ax4.set_xlabel('House Size')
        ax5.set_xlabel('Cum. Share of Agents from Lower to Higher')
        ax6.set_xlabel('Cum. Share of Agents from Lower to Higher')
        ax5.set_ylabel('Cum. Share of Asset Occupied')
        # ax6.set_ylabel('Cum. Share of House Occupied')
        ax6.set_ylabel('Cum. Share of Total Wealth')
        ax.set_title(title, y=1.08)
        ax1.set_title('Life-Cycle Liquid Asset Accumulation')
        ax2.set_title('Life-Cycle House Size')
        ax3.set_title('Dist. of Liquid Asset w/i Cohort')
        ax4.set_title('Dist. of House Size w/i Cohort')
        ax5.set_title('Lorenz Curve for Liquid Asset')
        # ax6.set_title('Lorenz Curve for House')
        ax6.set_title('Lorenz Curve for Total Wealth')
        if system() == 'Windows':
            path = 'D:\Huggett\Figs'
        else:
            path = '/Users/hyunchangyi/GitHub/Huggett/Figs'
        fullpath = os.path.join(path, filename)
        fig.savefig(fullpath)
        # ax4.axis([0, 80, 0, 1.1])
        # plt.show()
        plt.close()






class cohort:
    """ This class is just a "struct" to hold the collection of primitives defining
    a generation """
    def __init__(self, params, y=-1, a0 = 0):
        self.beta, self.sigma, self.psi = params.beta, params.sigma, params.psi
        self.R, self.W, self.y = params.R, params.W, y
        # self.yN = yN = (y+1 if (y >= 0) and (y <= W+R-2) else W+R)
        self.aN = aN = params.aN
        self.aa = aa = params.aa
        # agents start their life with asset aa[a0_id]
        self.a0_id = where(aa >= 0)[0][0]
        self.hh = hh = params.hh
        self.hN = hN = params.hN
        self.tol, self.neg = params.tol, params.neg
        self.tcost = params.tcost
        self.ltv = params.ltv
        """ SURVIVAL PROB., PRODUCTIVITY TRANSITION PROB. AND ... """
        self.sp1 = sp1 = params.sp1
        self.pi = pi = params.pi
        # muz[y] : distribution of productivity of y-yrs agents
        self.muz = muz = params.muz
        self.ef = ef = params.ef
        self.zN = zN = params.zN
        self.yN = yN = params.yN
        """ container for value function and expected value function """
        # v[y,h,j,i] is the value of y-yrs-old agent with asset i and prod. j, house h
        self.v = zeros((yN,hN,zN,aN))
        """ container for policy functions,
        which are used to calculate vmu and not stored """
        self.a = zeros((yN,hN,zN,aN))
        self.h = zeros((yN,hN,zN,aN))
        # self.c = zeros((yN,hN,zN,aN))
        """ distribution of agents w.r.t. age, productivity and asset
        for each age, distribution over all productivities and assets add up to 1 """
        self.vmu = zeros(yN*hN*zN*aN)
        self.vc = zeros(yN*hN*zN*aN)


    def optimalpolicy(self, prices):
        """ Given prices, transfers, benefits and tax rates over one's life-cycle,
        value and decision functions are calculated ***BACKWARD*** """
        aa, hh = self.aa, self.hh
        t = prices.shape[1]
        if t < self.yN:
            d = self.yN - t
            # prices = concatenate((tile(array([prices[:,0]]).T,(1,d)),prices), axis=1)
            prices = concatenate((repeat(prices[:,0][:,newaxis],d,axis=1),prices), axis=1)
        r, w, q, b, Bq, theta, tau = prices
        ef, yN, R, aN, zN, hN = self.ef, self.yN, self.R, self.aN, self.zN, self.hN
        sigma, psi, beta = self.sigma, self.psi, self.beta
        sp1, pi, tcost, ltv, neg = self.sp1, self.pi, self.tcost, self.ltv, self.neg
        """ev[y,nh,j,ni] is the expected value when next period asset ni and house hi"""
        ev = zeros((yN,hN,zN,aN))
        """ct is a channel to store optimal consumption in vc"""
        ct = self.vc.reshape(yN,hN,zN,aN)
        """ inline functions: utility and income adjustment by trading house """
        util = lambda c, h: (c*h**psi)**(1-self.sigma)/(1-self.sigma)
        hinc = lambda h, nh, q: (h-nh)*q - tcost*h*q*(h!=nh)
        """ y = -1 : just before the agent dies """
        for h in range(hN):
            for z in range(zN):
                c = aa*(1+(1-tau[-1])*r[-1]) + hinc(hh[h],hh[0],q[-1])\
                        + w[-1]*ef[-1,z]*(1-theta[-1]-tau[-1]) + b[-1] + Bq[-1]
                c[c<=0.0] = 1e-10
                ct[-1,h,z] = c
                self.v[-1,h,z] = util(c,hh[h])
            ev[-1,h] = pi.dot(self.v[-1,h])
        """ y = -2, -3,..., -60 """
        for y in range(-2, -(yN+1), -1):
            for h in range(hN):
                for z in range(zN):
                    vt = zeros((hN,aN,aN))
                    for nh in range(hN):
                        p = aa*(1+(1-tau[y])*r[y]) + b[y]*(y>=-R) \
                                + w[y]*ef[y,z]*(1-theta[y]-tau[y]) + Bq[y] \
                                + hinc(hh[h],hh[nh],q[y])
                        c = p[:,newaxis] - aa
                        c[c<=0.0] = 1e-10
                        vt[nh] = util(c,hh[h]) + beta*sp1[y+1]*ev[y+1,nh,z] \
                                    + neg*(-ltv*hh[nh]*q[y]>aa)
                    for a in range(aN):
                        """find optimal pairs of house and asset """
                        self.h[y,h,z,a], self.a[y,h,z,a] \
                            = unravel_index(vt[:,a,:].argmax(),vt[:,a,:].shape)
                        self.v[y,h,z,a] = vt[self.h[y,h,z,a],a,self.a[y,h,z,a]]
                        ct[y,h,z,a] = aa[a]*(1+(1-tau[y])*r[y]) + b[y]*(y>=-R) \
                                        + Bq[y] + w[y]*ef[y,z]*(1-theta[y]-tau[y]) \
                                        + hinc(hh[h],hh[self.h[y,h,z,a]],q[y]) \
                                        - aa[self.a[y,h,z,a]]
                ev[y,h] = pi.dot(self.v[y,h])
        """ find distribution of agents w.r.t. age, productivity and asset """
        self.vmu *= 0
        mu = self.vmu.reshape(yN,hN,zN,aN)
        mu[0,0,:,self.a0_id] = self.muz[0]
        for y in range(1,yN):
            for h in range(hN):
                for z in range(zN):
                    for a in range(aN):
                        mu[y,self.h[y-1,h,z,a],:,self.a[y-1,h,z,a]] += mu[y-1,h,z,a]*pi[z]






"""The following are procedures to get steady state of the economy using direct
age-profile iteration and projection method"""

def fss(params, N=20):
    """Find Old and New Steady States with population growth rates ng and ng1"""
    start_time = datetime.now()
    params.print_params()
    c = cohort(params)
    k = state(params)
    for n in range(N):
        c.optimalpolicy(k.prices)
        k.aggregate([c.vmu],[c.vc])
        k.print_prices(n=n+1)
        k.update_prices(n=n+1)
        if k.converged():
            print 'Economy Converged to SS! in',n+1
            break
        if n >= N-1:
            print 'Economy Not Converged in',n+1
            break
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    return k, c


# separate the procedure of finding optimal policy of each cohort for Parallel Process
def sub1(t,vmu,vc,prices,c1,params):
    c = cohort(params)
    if t < params.T-1:
        c.optimalpolicy(prices[:,max(t-c.yN+1,0):t+1])
        # c.optimalpolicy(prices.T[max(t-yN+1,0):t+1].T)
    else:
        c.vmu, c.vc = c1.vmu, c1.vc
    for i in range(c.yN*c.hN*c.zN*c.aN):
        vmu[i] = c.vmu[i]
        vc[i] = c.vc[i]


def tran(params, k0, c0, k1, c1, N=5):
    params.print_params()
    T = params.T
    kt = state(params, r_init=k0.r, r_term=k1.r, q_init=k0.q, q_term=k1.q,
                                                    Bq_init=k0.Bq, Bq_term=k1.Bq)
    vl = params.yN*params.hN*params.zN*params.aN
    """Generate mu of T cohorts who die in t = 0,...,T-1 with initial asset g0.apath[-t-1]"""
    VM = [RawArray(c_double, vl) for t in range(T)]
    VC = [RawArray(c_double, vl) for t in range(T)]
    for n in range(N):
        start_time = datetime.now()
        print str(n+1)+'th loop started at {}'.format(start_time)
        jobs = []
        # for t, vmu in enumerate(VM):
        for t in range(T):
            p = Process(target=sub1, args=(t,VM[t],VC[t],kt.prices,c1,params))
            # p = Process(target=transition_sub1, args=(t,vmu,kt.prices,c1.vmu,params))
            p.start()
            jobs.append(p)
            # if t % 40 == 0:
            #     print 'processing another 40 cohorts...'
            if len(jobs) % 8 == 0:
                for p in jobs:
                    p.join()
                    jobs=[]
        if len(jobs) > 0:
            for p in jobs:
                p.join()
        kt.aggregate(VM,VC)
        for t in linspace(0,T-1,20).astype(int):
            kt.print_prices(n=n+1,t=t)
        kt.update_prices(n=n+1)
        end_time = datetime.now()
        print 'this loop finished in {}\n'.format(end_time - start_time)
        if kt.converged():
            print 'Transition Path Converged! in', n+1,'iterations.'
            break
        if n >= N-1:
            print 'Transition Path Not Converged! in', n+1,'iterations.'
            break
    return kt, VM, VC


if __name__ == '__main__':
    start_time = datetime.now()
    par = params(T=1, psi=0.5, delta=0.08, aN=60, aL=-10, aH=50,
            Hs=0.3, hN=5, tol=0.001, phi=0.75, eps=0.075, tcost=0.02, gs=2.0,
            alpha=0.36, tau=0.2378, theta=0.1, zeta=0.3, sp_multiplier=0.1,
            savgol_windows=41, savgol_order=1, filter_on=1,
            beta=0.994, sigma=1.5, dti=0.5, ltv=0.7)

    sp0, sp1 = par.sp0, par.sp1

    par.pg = 1.000
    par.sp1 = sp0
    k0, c0 = fss(par, N=30)

    par.sp0 = par.sp1 = sp1
    k1, c1 = fss(par, N=30)

    par.pg, par.pg_change = 1.000, 1.000-1.000
    par.sp0, par.sp1 = sp0, sp1
    par.T = 300
    kt, mu, vc = tran(par, k0, c0, k1, c1, N=15)

    for t in linspace(0,par.T-1,10).astype(int):
        kt.plot(t=t,yi=10,ny=5)

    end_time = datetime.now()
    print 'Total Time: {}'.format(end_time - start_time)
