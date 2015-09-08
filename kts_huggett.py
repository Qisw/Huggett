# -*- coding: utf-8 -*-

from numpy import matrix, asmatrix, asarray, transpose, zeros, ones, shape, \
                    append, dot, divide, linspace, log, tile, power, squeeze, \
                    amax, argmax, array, reshape, sum, sort, argsort,\
                  linspace
from math import sqrt, exp
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
from datetime import datetime
from matplotlib import pyplot

import pickle

import warnings
warnings.filterwarnings("ignore")


class Params:
    def __init__(self, N = 0.012):
        self.SIGMA  = 1.5
        self.BETA   = 0.994
        self.ALPHA  = 0.36
        self.DELTA  = 0.06
        self.N      = N
        self.I      = 79
        self.R      = 46
        self.UNDERa = 0
        self.UPPERa = 50
        self.Na     = 50
        self.UNDERr = 0
        self.UPPERr = 0.2
        self.Nr     = 6
        self.Nz     = 18
        self.A      = 1
        self.GAMMA  = 0.96
        self.SIGMAe = sqrt(0.045)
        self.SIGMAy = sqrt(self.SIGMAe**2/(1-self.GAMMA**2))
        self.TAU    = 0.2378
        self.THETA  = 0.1
        self.ZETA   = 0.3
        self.age_e  = array([20, 35, 55, 70])
        self.earnings_e = array([0.11931, 0.41017, 0.47964, 0.23370])
        self.age_p  = array([22.5, 30, 40, 50, 60, 70])
        self.earnings_p = array([0.796, 0.919, 0.919, 0.875, 0.687, 0.19])
        self.s = array([1.00000, 0.99962, 0.99960, 0.99958, 0.99956, 0.99954, 0.99952, 0.99950, 0.99947, 0.99945, \
                        0.99942, 0.99940, 0.99938, 0.99934, 0.99930, 0.99925, 0.99919, 0.99910, 0.99899, 0.99887, \
                        0.99875, 0.99862, 0.99848, 0.99833, 0.99816, 0.99797, 0.99775, 0.99753, 0.99731, 0.99708, \
                        0.99685, 0.99659, 0.99630, 0.99599, 0.99566, 0.99529, 0.99492, 0.99454, 0.99418, 0.99381, \
                        0.99340, 0.99291, 0.99229, 0.99150, 0.99057, 0.98952, 0.98841, 0.98719, 0.98582, 0.98422, \
                        0.98241, 0.98051, 0.97852, 0.97639, 0.97392, 0.97086, 0.96714, 0.96279, 0.95795, 0.95241, \
                        0.94646, 0.94005, 0.93274, 0.92434, 0.91518, 0.90571, 0.89558, 0.88484, 0.87352, 0.86166, \
                        0.84930, 0.83652, 0.82338, 0.80997, 0.79638, 0.78271, 0.76907, 0.75559, 0.74239, 0.00000])

        self.agrid = linspace(self.UNDERa, self.UPPERa, self.Na)
        self.rgrid = linspace(self.UNDERr, self.UPPERr, self.Nr)

        self.mu = zeros((self.I, 1))
        self.mu[0] = 1
        for i in range(0,self.I-1):
            self.mu[i+1] = (self.s[i+1]*self.mu[i])/(1+self.N)
            # print(self.mu[i])

        earnings_f = InterpolatedUnivariateSpline(self.age_e, self.earnings_e, k=1)
        self.earnings = earnings_f(range(20,self.I+20))
        self.earnings[self.earnings < 0] = 0

        participation_f = InterpolatedUnivariateSpline(self.age_p, self.earnings_p, k=1)
        self.participation = participation_f(range(20,self.I+20))
        self.participation[self.participation < 0] = 0

        self.earningsprofile = self.earnings * self.participation
        self.ybar = log(self.earningsprofile)

        # Tauchen Distribution: pi, p, e
        # self.zgrid = [0 for col in range(self.Nz)]
        # self.zgrid[0] = -4*self.SIGMAy
        # for j in range(1,self.Nz):
        #     self.zgrid[j] = self.zgrid[j-1] + 0.5*self.SIGMAy
        # self.zgrid[self.Nz-1] = 6*self.SIGMAy
        self.zgrid = linspace(-4*self.SIGMAy, 4.5*self.SIGMAy, self.Nz)
        self.zgrid[-1] = 6*self.SIGMAy

        # self.zrange = [0 for col in range(self.Nz+1)]
        # self.zrange[0] = -float("inf")
        # for j in range(1,self.Nz):
        #     self.zrange[j] = (self.zgrid[j-1] + self.zgrid[j])*0.5
        # self.zrange[self.Nz] = float("inf")
        zrange = (self.zgrid[1:] + self.zgrid[:-1])/2.0
        self.zrange = concatenate((array([-float("inf")]),zrange,array([float("inf")])),axis=0)

        self.pi = zeros((self.Nz, self.Nz))
        for i in range(self.Nz):
            for j in range(self.Nz):
                self.pi[i,j] = norm.cdf((self.zrange[j+1] - self.GAMMA*self.zgrid[i])/self.SIGMAe, 0, 1) \
                    - norm.cdf((self.zrange[j] - self.GAMMA*self.zgrid[i])/self.SIGMAe, 0, 1)

        self.p = zeros((self.Nz, self.I))
        for j in range(self.Nz):
            self.p[j,0] = norm.cdf(self.zrange[j+1], 0, self.SIGMAy) - norm.cdf(self.zrange[j], 0, self.SIGMAy)
        for j in range(1,self.I):
            self.p[:,j] = transpose(dot(transpose(self.p[:,j-1]),self.pi))

        self.e = zeros((self.Nz, self.I))
        for j in range(self.Nz):
            for i in range(self.I):
                self.e[j,i] = exp(self.ybar[i] + self.zgrid[j])


class EconomicStatus:
    def __init__(self):
        self.savingv = 0
        self.savingg = 0
        self.savingg_ordered = 0
        self.savingpop = 0
        self.KS    = 0
        self.KD    = 0
        self.r     = 0
        self.BQ    = 0
        self.BQTAX = 0
        self.w     = 0
        self.L     = 0
        self.r     = 0


class EconomicStatusMAT:
    def __init__(self):
        self.KSmat = 0
        self.KDmat = 0
        self.BQmat = 0
        self.BQTAXmat = 0
        self.rmat = 0


def FindWageETCs(params, r):

    # Aggregate Labor Supply
    L = 0
    for i in range(params.I):
        L = L + dot(asmatrix(transpose(params.p[:,i])),params.e[:,i])*params.mu[i]

    # Aggregate Capital Demand
    KD = ((r+params.DELTA)/(params.ALPHA*params.A))**(1/(params.ALPHA-1))*L

    # wage
    w = ((r+params.DELTA)/(params.ALPHA*params.A))**(params.ALPHA/(params.ALPHA-1))*(1-params.ALPHA)*params.A

    # pension for retirees
    b = zeros((params.I,1))
    b[0:params.R-1] = zeros((params.R-1,1))
    mass_after = dot(append(zeros((1,params.R-1)), ones((1, params.I-params.R+1))),params.mu)
    b[params.R-1:params.I] = dot(ones((params.I-params.R+1,1)), params.THETA*w*L/mass_after)
    b = transpose(b)

    return L, KD, w, b



def FindValueFunction(params, r, w, b, BQ0, savingvp):

    # savingvp denotes next period value function
    # if savingvp is zero matrix, we find value function in steady state
    # if savingp has values, we find value function in transition

    savingv = zeros((params.Na, params.Nz, params.I+1))         # value function
    savingg = zeros((params.Na, params.Nz, params.I))           # policy function
    savingg_ordered = zeros((params.Na, params.Nz, params.I))   # ordered policy function

    # Since people surely die at I+1, value function becomes zero at I+1
    savingv[:,:,params.I] = zeros((params.Na, params.Nz))
    savingg[:,:,params.I-1] = zeros((params.Na, params.Nz))

    # for period I
    c0 = tile(transpose(asmatrix((1+(1-params.TAU)*r)*params.agrid)), (1, params.Nz))
    c1 = tile(asmatrix(transpose((1-params.THETA-params.TAU)*w*params.e[:,params.I-1])), (params.Na,1))
    c2 = b[params.I-1]*ones((params.Na,params.Nz))
    c3 = BQ0/sum(params.mu)*ones((params.Na,params.Nz))
    c  = asarray(c0 + c1 + c2 + c3)
    c[c <= 0.0] = exp(-10**10)
    uc = dot(power(c,(1-params.SIGMA)),1/(1-params.SIGMA))
    savingv[:,:,params.I-1] = uc

    # for period i < I
    for i in range(params.I-2,-1,-1):
        for j in range(0, params.Nz):
            c0 = (1+(1-params.TAU)*r)*params.agrid
            c1 = squeeze(((1-params.THETA-params.TAU)*w*params.e[j,i] + b[i] + BQ0/sum(params.mu))*ones((params.Na,1)))
            c2 = tile(transpose(asmatrix(c0+c1)), (1,params.Na))
            c3 = tile(params.agrid, (params.Na,1))
            c = asarray(c2 - c3)
            c[c <= 0.0] = exp(-10**10)
            uc = dot(power(c,(1-params.SIGMA)),1/(1-params.SIGMA))
            if sum(savingvp) == 0:
               star = uc + params.BETA*params.s[i+1]*tile(dot(asmatrix(transpose(params.pi[j,:])),asmatrix(transpose(savingv[:,:,i+1]))), (params.Na,1))
            else:
               star = uc + params.BETA*params.s[i+1]*tile(dot(asmatrix(transpose(params.pi[j,:])),asmatrix(transpose(savingvp[:,:,i+1]))), (params.Na,1))
            aa = amax(star,1)
            bb = star.argmax(1)
            for k in range(params.Na):
                savingv[k, j, i] = aa[k]
                savingg[k, j, i] = params.agrid[bb[k]]
                savingg_ordered[k, j, i] = bb[k]

    return savingv, savingg, savingg_ordered



def FindDistribution(params, savingg_ordered, savingpopl):

    # savingpopl denote savingpop at previous period
    # if savingpopl is zero matrix, we find distribution in steady state
    # if savingpopl has values, we find distribution in transition

    savingpop = zeros((params.Na, params.Nz, params.I))
    savingpop[0,:,0] = params.p[:,0]

    for i in range(1,params.I):
        for j in range(params.Na):
            for k in range(params.Nz):
                pop = zeros((params.Na, params.Nz))
                if sum(savingpopl == 0):
                    pop[savingg_ordered[j,k,i-1],:] = savingpop[j,k,i-1]*params.pi[k,:]
                else:
                    pop[savingg_ordered[j,k,i-1],:] = savingpopl[j,k,i-1]*params.pi[k,:]
                savingpop[:,:,i] = savingpop[:,:,i] + pop

    return savingpop


def FindCapitalSupply(params, savingpop):

    # Aggregate Capital Supply (KS) and Bequest (BQ)
    KS = 0
    BQ = 0
    for i in range(params.I):
        KS = KS + sum(dot(params.agrid,sum(savingpop[:,:,i],1)))*params.mu[i]
        BQ = BQ + (1/params.s[i])*(1-params.s[i])*sum(dot(params.agrid,sum(savingpop[:,:,i],1)))*params.mu[i]

    BQTAX = BQ*params.ZETA
    BQ = (1-params.ZETA)*BQ

    return KS, BQ, BQTAX


def FindEconomicStatusGivenR(params, r, savingvp, savingpopl):

    # savingvp denotes next period value function
    # savingpopl denotes savingpop at previous period
    # Set 0 for Both in steady state

    status = EconomicStatus()
    status.r = r
    status.L, status.KD, status.w, status.b = FindWageETCs(params, r) # Given r, find Labor Supply, Capital Demand, Wage, and Pension for retirees

    # initial value for bequest
    BQ0 = 0
    diff_BQ = 100

    while abs(diff_BQ) > 0.01:

        status.savingv, status.savingg, status.savingg_ordered = FindValueFunction(params, status.r, status.w, status.b, BQ0, savingvp) # Find Value Function
        status.savingpop = FindDistribution(params, status.savingg_ordered, savingpopl) # Find Population Distribution
        status.KS, status.BQ, status.BQTAX = FindCapitalSupply(params, status.savingpop) # Find Aggregate Capital Supply (KS) and Bequest (BQ)

        diff_BQ = status.BQ - BQ0
        BQ0 = status.BQ

    return status



def FindSteadyState(params):

    diff_r = zeros((params.Nr, 1))

    statusMAT = EconomicStatusMAT()

    statusMAT.KSmat = zeros((params.Nr, 1))
    statusMAT.KDmat = zeros((params.Nr, 1))
    statusMAT.BQmat = zeros((params.Nr, 1))
    statusMAT.rmat  = zeros((params.Nr, 1))
    statusMAT.BQTAXmat = zeros((params.Nr, 1))

    for i_r in range(params.Nr):

        status = FindEconomicStatusGivenR(params, params.rgrid[i_r], 0, 0)

        # Difference between KS and KD
        diff_r[i_r] = status.KS - status.KD

        print 'Capital Market:', 'r=', params.rgrid[i_r], 'KD=', status.KD, 'KS=', status.KS

        # Save results
        statusMAT.KSmat[i_r] = status.KS
        statusMAT.KDmat[i_r] = status.KD
        statusMAT.BQmat[i_r] = status.BQ
        statusMAT.BQTAXmat[i_r] = status.BQTAX
        statusMAT.rmat[i_r]  = status.r

    # Find Equilibrium r
    for i in range(params.Nr-1):
        if diff_r[i]*diff_r[i+1] < 0:
            r0 = params.rgrid[i]
            r1 = params.rgrid[i+1]
            diffnew = diff_r[i+1]
            break;

    it = 0

    while abs(diffnew) > 0.1 and abs(r0-r1) > 0.0001:

        it = it + 1

        r = (r0+r1)/2

        print 'Searching Equilibrium r:', 'it=', it, 'r=', r, 'r0=', r0, 'r1=', r1

        status = FindEconomicStatusGivenR(params, r, 0, 0)

        diffnew = status.KS - status.KD

        if diffnew >= 0:
            r1 = r
        else:
            r0 = r

        statusMAT.KSmat    = append(statusMAT.KSmat, status.KS)
        statusMAT.KDmat    = append(statusMAT.KDmat, status.KD)
        statusMAT.BQmat    = append(statusMAT.BQmat, status.BQ)
        statusMAT.BQTAXmat = append(statusMAT.BQTAXmat, status.BQTAX)
        statusMAT.rmat     = append(statusMAT.rmat, status.r)

    return status, statusMAT


def FindPopulationTransition(params0, params1, NP):

    Nmat = ones((params0.I, NP))*params0.N

    for t in range(NP):
        for i in range(params0.I):
            if i < t:
                Nmat[i,t] = params1.N

    smat = transpose(reshape(tile(params0.s, (1,NP)), (NP, shape(params0.s)[0])))
    mumat = zeros((params0.I, NP))
    popsize = zeros((params0.I, NP))

    for t in range(NP):
        if t == 0:
            popsize[0,t] = 1
        else:
            popsize[0,t] = popsize[0, t-1]*(1 + Nmat[0,t])

        for i in range(1,params0.I):
            popsize[i,t] = popsize[i-1,t]/(1 + Nmat[i,t])*smat[i,t]

    for t in range(NP):
        mumat[:,t] = popsize[:,t]/popsize[0,t]

    return mumat, smat, popsize, Nmat


def Transition(params, steady0, steady1, NP):

    rgrid0 = linspace(steady0.r, steady1.r, NP)
    rgrid1 = zeros(shape(rgrid0))

    BQmat0 = linspace(steady0.BQ, steady1.BQ, NP)
    BQmat1 = zeros(shape(BQmat0))

    Lmat  = zeros(shape(rgrid0))
    KDmat = zeros(shape(rgrid0))
    KSmat = zeros(shape(rgrid0))
    wmat  = zeros(shape(rgrid0))
    bmat  = zeros((params.I, shape(rgrid0)[0]))
    BQTAXmat = zeros(shape(rgrid0))

    savingvmat = zeros((params.Na, params.Nz, params.I+1, NP))
    savinggmat = zeros((params.Na, params.Nz, params.I, NP))
    savingg_orderedmat = zeros((params.Na, params.Nz, params.I, NP))
    savingpopmat = zeros((params.Na, params.Nz, params.I, NP))

    it = 0

    while (max(abs(rgrid0-rgrid1))) > 0.0001:

        it = it + 1

        if it > 1:
            rgrid0 = (rgrid0 + rgrid1)/2

        print 'it =', it, 'diff of rgrid =', max(abs(rgrid0-rgrid1))

        for t in range(NP):

            paramst = Params()
            paramst.N = Nmat[0,t]
            paramst.mu = mumat[:,t]
            paramst.s  = smat[:,t]

            LL, KDKD, ww, bb = FindWageETCs(paramst, rgrid0[t])
            Lmat[t] = LL
            KDmat[t] = KDKD
            wmat[t] = ww
            bmat[:,t] = asmatrix(bb)



        KSmat[NP-1] = steady1.KS
        BQmat0[NP-1] = steady1.BQ
        BQTAXmat[NP-1] = steady1.BQTAX

        savingvmat[:, :, :, NP-1] = steady1.savingv
        savinggmat[:, :, :, NP-1] = steady1.savingg
        savingg_orderedmat[:, :, :, NP-1] = steady1.savingg_ordered


        for t in range(NP-1):

            print 'it = ', it, 'Back: ', 't = ', t

            BQ0 = 0
            diff_BQ = 100

            paramst = Params()
            paramst.N = Nmat[0, NP-2-t]
            paramst.s = smat[:, NP-2-t]
            paramst.mu = mumat[:, NP-2-t]

            savingvmat[:,:,:,NP-2-t], savinggmat[:,:,:,NP-2-t], savingg_orderedmat[:,:,:,NP-2-t] = FindValueFunction(paramst, rgrid0[NP-2-t], wmat[NP-2-t], bmat[:,NP-2-t], BQmat0[NP-2-t], savingvmat[:,:,:,NP-2-t+1])
            savingpopmat[:,:,:,NP-2-t] = FindDistribution(paramst, savingg_orderedmat[:,:,:,NP-2-t], 0)

        BQmat1[0] = steady0.BQ
        BQTAXmat[0] = steady0.BQTAX
        KSmat[0] = steady0.KS
        rgrid1[0] = steady0.r
        savingpopmat[:,:,:,0] = steady0.savingpop

        for t in range(1,NP):

            print 'it = ', it, 'Go: ', 't = ', t

            paramst = Params()
            paramst.N = Nmat[0,t]
            paramst.s = smat[:,t]
            paramst.mu = mumat[:,t]

            savingpopmat[:,:,:,t] = FindDistribution(paramst, savingg_orderedmat[:,:,:,t], savingpopmat[:,:,:,t-1])

            KSmat[t], BQmat1[t], BQTAXmat[t] = FindCapitalSupply(paramst, savingpopmat[:,:,:,t])

            rgrid1[t] = paramst.ALPHA*paramst.A*KSmat[t]^(paramst.ALPHA-1)*Lmat[t]^(1-paramst.ALPHA) - paramst.DELTA

        BQmat0 = (BQmat0+BQmat1)/2

        plot(range(NP), rgrid0, 'r--', range(NP), rgrid1, 'bs')
        show()

    return Lmat, KDmat, wmat, bmat



# if __name__ == '__main__':
    """
    start_time = datetime.now()
    params0 = Params(N=0.012)
    steady0, steadyMAT0 = FindSteadyState(params0)
    pickle.dump(params0, open('params0.p', 'w'))
    pickle.dump(steady0, open('steady0.p', 'w'))
    end_time = datetime.now()
    print 'Total Duration: {}'.format(end_time - start_time)

    start_time = datetime.now()
    params1 = Params(N=-0.012)
    steady1, steadyMAT1 = FindSteadyState(params1)
    pickle.dump(params1, open('params1.p', 'w'))
    pickle.dump(steady1, open('steady1.p', 'w'))
    end_time = datetime.now()
    print 'Total Duration: {}'.format(end_time - start_time)
    """

    NP = 150
    params0 = pickle.load(open('params0.p', 'r'))
    params1 = pickle.load(open('params1.p', 'r'))
    steady0 = pickle.load(open('steady0.p', 'r'))
    steady1 = pickle.load(open('steady1.p', 'r'))

    mumat, smat, popsize, Nmat = FindPopulationTransition(params0, params1, NP)

    Lmat, KDmat, wmat, bmat = Transition(params0, steady0, steady1, NP)
