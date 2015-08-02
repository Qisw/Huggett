from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize_scalar, broyden1, broyden2
from numpy import linspace, mean, array, zeros, absolute, loadtxt, dot, prod, int, \
                    genfromtxt, sum, argmax, tile, concatenate, ones, log, \
                    unravel_index, cumsum
from matplotlib import pyplot as plt
from datetime import datetime
import time
import pickle
import os
from platform import system
from multiprocessing import Process, Lock, Manager
from multiprocessing.sharedctypes import Value, Array, RawArray
from ctypes import Structure, c_double

def plot(k, t=0, yi=0, yt=78, ny=10):
    """plot life-path of aggregate capital accumulation and house demand"""
    mls = k.mls
    pop, aa, hh, aN, hN = k.pop, k.aa, k.hh, k.aN, k.hN
    mu = k.mu[t]
    a = zeros(mls)
    h = zeros(mls)
    ap = zeros(mls)
    hp = zeros(mls)
    al = zeros(aN)
    hl = zeros(hN)
    """Aggregate all cohorts' capital and labor supply at each year"""
    for y in range(mls):
        ap[y] = sum(mu[y],(0,1)).dot(aa)*pop[t,y]
        hp[y] = sum(mu[y],(1,2)).dot(hh)*pop[t,y]
        a[y] = sum(mu[y],(0,1)).dot(aa)
        h[y] = sum(mu[y],(1,2)).dot(hh)
        al += sum(mu[y],(0,1))*pop[t,y]
        hl += sum(mu[y],(1,2))*pop[t,y]
    title = 'psi=%2.2f'%(k.psi) + \
            ' r=%2.2f%%'%(k.r[t]*100) + ' q=%2.2f'%(k.q[t]) + \
            ' K=%2.1f'%(k.K[t]) + ' Hd=%2.1f'%(k.Hd[t])
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
    fig.subplots_adjust(hspace=.5, wspace=.3, left=None, right=None, top=None, bottom=None)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    ax1.plot(range(mls),ap,label='aggregate')
    ax1.plot(range(mls),a,label='per capita')
    ax2.plot(range(mls),hp,label='aggregate')
    ax2.plot(range(mls),h,label='per capita')
    for y in linspace(yi,yt,ny).astype(int):
        ax3.plot(aa,sum(mu[y],(0,1)),label='age %i'%(y))
    for y in linspace(yi,yt,ny).astype(int):
        ax4.plot(hh,sum(mu[y],(1,2)),label='age %i'%(y))
    ax5.plot(cumsum(al)/sum(al),cumsum(aa*al)/sum(aa*al),".")
    ax6.plot(cumsum(hl)/sum(hl),cumsum(hh*hl)/sum(hh*hl),".")
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
    ax6.set_ylabel('Cum. Share of House Occupied')
    ax.set_title(title, y=1.08)
    ax1.set_title('Life-Cycle Liquid Asset Accumulation')
    ax2.set_title('Life-Cycle House Size')
    ax3.set_title('Dist. of Liquid Asset w/i Cohort')
    ax4.set_title('Dist. of House Size w/i Cohort')
    ax5.set_title('Lorenz Curve for Liquid Asset')
    ax6.set_title('Lorenz Curve for House')
    if system() == 'Windows':
        path = 'D:\Huggett\Figs'
    else:
        path = '/Users/hyunchangyi/GitHub/Huggett/Figs'
    fullpath = os.path.join(path, filename)
    fig.savefig(fullpath, dpi=200)
    # ax4.axis([0, 80, 0, 1.1])
    plt.show()
