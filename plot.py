from scipy.interpolate import interp1d
from scipy.optimize import fsolve, minimize_scalar, broyden1, broyden2
from numpy import linspace, mean, array, zeros, absolute, loadtxt, dot, prod, int, \
                    genfromtxt, sum, argmax, tile, concatenate, ones, log, unravel_index
from matplotlib import pyplot as plt
from datetime import datetime
import time
import pickle
import os
from platform import system
from multiprocessing import Process, Lock, Manager
from multiprocessing.sharedctypes import Value, Array, RawArray
from ctypes import Structure, c_double

def plot(k, t=0, ny=10):
    """plot life-path of aggregate capital accumulation and house demand"""
    mls = k.mls
    aa, pop, hh = k.aa, k.pop, k.hh
    mu = k.mu[t]
    a = zeros(mls)
    h = zeros(mls)
    ap = zeros(mls)
    hp = zeros(mls)
    """Aggregate all cohorts' capital and labor supply at each year"""
    for y in range(mls):
        ap[y] = sum(mu[y],(0,1)).dot(aa)*pop[t,y]
        hp[y] = sum(mu[y],(1,2)).dot(hh)*pop[t,y]
        a[y] = sum(mu[y],(0,1)).dot(aa)
        h[y] = sum(mu[y],(1,2)).dot(hh)
    title = 'psi=%2.2f'%(k.psi) + \
            ' r=%2.2f%%'%(k.r[t]*100) + ' q=%2.2f'%(k.q[t]) + \
            ' K=%2.1f'%(k.K[t]) + ' Hd=%2.1f'%(k.Hd[t])
    filename = title + '.png'
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
    ax1.plot(range(mls),ap,label='w/ pop. adj.')
    ax1.plot(range(mls),a,label='w/o pop. adj.')
    ax2.plot(range(mls),hp,label='w/ pop. adj.')
    ax2.plot(range(mls),h,label='w/o pop. adj.')
    for y in linspace(0,mls-1,ny).astype(int):
        ax3.plot(aa,sum(mu[y],(0,1)),label='age %i'%(y))
    for y in linspace(0,mls-1,ny).astype(int):
        ax4.plot(hh,sum(mu[y],(1,2)),label='age %i'%(y))
    # ax1.legend(bbox_to_anchor=(0.9,1.0),loc='center',prop={'size':8})
    ax1.legend(prop={'size':7})
    ax2.legend(prop={'size':7})
    ax3.legend(prop={'size':7})
    ax4.legend(prop={'size':7})
    ax3.axis([0, 15, 0, 0.1])
    # ax4.axis([0, 80, 0, 1.0])
    ax1.set_xlabel('Age')
    ax2.set_xlabel('Age')
    ax3.set_xlabel('Asset Size')
    ax4.set_xlabel('House Size')
    ax.set_title(title, y=1.08)
    ax1.set_title('Liquid Assets over Ages')
    ax2.set_title('House Sizes over Ages')
    ax3.set_title('Liquid Assets Dist. w/i Cohorts')
    ax4.set_title('House Sizes Dist. w/i Cohorts')
    if system() == 'Windows':
        path = 'D:\Huggett\Figs'
    else:
        path = '/Users/hyunchangyi/GitHub/Huggett/Figs'
    fullpath = os.path.join(path, filename)
    fig.savefig(fullpath)
    # ax4.axis([0, 80, 0, 1.1])
    plt.show()
