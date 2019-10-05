#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Arefe Abghari)s
"""

from matplotlib.pylab import figure, plot, ylabel, show, subplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decimal import Decimal
from scipy.stats import gaussian_kde
#import data for delta which is integrated numerically
delta = np.loadtxt("/Users/apple/Desktop/Project/Arefe/delta", delimiter="\\")
delta = delta / delta[0] # D(z) should be normalized to 1 in z=0
# top-hat filter correlation formula
#C = np.loadtxt("/Users/arefe/Desktop/Arefe/Cosmo_Research/code/CorrelationMatrix_-5")
#L = np.loadtxt("/Users/arefe/Desktop/Arefe/Cosmo_Research/code/CholeskiDecompose_-5")
def corr(s, sp):
    C = (s / 4.0) * (5.0 - np.power(s, 2.0)/np.power(sp, 2.0))
    return C
# conditional correlation matrix. p(k)~k^-2

def corr_Markov(s, sp):
    C = s
    return C


def correlationMatrix_nonconditional_Markov (S , ds):

    Ns = int(S / ds)
    #k = int(S0 / ds)
    size = (Ns, Ns)
    C = np.zeros(size)

    i = 1


    while i < Ns+1 :
        j = 1
        while j < i+1 :
            C[i-1, j-1] = corr_Markov(j*ds, i*ds)

            j+=1
        while j <Ns+1:
            C[i-1, j-1] = corr_Markov(i*ds, j*ds)
            j+=1

        i+=1

    return C


def correlationMatrix_nonconditional (S , ds):

    Ns = int(S / ds)
    #k = int(S0 / ds)
    size = (Ns, Ns)
    C = np.zeros(size)

    i = 1


    while i < Ns+1 :
        j = 1
        while j < i+1 :
            C[i-1, j-1] = corr(j*ds, i*ds)

            j+=1
        while j <Ns+1:
            C[i-1, j-1] = corr(i*ds, j*ds)
            j+=1

        i+=1

    return C


def correlationMatrix(S, S0, ds):
    Ns = int(S / ds)
    k = int(S0 / ds)
    size = (Ns, Ns)
    C = np.zeros(size)

    i = 1
    while i < k + 1:
        j = 1
        while j < i + 1:
            C[i-1, j-1] = corr(j*ds, i*ds) - 1/S0 * corr(j*ds, k*ds) * corr(i*ds, k*ds)
            j += 1

        while j < k + 1:
            C[i-1, j-1] = corr(i*ds, j*ds) - 1/S0 * corr(j*ds, k*ds) * corr(i*ds, k*ds)
            j += 1

        i += 1

    ip, jp = i, j

    while i < Ns + 1:
        j1 = 1
        while j1 < jp:
            C[i-1, j1-1] = corr(j1*ds, i*ds) - 1/S0 * corr(j1*ds, k*ds) * corr(k*ds, i*ds)
            j1 += 1
        i += 1


    while j < Ns + 1:
        i1 = 1
        while i1 < ip:
            C[i1-1, j-1] = corr(i1*ds, j*ds) - 1/S0 * corr(i1*ds, k*ds) * corr(k*ds, j*ds)
            i1 += 1
        j += 1


    i = ip
    while i < Ns + 1:
        j = jp
        while j < i + 1:
            C[i-1, j-1] = corr(j*ds, i*ds) - 1/S0 * corr(k*ds, j*ds) * corr(k*ds, i*ds)
            j += 1

        while j < Ns + 1:
            C[i-1, j-1] = corr(i*ds, j*ds) - 1/S0 * corr(k*ds, i*ds) * corr(k*ds, j*ds)
            j += 1

        i += 1

    C = np.delete(C, k-1, 0)
    C = np.delete(C, k-1, 1)

    return C


def paths_nonconditional (S , ds , L , Npaths):

    Ns = int(S / ds)
    #k = int(S0 / ds)
    Npaths = int(Npaths)

    X = np.dot(L, np.random.randn(Ns, Npaths))

    return X


def paths(S, S0, ds, L , Npaths): #returns a Ns*Npaths matrix containing deltas.

    Ns = int(S / ds) - 1
    k = int(S0 / ds)
    Npaths = int(Npaths)
    Mu = np.ones((Npaths, Ns))
    Mu[:, :k] = delta0[0]/S0 * corr(np.arange(1, k+1)*ds, k*ds)
    Mu[:, k:] = delta0[0]/S0 * corr(k*ds, np.arange(k+1, Ns+1)*ds)
    X = np.dot(L, np.random.randn(Ns, Npaths)) + Mu.T

    return X


def firstUpCrossing(df, level, ds):
    Npaths = np.size(df, 1)
    xC = level
    f = np.array([])
    i = 0
    while i < Npaths:
        xTrj = df[:, i]
        vTrj = np.diff(xTrj) / ds
        xTrj = np.delete(xTrj, -1)
        ind = np.where((xC - vTrj * ds < xTrj) & (xTrj < xC))[0]
        if len(ind) != 0:
            f = np.append(f, ind[0] * ds)
        else:
            f = np.append(f, 0)
        i += 1
    return f

def BeforeS0Del(S, S0, ds, Npaths, p ,flevel ):
    # exclude trajectories that dont FUC at z=8
    f0 = firstUpCrossing(p, flevel, ds)
    drp = np.where(f0 == 0)[0]
    p = np.delete(p, drp, 1)
    Ns = int(S / ds) - 1
    k = int(S0 / ds)
    Npaths = int(Npaths)
    # exclude trajectories that FUC before S0
    ind0, ind1 = np.where(p[:k, :] > delta0[0])  #in kharabe
    """
    H = list(zip(ind0, ind1))
    H = np.array(H)
    for i, j in H:
        p[i,j] = 0
    j = np.array([])
    j = p.T[np.all(p.T != 0, axis = 1)]
    """
    p = np.delete(p, ind1, 1)
    return p


def FUPtoMass(Nz , NCrossed , FUP):
    size     = (Nz, NCrossed)

    yarray = []
    #plotting merger trees
    for i in range(NCrossed):
        y = FUP[:, i]
        len0 = len(y)
        #print (np.where (y==0)[0])
        y = y[y != 0]
        #print (len(y))
        y = y ** (-1/3)
        y = y[m:]
        print (len(y))
        yarray.append  (y)

    return yarray


def FiftySeventy(yarray):

    massfifty =np.copy( [np.argmin(abs(yarray[i][:]-0.5))*0.01 for i in range (0, len(yarray))])
    massseventy = np.copy([np.argmin(abs(yarray[i][:]-0.7))*0.01 for i in range (0, len(yarray))])
    #massfifty *=0.01
    #massseventy *=0.01


    return massfifty , massseventy


def YmaxYmin(FUP) :
    FUP_p = FUP[m,:]
    yarray1 = FUP_p[FUP_p!=0]
    yarray1 = yarray1**(-1/3)
    ymax = max(yarray1)
    ymin = (S**(-1/3))/ymax

    return ymax , ymin


level0 = 1.686
S      = 50
ds     = 0.01
S0     = 1 #condition
delta0 = np.zeros(1) + level0   #condition

R0 = (9 * np.pi / 20) * 10 * 10**6 * 3.09 * 10**16 /S0
M0 =   (4 * np.pi / 3) * 9.9 * 0.227 * 10**(-27) * R0**3 /(1.99 * 10**(30))


Npaths = 50000
Nz     = len(delta)
zarray = np.linspace(0, 8, num = Nz)

flevel = level0 / delta[Nz -1]
bins = 100
fig, (ax1) = plt.subplots()


#markov
C = correlationMatrix_nonconditional_Markov(S, ds)
L = np.linalg.cholesky(C)
p1  = paths_nonconditional(S, ds,L ,  Npaths)
FUPCrossing = firstUpCrossing(p1 ,  level0 ,  ds)
ind0 = np.where(FUPCrossing==0 )[0]
FUPCrossing = np.delete(FUPCrossing , ind0 , 0)

density = gaussian_kde(FUPCrossing)
xs = np.linspace(0, np.max(FUPCrossing),bins)
density.covariance_factor = lambda : .05
density._compute_covariance()
densities = density(xs)
ax1.plot(xs,densities, color = 'maroon', label = 'Markovian Trajectories')
ax1.fill_between(xs, 0, densities, facecolor='pink', alpha=0.7)


#non-markov
C2 = correlationMatrix_nonconditional(S, ds)
L2 = np.linalg.cholesky(C2)
p2  = paths_nonconditional(S, ds,L2 ,  Npaths)
FUPCrossing2 = firstUpCrossing(p2 ,  level0 ,  ds)
ind1 = np.where(FUPCrossing2==0 )[0]
FUPCrossing2 = np.delete(FUPCrossing2 , ind1 , 0)

density2 = gaussian_kde(FUPCrossing2)
xs2 = np.linspace(0, np.max(FUPCrossing2),bins)
density2.covariance_factor = lambda : .05
density2._compute_covariance()
densities2 = density2(xs2)
ax1.plot(xs2,densities2, color = 'navy', label = 'Non-Markovian Trajectories')
ax1.fill_between(xs2, 0, densities2, facecolor='lightsteelblue', alpha=0.5)

#press-schechter
#xs4 = S in PS formalism
f_PS = xs ** (-3/2) * np.exp(- level0 ** 2 / (2 * xs)) * level0 * (2 * np.pi)**(-1/2)
ax1.plot(xs,f_PS, color = 'black', label = 'PS Mass Function')

'''
#conditional
C3 = correlationMatrix(S, S0, ds)
L3= np.linalg.cholesky(C3)
p3  = paths(S, S0, ds, L3 , Npaths)
FUPCrossing3 = firstUpCrossing(p3 ,  level0 ,  ds)
ind1 = np.where(FUPCrossing3==0 )[0]
FUPCrossing3 = np.delete(FUPCrossing3 , ind1 , 0)

density3 = gaussian_kde(FUPCrossing3)
xs3 = np.linspace(0, np.max(FUPCrossing3),bins)
density3.covariance_factor = lambda : .1
density3._compute_covariance()

densities3 = density3(xs3) / max(density3(xs3))

ax1.plot(xs3,densities3, color = 'navy', label = 'Conditional')
ax1.fill_between(xs3, 0, densities3, facecolor='lightsteelblue', alpha=0.5)
'''
np.save('xs', xs)
np.save('densities', densities)
np.save('densities2', densities2)
np.save('f_PS', f_PS)
ax1.legend()
ax1.set_xlabel('S')
ax1.set_ylabel('Density')
#fig.suptitle('Histogram of First up crossing')
plt.show()
