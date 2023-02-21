#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:13:00 2023

@author: davidhealy
"""

#   all the libraries we need...
import time
import pfs 
import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import linalg
from random import  uniform
from tqdm import tqdm

print('\nStarting pfsRuhrBochum_Sf - FRACTURE SUSCEPTIBILITY - version 1...')
print('*** Sf analysis for Ruhr-Bochum - deep carbonate ***\n')
tic = time.time() 

#   initialise 
nMC = 5000          #   number of Monte Carlo runs 
nBins = 20          #   number of bins in histograms  

###########################################################################
#   define some stresses: magnitudes and orientations 

#   set ranges of all variables other than strike 
#   SV = sigmaV - vertical stress, in MPa 
muSV = 110.6
sdSV = 24.4 
#   SH = sigmaH - max horizontal stress, in MPa 
muSH = 105.5
sdSH = 54.
#   Sh = sigmah - min horizontal stress, in MPa 
muSh = 62.
sdSh = 19.1 

#   build distributions of each stress, such that s1 >= s2 >= s3 for all nMC
# stress = pfs.setStressDist2(nMC, muSV, sdSV, muSH, sdSH, muSh, sdSh)
# SV = stress[:,0]
# SH = stress[:,1]
# Sh = stress[:,2]
a = +3. 
SV = stats.norm.rvs(muSV, sdSV, nMC) 
SH = stats.skewnorm.rvs(a, muSH, sdSH, nMC)
Sh = stats.skewnorm.rvs(a, muSh, sdSh, nMC)

minSV = SV.min()
maxSV = SV.max()
meanSV = SV.mean()
stdSV = SV.std()
rangeSV = maxSV - minSV 

minSH = SH.min()
maxSH = SH.max()
meanSH = SH.mean()
stdSH = SH.std()
rangeSH = maxSH - minSH 

minSh = Sh.min()
maxSh = Sh.max()
meanSh = Sh.mean()
stdSh = Sh.std()
rangeSh = maxSh - minSh 

#   pore fluid pressure; can assume hydrostatic in many cases (= 0.4 * sV)
muPf = 40.3       
sdPf = muPf * .1 
Pf = stats.norm.rvs(muPf, sdPf, nMC)
minPf = Pf.min()
maxPf = Pf.max()
meanPf = Pf.mean()
stdPf = Pf.std()
rangePf = maxPf - minPf 

#   SHmax azimuth
SHazMean = 161.
SHazKappa = 30. 
#   von Mises symmetrical about mean, in range -pi to +pi
SHaz = pfs.rad2deg(np.random.vonmises(pfs.deg2rad(SHazMean), SHazKappa, nMC))
minSHaz, maxSHaz, meanSHaz, medianSHaz, stddevSHaz = pfs.getCircStats2(SHaz) 
rangeSHaz = maxSHaz - minSHaz 
SHazTrue = np.copy(SHaz)
#   keep the range to 0-180 
SHaz[np.ix_(SHaz<0)] += 180.
SHaz[np.ix_(SHaz>180)] -= 180.

#   read in file of fault strikes
fnSeg = 'FaultStrikes_RuhrBochum.txt'
data = np.loadtxt(fnSeg, delimiter="\t")
nFaults = len(data)
FaultID = np.copy(data[:,0])
Strike = np.copy(data[:,1])
#   get the stats of this distribution 
minStrike, maxStrike, meanStrike, medianStrike, stddevStrike = pfs.getCircStats2(Strike) 

#   fault dips  
minDip = 65. 
maxDip = 85. 
Dip = np.zeros([nFaults,1])
for iF in range(0, nFaults):
    Dip[iF] = uniform(minDip, maxDip)
minDip, maxDip, meanDip, medianDip, stddevDip = pfs.getCircStats2(Dip) 
rangeDip = maxDip - minDip 

###########################################################################
#   rock properties - friction and cohesion 
#   - using skew normal distributions for generality and flexibility 
#   - easy to adjust to normal distribution (alpha = 0) if required 
#   - log normal distributions also possible 
#   - alpha > 0 => skew high, alpha < 0 => skew low 

#   fault friction 
muMu = 0.6 
sdMu = muMu * .1 
#aMu = -3.             
#Mu = stats.skewnorm.rvs(aMu, muMu, sdMu, nMC)
Mu = stats.norm.rvs(muMu, sdMu, nMC)
Mu[np.ix_(Mu < 0.)] = 0.
minMu = Mu.min()
maxMu = Mu.max()
rangeMu = maxMu - minMu 
meanMu = Mu.mean()
stdMu = Mu.std()

#   fault cohesion 
muC0 = 1. 
sdC0 = muC0 * .2
#aC0 = +3. 
C0 = stats.norm.rvs(muC0, sdC0, nMC)
C0[np.ix_(C0 < 0.)] = 0.
minC0 = C0.min()
maxC0 = C0.max()
rangeC0 = maxC0 - minC0 
meanC0 = C0.mean()
stdC0 = C0.std() 

###########################################################################
#   plot summary histograms of input variables
 
fig, ax = plt.subplots(figsize=(12,12))

ax1 = plt.subplot(331)
nSV, bSV, pSV = ax1.hist(SV, nBins)

ax2 = plt.subplot(332)
nSH, bSH, pSH = ax2.hist(SH, nBins)

ax3 = plt.subplot(333)
nSh, bSh, pSh = ax3.hist(Sh, nBins)

nMax123 = np.max([nSh.max(), nSH.max(), nSV.max()])*1.05 

ax1.plot([muSV, muSV], [0, np.max(nSV)], '-r')
ax1.text(SV.min()*1.01, nMax123*.75, 
              ('Mean: %3.1f\nStdDev: %2.1f' % (meanSV, stdSV)), 
                           bbox=dict(facecolor='white', edgecolor='black'))
ax1.set_ylim(0., nMax123)
ax1.set_xlabel(r'$\sigma_V$, MPa')
ax1.set_ylabel('Count')
ax1.grid(True)
ax1.set_title('Variation in vertical stress, n=%i' % nMC)

ax2.plot([muSH, muSH], [0, np.max(nSH)], '-r')
ax2.text(SH.min()*1.01, nMax123*.75, 
              ('Mean: %3.1f\nStdDev: %2.1f' % (meanSH, stdSH)),
                           bbox=dict(facecolor='white', edgecolor='black'))
ax2.set_ylim(0., nMax123)
ax2.set_xlabel(r'$\sigma_{Hmax}$, MPa')
ax2.set_ylabel('Count')
ax2.grid(True)
ax2.set_title('Variation in max. horizontal stress, n=%i' % nMC)

ax3.plot([muSh, muSh], [0, np.max(nSh)], '-r')
ax3.text(Sh.min()*1.01, nMax123*.75, 
              ('Mean: %3.1f\nStdDev: %2.1f' % (meanSh, stdSh)),
                           bbox=dict(facecolor='white', edgecolor='black'))
ax3.set_ylim(0., nMax123)
ax3.set_xlabel(r'$\sigma_{hmin}$, MPa')
ax3.set_ylabel('Count')
ax3.grid(True)
ax3.set_title('Variation in min. horizontal stress, n=%i' % nMC)

ax4 = plt.subplot(334)
nSHaz, bSHaz, pSHaz = ax4.hist(SHaz, nBins)
ax5 = plt.subplot(335)
nStrike, bStrike, pStrike = ax5.hist(Strike, nBins)
ax6 = plt.subplot(336)
nDip, bDip, pDip = ax6.hist(Dip, nBins)

ax4.plot([meanSHaz, meanSHaz], [0, np.max(nSHaz)], '-r')
ax4.text(10., nSHaz.max()*.75, 
              ('Mean: %3.1f\nKappa: %3.1f' % (meanSHaz, SHazKappa)),
                           bbox=dict(facecolor='white', edgecolor='black'))
ax4.set_xlabel(r'$\sigma_{Hmax}$ azimuth, degrees')
ax4.set_ylabel('Count')
ax4.set_xlim(0., 180.)
ax4.grid(True)
ax4.set_title(r'Variation in $\sigma_{Hmax}$ azimuth, n=%i' % nMC)

ax5.plot([meanStrike, meanStrike], [0, np.max(nStrike)], '-r')
ax5.text(10., nStrike.max()*.75, 
              ('Mean: %3.1f\nStdDev: %3.1f' % (meanStrike, stddevStrike)),
                           bbox=dict(facecolor='white', edgecolor='black'))
ax5.set_xlabel('Strike, degrees')
ax5.set_ylabel('Count')
ax5.set_xlim(0., 180.)
ax5.grid(True)
ax5.set_title('Variation in fault strike, n=%i' % nFaults)

ax6.plot([meanDip, meanDip], [0, np.max(nDip)], '-r')
ax6.text(minDip*1.01, nDip.max()*.75, 
              ('Mean: %3.1f\nStdDev: %3.1f' % (meanDip, stddevDip)),
                           bbox=dict(facecolor='white', edgecolor='black'))
ax6.set_xlabel('Dip, degrees')
ax6.set_ylabel('Count')
ax6.set_xlim(minDip, maxDip)
ax6.grid(True)
ax6.set_title('Variation in fault dip, n=%i' % nFaults)

ax7 = plt.subplot(337) 
nMu, bMu, pMu = ax7.hist(Mu, nBins)
ax8 = plt.subplot(338) 
nC0, bC0, pC0 = ax8.hist(C0, nBins)
ax9 = plt.subplot(339) 
nPf, bPf, pPf = ax9.hist(Pf, nBins)

#   get mode of mu and C0 as better measures of distrib 
modeMu = bMu[np.ix_(nMu == nMu.max())] + (bMu[1]-bMu[0])/2.
modeC0 = bC0[np.ix_(nC0 == nC0.max())] + (bC0[1]-bC0[0])/2.

ax7.plot([modeMu[0], modeMu[0]], [0, np.max(nMu)], '-r')
ax7.text(minMu*1.01, nMu.max()*.75, 
              ('Mode: %3.2f\nStdDev: %3.2f' % (modeMu[0], stdMu)),
                           bbox=dict(facecolor='white', edgecolor='black'))
ax7.set_xlabel('Friction coefficient')
ax7.set_ylabel('Count')
ax7.grid(True)
ax7.set_title('Variation in friction, n=%i' % nMC)

ax8.plot([modeC0[0], modeC0[0]], [0, np.max(nC0)], '-r')
ax8.text(minC0*1.01, nC0.max()*.75, 
              ('Mode: %3.1f\nStdDev: %3.2f' % (modeC0[0], stdC0)),
                           bbox=dict(facecolor='white', edgecolor='black'))
ax8.set_xlabel('Cohesion, MPa')
ax8.set_ylabel('Count')
ax8.grid(True)
ax8.set_title('Variation in cohesion, n=%i' % nMC)

ax9.plot([muPf, muPf], [0, np.max(nPf)], '-r')
ax9.text(minPf*1.01, nPf.max()*.75, 
              ('Mode: %3.1f\nStdDev: %3.1f' % (muPf, stdPf)),
                           bbox=dict(facecolor='white', edgecolor='black'))
ax9.set_xlabel('Pore fluid pressure, MPa')
ax9.set_ylabel('Count')
ax9.grid(True)
ax9.set_title('Variation in pore fluid pressure, n=%i' % nMC)

plt.tight_layout() 
plt.savefig('RuhrBochum Input Histograms - Sf.png', dpi=300)

#   Mohr plots showing variations in stresses and rock properties  
#frs.plotMohr3(SV, SH, Sh, Mu, modeMu[0], C0, modeC0[0])
#frs.plotMohr4(SV, SH, Sh, Pf, Mu, modeMu[0], C0, modeC0[0])

###########################################################################
#   Response Surface Method (RSM)
#   linear regression over multiple variables, with linear & quadratic fits 
#   assume 3^q design of calculation points - min, max and mean of each variable 
#   where q is the number of variables in each term:
#      fracture susceptibility Sf - ( variables (as above + Pf, mu and C0) 
#   all assuming Andersonian stresses with one principal stress vertical 
qSf = 9 

###############################################################################
#   linear fit for Sf 
nSf = pow(3,qSf)         #   3 coordinates for each variable 

#   solve y = BetaSf * XlSf for BetaSf as the coefficients in the regression
#   XlSf is the 'design matrix' of dummy variables; one per variable 
#   values of -1 = min, 0 = mean, +1 = max
XlSf = np.zeros([nSf,qSf+1])
iRow = 0 
for iA in range(-1,+2):
    for iB in range(-1,+2):
        for iC in range(-1,+2):
            for iD in range(-1,+2):
                for iE in range(-1,+2):
                    for iF in range(-1,+2):
                        for iG in range(-1,+2):
                            for iH in range(-1,+2):
                                for iJ in range(-1,+2):
                                    XlSf[iRow, :] = [1, iA, iB, iC, iD, iE, iF, iG, iH, iJ]
                                    iRow += 1

###############################################################################
#   quadratic fit for Sf     
q2Sf = 0.5 * qSf * (qSf + 1) + qSf  #   total terms incl. squares and products 
XqSf = np.zeros([int(nSf), int(q2Sf)+1])
for iQ in range(0,nSf):
    #   linear terms 
    XqSf[iQ,1] = XlSf[iQ,1]
    XqSf[iQ,2] = XlSf[iQ,2]
    XqSf[iQ,3] = XlSf[iQ,3]
    XqSf[iQ,4] = XlSf[iQ,4]
    XqSf[iQ,5] = XlSf[iQ,5]
    XqSf[iQ,6] = XlSf[iQ,6]
    XqSf[iQ,7] = XlSf[iQ,7]
    XqSf[iQ,8] = XlSf[iQ,8]
    XqSf[iQ,9] = XlSf[iQ,9]
    
    #   cross products 
    XqSf[iQ,10] = XlSf[iQ,1]*XlSf[iQ,2]
    XqSf[iQ,11] = XlSf[iQ,1]*XlSf[iQ,3]
    XqSf[iQ,12] = XlSf[iQ,1]*XlSf[iQ,4]
    XqSf[iQ,13] = XlSf[iQ,1]*XlSf[iQ,5]
    XqSf[iQ,14] = XlSf[iQ,1]*XlSf[iQ,6]
    XqSf[iQ,15] = XlSf[iQ,1]*XlSf[iQ,7]
    XqSf[iQ,16] = XlSf[iQ,1]*XlSf[iQ,8]
    XqSf[iQ,17] = XlSf[iQ,1]*XlSf[iQ,9]
    
    XqSf[iQ,18] = XlSf[iQ,2]*XlSf[iQ,3]
    XqSf[iQ,19] = XlSf[iQ,2]*XlSf[iQ,4]
    XqSf[iQ,20] = XlSf[iQ,2]*XlSf[iQ,5]
    XqSf[iQ,21] = XlSf[iQ,2]*XlSf[iQ,6]
    XqSf[iQ,22] = XlSf[iQ,2]*XlSf[iQ,7]
    XqSf[iQ,23] = XlSf[iQ,2]*XlSf[iQ,8]
    XqSf[iQ,24] = XlSf[iQ,2]*XlSf[iQ,9]
    
    XqSf[iQ,25] = XlSf[iQ,3]*XlSf[iQ,4]
    XqSf[iQ,26] = XlSf[iQ,3]*XlSf[iQ,5]
    XqSf[iQ,27] = XlSf[iQ,3]*XlSf[iQ,6]
    XqSf[iQ,28] = XlSf[iQ,3]*XlSf[iQ,7]
    XqSf[iQ,29] = XlSf[iQ,3]*XlSf[iQ,8]
    XqSf[iQ,30] = XlSf[iQ,3]*XlSf[iQ,9]
    
    XqSf[iQ,31] = XlSf[iQ,4]*XlSf[iQ,5]
    XqSf[iQ,32] = XlSf[iQ,4]*XlSf[iQ,6]
    XqSf[iQ,33] = XlSf[iQ,4]*XlSf[iQ,7]
    XqSf[iQ,34] = XlSf[iQ,4]*XlSf[iQ,8]
    XqSf[iQ,35] = XlSf[iQ,4]*XlSf[iQ,9]
    
    XqSf[iQ,36] = XlSf[iQ,5]*XlSf[iQ,6]
    XqSf[iQ,37] = XlSf[iQ,5]*XlSf[iQ,7]
    XqSf[iQ,38] = XlSf[iQ,5]*XlSf[iQ,8]
    XqSf[iQ,39] = XlSf[iQ,5]*XlSf[iQ,9]
    
    XqSf[iQ,40] = XlSf[iQ,6]*XlSf[iQ,7]
    XqSf[iQ,41] = XlSf[iQ,6]*XlSf[iQ,8]
    XqSf[iQ,42] = XlSf[iQ,6]*XlSf[iQ,9]
    
    XqSf[iQ,43] = XlSf[iQ,7]*XlSf[iQ,8]
    XqSf[iQ,44] = XlSf[iQ,7]*XlSf[iQ,9]
    
    XqSf[iQ,45] = XlSf[iQ,8]*XlSf[iQ,9]
    
    #   squares 
    XqSf[iQ,46] = XlSf[iQ,1]*XlSf[iQ,1]
    XqSf[iQ,47] = XlSf[iQ,2]*XlSf[iQ,2]
    XqSf[iQ,48] = XlSf[iQ,3]*XlSf[iQ,3]
    XqSf[iQ,49] = XlSf[iQ,4]*XlSf[iQ,4]
    XqSf[iQ,50] = XlSf[iQ,5]*XlSf[iQ,5]
    XqSf[iQ,51] = XlSf[iQ,6]*XlSf[iQ,6]
    XqSf[iQ,52] = XlSf[iQ,7]*XlSf[iQ,7]
    XqSf[iQ,53] = XlSf[iQ,8]*XlSf[iQ,8]
    XqSf[iQ,54] = XlSf[iQ,9]*XlSf[iQ,9]
    
XqSf[:,0] = 1 
Xq2Sf = np.dot(XqSf.T, XqSf)  

#   get y values for Sf 
ySV = XlSf[:,1] * rangeSV / 2. + meanSV 
ySH = XlSf[:,2] * rangeSH / 2. + meanSH 
ySh = XlSf[:,3] * rangeSh / 2. + meanSh 
yPf = XlSf[:,4] * rangePf / 2. + meanPf
ySHaz = XlSf[:,5] * rangeSHaz / 2. + meanSHaz 
ySHaz[np.ix_(ySHaz < 0.)] += 180.
yMu = XlSf[:,8] * rangeMu / 2. + meanMu 
yC0 = XlSf[:,9] * rangeC0 / 2. + meanC0 

#   start Ts CDF figure for all faults 
xSV = (SV - meanSV) / (rangeSV / 2.)
xSH = (SH - meanSH) / (rangeSH / 2.)  
xSh = (Sh - meanSh) / (rangeSh / 2.)  
xPf = (Pf - meanPf) / (rangePf / 2.) 
xSHaz = (SHazTrue - meanSHaz) / (rangeSHaz / 2.)  
xMu = (Mu - meanMu) / (rangeMu / 2.)
xC0 = (C0 - meanC0) / (rangeC0 / 2.)

fig, ax = plt.subplots(figsize=(6,4))
#   arbitrary critical Pf value to assess fault stability 
PfCrit = 5.     #   in MPa
sColour = []
ysigmaN = np.zeros([nSf,])
ytau = np.zeros([nSf,])

#   for each fault  
nRed = 0
nOrange = 0
nGreen = 0  
#for j in range(0, nFaults):
for j in tqdm(range(nFaults)):
    
    #   set fault strike, with +/- 1 degree 'error' 
    thisStrike = Strike[j]
    minStrike = thisStrike - 1. 
    maxStrike = thisStrike + 1. 
    meanStrike = thisStrike 
    rangeStrike = maxStrike - minStrike 
    
    #   set fault dip, with +/- 1 degree 'error' 
    thisDip = Dip[j]
    minDip = thisDip - 1. 
    maxDip = thisDip + 1. 
    meanDip = thisDip 
    rangeDip = maxDip - minDip 
    
    yStrike = XlSf[:,6] * rangeStrike / 2. + meanStrike 
    yStrike[np.ix_(yStrike < 0.)] += 360.

    yDip = XlSf[:,7] * rangeDip / 2. + meanDip 

    for i in range(0, nSf):
        #   Andersonian normal: sV = s1, sH = s2, sh = s3 
        ysigmaN[i], ytau[i] = pfs.calcAndersonianStressOnPlane(ySV[i], ySH[i], ySh[i], ySHaz[i], thisStrike, thisDip) 
            
    ySf = (ysigmaN - yPf) - ((ytau - yC0) / yMu) 

    #   get y values for Sf
    ySfq = np.zeros([int(q2Sf)+1,])
    for i in range(0, int(q2Sf)+1):
        for k in range(0, nSf):
            ySfq[i] = ySfq[i] + XqSf[k,i] * ySf[k] 
            
    ySfq[0] = sum(ySf)
        
    #   solve for beta using least squares 
    BetaSfq, _, _, _ = linalg.lstsq(Xq2Sf, ySfq)

    ###############################################################################
    #   Monte Carlo simulation of Sf using quadratic response surface 
    mcSfq = np.zeros([nMC,])
    xStrike = (thisStrike - meanStrike) / (rangeStrike / 2.)  
    xDip = (thisDip - meanDip) / (rangeDip / 2.)  

    for i in range(0, nMC):
        mcSfq[i] = (BetaSfq[0] + 
                    BetaSfq[1] * xSV[i] +
                    BetaSfq[2] * xSH[i] +
                    BetaSfq[3] * xSh[i] +
                    BetaSfq[4] * xPf[i] +
                    BetaSfq[5] * xSHaz[i] +
                    BetaSfq[6] * xStrike +
                    BetaSfq[7] * xDip +
                    BetaSfq[8] * xMu[i] +
                    BetaSfq[9] * xC0[i] +
                    
                    BetaSfq[10] * xSV[i] * xSH[i] + 
                    BetaSfq[11] * xSV[i] * xSh[i] + 
                    BetaSfq[12] * xSV[i] * xPf[i] + 
                    BetaSfq[13] * xSV[i] * xSHaz[i] + 
                    BetaSfq[14] * xSV[i] * xStrike + 
                    BetaSfq[15] * xSV[i] * xDip + 
                    BetaSfq[16] * xSV[i] * xMu[i] + 
                    BetaSfq[17] * xSV[i] * xC0[i] + 
                    
                    BetaSfq[18] * xSH[i] * xSh[i] + 
                    BetaSfq[19] * xSH[i] * xPf[i] + 
                    BetaSfq[20] * xSH[i] * xSHaz[i] + 
                    BetaSfq[21] * xSH[i] * xStrike + 
                    BetaSfq[22] * xSH[i] * xDip + 
                    BetaSfq[23] * xSH[i] * xMu[i] + 
                    BetaSfq[24] * xSH[i] * xC0[i] + 
                    
                    BetaSfq[25] * xSh[i] * xPf[i] + 
                    BetaSfq[26] * xSh[i] * xSHaz[i] + 
                    BetaSfq[27] * xSh[i] * xStrike + 
                    BetaSfq[28] * xSh[i] * xDip + 
                    BetaSfq[29] * xSh[i] * xMu[i] + 
                    BetaSfq[30] * xSh[i] * xC0[i] + 
                    
                    BetaSfq[31] * xPf[i] * xSHaz[i] + 
                    BetaSfq[32] * xPf[i] * xStrike + 
                    BetaSfq[33] * xPf[i] * xDip + 
                    BetaSfq[34] * xPf[i] * xMu[i] + 
                    BetaSfq[35] * xPf[i] * xC0[i] + 
    
                    BetaSfq[36] * xSHaz[i] * xStrike + 
                    BetaSfq[37] * xSHaz[i] * xDip + 
                    BetaSfq[38] * xSHaz[i] * xMu[i] + 
                    BetaSfq[39] * xSHaz[i] * xC0[i] + 
                    
                    BetaSfq[40] * xStrike * xDip + 
                    BetaSfq[41] * xStrike * xMu[i] + 
                    BetaSfq[42] * xStrike * xC0[i] + 
                    
                    BetaSfq[43] * xDip * xMu[i] + 
                    BetaSfq[44] * xDip * xC0[i] + 
                    
                    BetaSfq[45] * xMu[i] * xC0[i] + 
                    
                    BetaSfq[46] * xSV[i] * xSV[i] + 
                    BetaSfq[47] * xSH[i] * xSH[i] + 
                    BetaSfq[48] * xSh[i] * xSh[i] + 
                    BetaSfq[49] * xPf[i] * xPf[i] + 
                    BetaSfq[50] * xSHaz[i] * xSHaz[i] + 
                    BetaSfq[51] * xStrike * xStrike + 
                    BetaSfq[52] * xDip * xDip +
                    BetaSfq[53] * xMu[i] * xMu[i] +
                    BetaSfq[54] * xC0[i] * xC0[i])

    #   calc CDF from all nMC runs for this fault  
    sortSf = np.sort(mcSfq)
    sortSfadj = sortSf - sortSf.min()
    cumSf = (np.cumsum(sortSfadj) / np.sum(sortSfadj)) * 100.
    
    #   P value for Sf > 5 MPa (say)
    if np.min(sortSf) <= PfCrit and np.max(sortSf) >= PfCrit :    
        PSfCrit = np.max(cumSf[np.ix_(sortSf < PfCrit)])
        if PSfCrit > 33.:
            sColour.append('red')
            nRed += 1 
        elif PSfCrit < 1.:
            sColour.append('green')
            nGreen += 1 
        elif PSfCrit >= 1. and PSfCrit <= 33.:
            sColour.append('orange')
            nOrange += 1 
    elif np.max(sortSf) < PfCrit:
        PSfCrit = 100.
        sColour.append('red') 
        nRed += 1 
    else:
        PSfCrit = 0.
        sColour.append('green') 
        nGreen += 1 

    ax.plot(sortSf, cumSf, color=sColour[j], lw=0.5)

#   save CDF figure for all faults 
ax.plot([-100., 100.], [33., 33.], '--r')
ax.plot([PfCrit, PfCrit], [0., 100.], '-r')
ax.grid(True)
ax.set_xlim(-100., 100.)
ax.set_ylim(0., 100.) 
ax.set_xlabel(r'Fracture Susceptibility, MPa')
ax.set_ylabel('Conditional Probability, %')
ax.set_title(r'CDFs from MC simulation: $N_{MC}$=%i, $N_{faults}$=%i' % (nMC, nFaults))
plt.savefig('pfsRuhrBochum_Sf - CDF.png', dpi=300)

print('Proportions of red, orange & green faults:')
print(nRed/nFaults, nOrange/nFaults, nGreen/nFaults)

toc = time.time()
print('\n...finished pfsRuhrBochum_Sf, %3.2f sec elapsed.' % (toc-tic))
