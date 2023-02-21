#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:55:20 2021

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

print('\nStarting pfsRuhrBochum_Td - DILATION TENDENCY - version 1...')
print('*** Td analysis for Ruhr-Bochum - deep carbonate ***\n')
tic = time.time() 

###########################################################################
#   initialise 
nMC = 5000        #   number of Monte Carlo runs; keep this small (e.g., 100) for testing 
nBins = 20        #   number of bins in histograms  

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

#   build distributions of each stress
#stress = pfs.setStressDist2(nMC, muSV, sdSV, muSH, sdSH, muSh, sdSh)
#SV = stress[:,0]
#SH = stress[:,1]
#Sh = stress[:,2]
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
#   plot summary histograms of input variables
 
fig, ax = plt.subplots(figsize=(12,12))

ax1 = plt.subplot(331)
nSV, bSV, pSV = ax1.hist(SV, nBins)

ax2 = plt.subplot(332)
nSH, bSH, pSH = ax2.hist(SH, nBins)

ax3 = plt.subplot(333)
nSh, bSh, pSh = ax3.hist(Sh, nBins)

nMax123 = np.max([nSh.max(), nSH.max(), nSV.max()])*1.05 

#   use modes rather than means as measure of skewed distributions  
modeSH = bSH[np.ix_(nSH == nSH.max())] + (bSH[1]-bSH[0])/2.
modeSh = bSh[np.ix_(nSh == nSh.max())] + (bSh[1]-bSh[0])/2.

ax1.plot([muSV, muSV], [0, np.max(nSV)], '-r')
ax1.text(SV.min()*1.01, nMax123*.75, 
              ('Mean: %3.1f\nStdDev: %2.1f' % (meanSV, stdSV)), 
                           bbox=dict(facecolor='white', edgecolor='black'))
ax1.set_ylim(0., nMax123)
ax1.set_xlabel(r'$\sigma_V$, MPa')
ax1.set_ylabel('Count')
ax1.grid(True)
ax1.set_title('Variation in vertical stress, n=%i' % nMC)

ax2.plot([modeSH[0], modeSH[0]], [0, np.max(nSH)], '-r')
ax2.text(SH.min()*1.01, nMax123*.75, 
              ('Mode: %3.1f\nStdDev: %2.1f' % (modeSH[0], stdSH)),
                           bbox=dict(facecolor='white', edgecolor='black'))
ax2.set_ylim(0., nMax123)
ax2.set_xlabel(r'$\sigma_{Hmax}$, MPa')
ax2.set_ylabel('Count')
ax2.grid(True)
ax2.set_title('Variation in max. horizontal stress, n=%i' % nMC)

ax3.plot([modeSh[0], modeSh[0]], [0, np.max(nSh)], '-r')
ax3.text(Sh.min()*1.01, nMax123*.75, 
              ('Mode: %3.1f\nStdDev: %2.1f' % (modeSh[0], stdSh)),
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
nPf, bPf, pPf = ax7.hist(Pf, nBins)

ax7.plot([muPf, muPf], [0, np.max(nPf)], '-r')
ax7.text(Pf.min()*1.01, nPf.max()*.75, 
              ('Mean: %3.1f\nStdDev: %2.1f' % (meanPf, stdPf)), 
                           bbox=dict(facecolor='white', edgecolor='black'))
ax7.set_ylim(0., nPf.max()*1.05)
ax7.set_xlabel(r'$P_f$, MPa')
ax7.set_ylabel('Count')
ax7.grid(True)
ax7.set_title('Variation in pore pressure, n=%i' % nMC)

plt.tight_layout() 
plt.savefig('RuhrBochum Input Histograms - Td.png', dpi=300)

###########################################################################
#   Response Surface Method (RSM)
#   build a quadratic response surface using these distributions 
#   linear regression over multiple variables, with linear & quadratic fits 
#   assume 3^q design of calculation points - min, max and mean of each variable 
#   where q is the number of variables in each term:
#      for Td q = 7
#            for the parameters sV, sH, sh, sHaz, pf, strike & dip  
#   all assuming Andersonian stresses with one principal stress vertical 
qTd = 7      

# ###############################################################################
# #   linear fit for Sf 
nTd = pow(3, qTd)         #   3 coordinates for each variable in the RS

# #   solve y = BetaSf * XlTs for BetaSf as the coefficients in the regression
# #   XlTs is the 'design matrix' of dummy variables; one per variable 
# #   values of -1 = min, 0 = mean, +1 = max
XlTd = np.zeros([nTd, qTd+1])
iRow = 0 
for iA in range(-1,+2):                         #   sV
    for iB in range(-1,+2):                     #   sH
        for iC in range(-1,+2):                 #   sh
            for iD in range(-1,+2):             #   sHaz
                for iE in range(-1,+2):         #   strike 
                    for iF in range(-1,+2):     #   dip 
                        for iG in range(-1,+2): #   pf 
                            XlTd[iRow, :] = [1, iA, iB, iC, iD, iE, iF, iG]
                            iRow += 1

# ###############################################################################
# #   quadratic fit for Ts     
q2Td = 0.5 * qTd * (qTd + 1) + qTd  #   total terms incl. squares and products 
XqTd = np.zeros([int(nTd), int(q2Td)+1])
for iQ in range(0, nTd):
    #   linear terms 
    XqTd[iQ,1] = XlTd[iQ,1]
    XqTd[iQ,2] = XlTd[iQ,2]
    XqTd[iQ,3] = XlTd[iQ,3]
    XqTd[iQ,4] = XlTd[iQ,4]
    XqTd[iQ,5] = XlTd[iQ,5]
    XqTd[iQ,6] = XlTd[iQ,6]
    XqTd[iQ,7] = XlTd[iQ,7]
    
    #   cross products 
    XqTd[iQ,8] = XlTd[iQ,1]*XlTd[iQ,2]
    XqTd[iQ,9] = XlTd[iQ,1]*XlTd[iQ,3]
    XqTd[iQ,10] = XlTd[iQ,1]*XlTd[iQ,4]
    XqTd[iQ,11] = XlTd[iQ,1]*XlTd[iQ,5]
    XqTd[iQ,12] = XlTd[iQ,1]*XlTd[iQ,6]
    XqTd[iQ,13] = XlTd[iQ,1]*XlTd[iQ,7]
    
    XqTd[iQ,14] = XlTd[iQ,2]*XlTd[iQ,3]
    XqTd[iQ,15] = XlTd[iQ,2]*XlTd[iQ,4]
    XqTd[iQ,16] = XlTd[iQ,2]*XlTd[iQ,5]
    XqTd[iQ,17] = XlTd[iQ,2]*XlTd[iQ,6]
    XqTd[iQ,18] = XlTd[iQ,2]*XlTd[iQ,7]
    
    XqTd[iQ,19] = XlTd[iQ,3]*XlTd[iQ,4]
    XqTd[iQ,20] = XlTd[iQ,3]*XlTd[iQ,5]
    XqTd[iQ,21] = XlTd[iQ,3]*XlTd[iQ,6]
    XqTd[iQ,22] = XlTd[iQ,3]*XlTd[iQ,7]
    
    XqTd[iQ,23] = XlTd[iQ,4]*XlTd[iQ,5]
    XqTd[iQ,24] = XlTd[iQ,4]*XlTd[iQ,6]
    XqTd[iQ,25] = XlTd[iQ,4]*XlTd[iQ,7]
    
    XqTd[iQ,26] = XlTd[iQ,5]*XlTd[iQ,6]
    XqTd[iQ,27] = XlTd[iQ,5]*XlTd[iQ,7]
    
    XqTd[iQ,28] = XlTd[iQ,6]*XlTd[iQ,7]
    
    #   squares 
    XqTd[iQ,29] = XlTd[iQ,1]*XlTd[iQ,1]
    XqTd[iQ,30] = XlTd[iQ,2]*XlTd[iQ,2]
    XqTd[iQ,31] = XlTd[iQ,3]*XlTd[iQ,3]
    XqTd[iQ,32] = XlTd[iQ,4]*XlTd[iQ,4]
    XqTd[iQ,33] = XlTd[iQ,5]*XlTd[iQ,5]
    XqTd[iQ,34] = XlTd[iQ,6]*XlTd[iQ,6]
    XqTd[iQ,35] = XlTd[iQ,7]*XlTd[iQ,7]
    
XqTd[:,0] = 1 
Xq2Td = np.dot(XqTd.T, XqTd)  

# #   get y values for Ts 
ySV = XlTd[:,1] * rangeSV / 2. + meanSV 
ySH = XlTd[:,2] * rangeSH / 2. + meanSH 
ySh = XlTd[:,3] * rangeSh / 2. + meanSh 
ySHaz = XlTd[:,4] * rangeSHaz / 2. + meanSHaz 
ySHaz[np.ix_(ySHaz < 0.)] += 180.
yPf = XlTd[:,7] * rangePf / 2. + meanPf 

# #   start Ts CDF figure for all faults 
xSV = (SV - meanSV) / (rangeSV / 2.)
xSH = (SH - meanSH) / (rangeSH / 2.)  
xSh = (Sh - meanSh) / (rangeSh / 2.)  
xSHaz = (SHazTrue - meanSHaz) / (rangeSHaz / 2.)  
xPf = (Pf - meanPf) / (rangePf / 2.) 

fig, ax = plt.subplots(figsize=(6,4))
sColour = []
strikeTsq = np.zeros([nFaults,])
ysigmaN = np.zeros([nTd,])
ytau = np.zeros([nTd,])

#   for each fault  
TdCrit = 0.5 
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
    
    #   do regression for this fault 
    yStrike = XlTd[:,5] * rangeStrike / 2. + meanStrike 
    yStrike[np.ix_(yStrike < 0.)] += 360.

    yDip = XlTd[:,6] * rangeDip / 2. + meanDip 

    ySigma1 = np.zeros(nTd,)
    ySigma3 = np.zeros(nTd,)
    for i in range(0, nTd):
        ysigmaN[i], ytau[i] = pfs.calcAndersonianStressOnPlane(ySV[i], ySH[i], ySh[i], ySHaz[i], thisStrike, thisDip) 
        ySigma1[i] = np.max([ySV[i], ySH[i], ySh[i]])
        ySigma3[i] = np.min([ySV[i], ySH[i], ySh[i]])
        if ysigmaN[i] < ySigma3[i]:
            print('\nNormal stress < minimum stress!')
        
    yTd = (ySigma1 - ysigmaN) / (ySigma1 - ySigma3) 
    
    #   get y values for Td
    yTdq = np.zeros([int(q2Td)+1,])
    for i in range(0, int(q2Td)+1):
        for k in range(0, nTd):
            yTdq[i] = yTdq[i] + XqTd[k,i] * yTd[k] 
            
    yTdq[0] = sum(yTd)
        
    #   solve for beta using least squares 
    BetaTdq, res, rank, s = linalg.lstsq(Xq2Td, yTdq)
    
    #   analyse variance, residuals etc 
#    RsqTsq, RsqATsq, FTsq, MSRTsq, MSETsq = frs.calcANOVAresid(yTs, XqTs, BetaTsq)
#    sRsqTs = 'Adjusted R^2 for Ts fit: {:.3f}'
#    print(sRsqTs.format(RsqATsq))
    
    # #   get RS Ts value for this strike, this fault 
    xStrike = (thisStrike - meanStrike) / (rangeStrike / 2.)  
    xDip = (thisDip - meanDip) / (rangeDip / 2.)  
    
    #   run nMC calculations of Ts using set ranges     
    mcTdq = np.zeros([nMC,])
    for i in range(0, nMC):
        
        mcTdq[i] = abs(BetaTdq[0] + 
                        BetaTdq[1] * xSV[i] +
                        BetaTdq[2] * xSH[i] +
                        BetaTdq[3] * xSh[i] +
                        BetaTdq[4] * xSHaz[i] +
                        BetaTdq[5] * xStrike +
                        BetaTdq[6] * xDip +
                        BetaTdq[7] * xPf[i] +
                    
                        BetaTdq[8] * xSV[i] * xSH[i] + 
                        BetaTdq[9] * xSV[i] * xSh[i] + 
                        BetaTdq[10] * xSV[i] * xSHaz[i] + 
                        BetaTdq[11] * xSV[i] * xStrike + 
                        BetaTdq[12] * xSV[i] * xDip + 
                        BetaTdq[13] * xSV[i] * xPf[i] + 
                    
                        BetaTdq[14] * xSH[i] * xSh[i] + 
                        BetaTdq[15] * xSH[i] * xSHaz[i] + 
                        BetaTdq[16] * xSH[i] * xStrike + 
                        BetaTdq[17] * xSH[i] * xDip + 
                        BetaTdq[18] * xSH[i] * xPf[i] + 
                    
                        BetaTdq[19] * xSh[i] * xSHaz[i] + 
                        BetaTdq[20] * xSh[i] * xStrike + 
                        BetaTdq[21] * xSh[i] * xDip + 
                        BetaTdq[22] * xSh[i] * xPf[i] + 
                    
                        BetaTdq[23] * xSHaz[i] * xStrike + 
                        BetaTdq[24] * xSHaz[i] * xDip + 
                        BetaTdq[25] * xSHaz[i] * xPf[i] + 
                    
                        BetaTdq[26] * xStrike * xDip + 
                        BetaTdq[27] * xStrike * xPf[i] + 
                    
                        BetaTdq[28] * xDip * xPf[i] +

                        BetaTdq[29] * xSV[i] * xSV[i] + 
                        BetaTdq[30] * xSH[i] * xSH[i] + 
                        BetaTdq[31] * xSh[i] * xSh[i] + 
                        BetaTdq[32] * xSHaz[i] * xSHaz[i] + 
                        BetaTdq[33] * xStrike * xStrike + 
                        BetaTdq[34] * xDip * xDip +
                        BetaTdq[35] * xPf[i] * xPf[i])
       
    #   calc CDF from all nMC runs for this fault segment  
    sortTd = np.sort(mcTdq)
    cumTd = (np.cumsum(sortTd) / np.sum(sortTd)) * 100.
    
    #   get the P value for the chosen critical Ts
    #   colour code this fault segment 
    if np.min(sortTd) <= TdCrit and np.max(sortTd) >= TdCrit:    
        PTdCrit = np.min(cumTd[np.ix_(sortTd > TdCrit)])
        if PTdCrit > 99.:
            sColour.append('green')
            nGreen += 1 
        elif PTdCrit < 67.:
            sColour.append('red')
            nRed += 1 
        elif PTdCrit < 99. and PTdCrit > 67.:
            sColour.append('orange')
            nOrange += 1 
    elif np.min(sortTd) > TdCrit:
        PTdCrit = 0.
        sColour.append('red')
        nRed += 1 
    else:
        PTdCrit = 100.
        sColour.append('green') 
        nGreen += 1 
        
    ax.plot(sortTd, cumTd, color=sColour[j], lw=0.5)
    

#   save CDF figure for all faults 
ax.plot([0., 1.], [67., 67.], '--r')
ax.plot([TdCrit, TdCrit], [0., 100.], '-r')
ax.grid(True)
ax.set_xlim(0., 1.)
ax.set_ylim(0., 100.) 
ax.set_xlabel('Dilation Tendency (Effective), T$_s$')
ax.set_ylabel('Conditional Probability, %')
ax.set_title(r'CDFs from MC simulation: $N_{MC}$=%i, $N_{faults}$=%i' % (nMC, nFaults))
plt.savefig('RuhrBochum fault CDFs - Td.png', dpi=300)

print('\nProportions of red, orange & green faults:')
print(nRed/nFaults, nOrange/nFaults, nGreen/nFaults)

#   write out the CDF colour-code for each fault 
fOut = open("pfsRuhrBochum_faults_colourcoded.txt", "w")
for i in range(nFaults):
    sLine = str(int(FaultID[i])) + ", " + sColour[i] + '\n'
    fOut.write(sLine)
fOut.close() 

toc = time.time()
print('\n...finished pfsRuhrBochum_Td, %3.2f sec elapsed.' % (toc-tic))
