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

print('\nStarting pfsRuhrBochum_Ts, version 1...')
print('*** Ts (effective) analysis for Ruhr-Bochum - deep carbonate ***\n')
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
plt.savefig('RuhrBochum Input Histograms.png', dpi=300)

###########################################################################
#   Response Surface Method (RSM)
#   build a quadratic response surface using these distributions 
#   linear regression over multiple variables, with linear & quadratic fits 
#   assume 3^q design of calculation points - min, max and mean of each variable 
#   where q is the number of variables in each term:
#      for Ts q = 7
#            for the parameters sV, sH, sh, sHaz, pf, strike & dip  
#   all assuming Andersonian stresses with one principal stress vertical 
qTs = 7      

# ###############################################################################
# #   linear fit for Sf 
nTs = pow(3, qTs)         #   3 coordinates for each variable in the RS

# #   solve y = BetaSf * XlTs for BetaSf as the coefficients in the regression
# #   XlTs is the 'design matrix' of dummy variables; one per variable 
# #   values of -1 = min, 0 = mean, +1 = max
XlTs = np.zeros([nTs, qTs+1])
iRow = 0 
for iA in range(-1,+2):                         #   sV
    for iB in range(-1,+2):                     #   sH
        for iC in range(-1,+2):                 #   sh
            for iD in range(-1,+2):             #   sHaz
                for iE in range(-1,+2):         #   strike 
                    for iF in range(-1,+2):     #   dip 
                        for iG in range(-1,+2): #   pf 
                            XlTs[iRow, :] = [1, iA, iB, iC, iD, iE, iF, iG]
                            iRow += 1

# ###############################################################################
# #   quadratic fit for Ts     
q2Ts = 0.5 * qTs * (qTs + 1) + qTs  #   total terms incl. squares and products 
XqTs = np.zeros([int(nTs), int(q2Ts)+1])
for iQ in range(0, nTs):
    #   linear terms 
    XqTs[iQ,1] = XlTs[iQ,1]
    XqTs[iQ,2] = XlTs[iQ,2]
    XqTs[iQ,3] = XlTs[iQ,3]
    XqTs[iQ,4] = XlTs[iQ,4]
    XqTs[iQ,5] = XlTs[iQ,5]
    XqTs[iQ,6] = XlTs[iQ,6]
    XqTs[iQ,7] = XlTs[iQ,7]
    
    #   cross products 
    XqTs[iQ,8] = XlTs[iQ,1]*XlTs[iQ,2]
    XqTs[iQ,9] = XlTs[iQ,1]*XlTs[iQ,3]
    XqTs[iQ,10] = XlTs[iQ,1]*XlTs[iQ,4]
    XqTs[iQ,11] = XlTs[iQ,1]*XlTs[iQ,5]
    XqTs[iQ,12] = XlTs[iQ,1]*XlTs[iQ,6]
    XqTs[iQ,13] = XlTs[iQ,1]*XlTs[iQ,7]
    
    XqTs[iQ,14] = XlTs[iQ,2]*XlTs[iQ,3]
    XqTs[iQ,15] = XlTs[iQ,2]*XlTs[iQ,4]
    XqTs[iQ,16] = XlTs[iQ,2]*XlTs[iQ,5]
    XqTs[iQ,17] = XlTs[iQ,2]*XlTs[iQ,6]
    XqTs[iQ,18] = XlTs[iQ,2]*XlTs[iQ,7]
    
    XqTs[iQ,19] = XlTs[iQ,3]*XlTs[iQ,4]
    XqTs[iQ,20] = XlTs[iQ,3]*XlTs[iQ,5]
    XqTs[iQ,21] = XlTs[iQ,3]*XlTs[iQ,6]
    XqTs[iQ,22] = XlTs[iQ,3]*XlTs[iQ,7]
    
    XqTs[iQ,23] = XlTs[iQ,4]*XlTs[iQ,5]
    XqTs[iQ,24] = XlTs[iQ,4]*XlTs[iQ,6]
    XqTs[iQ,25] = XlTs[iQ,4]*XlTs[iQ,7]
    
    XqTs[iQ,26] = XlTs[iQ,5]*XlTs[iQ,6]
    XqTs[iQ,27] = XlTs[iQ,5]*XlTs[iQ,7]
    
    XqTs[iQ,28] = XlTs[iQ,6]*XlTs[iQ,7]
    
    #   squares 
    XqTs[iQ,29] = XlTs[iQ,1]*XlTs[iQ,1]
    XqTs[iQ,30] = XlTs[iQ,2]*XlTs[iQ,2]
    XqTs[iQ,31] = XlTs[iQ,3]*XlTs[iQ,3]
    XqTs[iQ,32] = XlTs[iQ,4]*XlTs[iQ,4]
    XqTs[iQ,33] = XlTs[iQ,5]*XlTs[iQ,5]
    XqTs[iQ,34] = XlTs[iQ,6]*XlTs[iQ,6]
    XqTs[iQ,35] = XlTs[iQ,7]*XlTs[iQ,7]
    
XqTs[:,0] = 1 
Xq2Ts = np.dot(XqTs.T, XqTs)  

# #   get y values for Ts 
ySV = XlTs[:,1] * rangeSV / 2. + meanSV 
ySH = XlTs[:,2] * rangeSH / 2. + meanSH 
ySh = XlTs[:,3] * rangeSh / 2. + meanSh 
ySHaz = XlTs[:,4] * rangeSHaz / 2. + meanSHaz 
ySHaz[np.ix_(ySHaz < 0.)] += 180.
yPf = XlTs[:,7] * rangePf / 2. + meanPf 

# #   start Ts CDF figure for all faults 
xSV = (SV - meanSV) / (rangeSV / 2.)
xSH = (SH - meanSH) / (rangeSH / 2.)  
xSh = (Sh - meanSh) / (rangeSh / 2.)  
xSHaz = (SHazTrue - meanSHaz) / (rangeSHaz / 2.)  
xPf = (Pf - meanPf) / (rangePf / 2.) 

fig, ax = plt.subplots(figsize=(6,4))
#   arbitrary critical friction value to assess fault stability 
muCrit = 0.6 
sColour = []
strikeTsq = np.zeros([nFaults,])
ysigmaN = np.zeros([nTs,])
ytau = np.zeros([nTs,])

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
    
    #   do regression for this fault 
    yStrike = XlTs[:,5] * rangeStrike / 2. + meanStrike 
    yStrike[np.ix_(yStrike < 0.)] += 360.

    yDip = XlTs[:,6] * rangeDip / 2. + meanDip 

    for i in range(0, nTs):
        ysigmaN[i], ytau[i] = pfs.calcEffAndersonianStressOnPlane(ySV[i], ySH[i], ySh[i], ySHaz[i], yStrike[i], yDip[i], yPf[i]) 
    yTs = ytau / ysigmaN
    
    #   get y values for Ts
    yTsq = np.zeros([int(q2Ts)+1,])
    for i in range(0, int(q2Ts)+1):
        for k in range(0, nTs):
            yTsq[i] = yTsq[i] + XqTs[k,i] * yTs[k] 
            
    yTsq[0] = sum(yTs)
        
    #   solve for beta using least squares 
    BetaTsq, res, rank, s = linalg.lstsq(Xq2Ts, yTsq)
    
    #   analyse variance, residuals etc 
#    RsqTsq, RsqATsq, FTsq, MSRTsq, MSETsq = frs.calcANOVAresid(yTs, XqTs, BetaTsq)
#    sRsqTs = 'Adjusted R^2 for Ts fit: {:.3f}'
#    print(sRsqTs.format(RsqATsq))
    
    # #   get RS Ts value for this strike, this fault 
    xStrike = (thisStrike - meanStrike) / (rangeStrike / 2.)  
    xDip = (thisDip - meanDip) / (rangeDip / 2.)  
    
    #   run nMC calculations of Ts using set ranges     
    mcTsq = np.zeros([nMC,])
    for i in range(0, nMC):
        
        mcTsq[i] = abs(BetaTsq[0] + 
                        BetaTsq[1] * xSV[i] +
                        BetaTsq[2] * xSH[i] +
                        BetaTsq[3] * xSh[i] +
                        BetaTsq[4] * xSHaz[i] +
                        BetaTsq[5] * xStrike +
                        BetaTsq[6] * xDip +
                        BetaTsq[7] * xPf[i] +
                    
                        BetaTsq[8] * xSV[i] * xSH[i] + 
                        BetaTsq[9] * xSV[i] * xSh[i] + 
                        BetaTsq[10] * xSV[i] * xSHaz[i] + 
                        BetaTsq[11] * xSV[i] * xStrike + 
                        BetaTsq[12] * xSV[i] * xDip + 
                        BetaTsq[13] * xSV[i] * xPf[i] + 
                    
                        BetaTsq[14] * xSH[i] * xSh[i] + 
                        BetaTsq[15] * xSH[i] * xSHaz[i] + 
                        BetaTsq[16] * xSH[i] * xStrike + 
                        BetaTsq[17] * xSH[i] * xDip + 
                        BetaTsq[18] * xSH[i] * xPf[i] + 
                    
                        BetaTsq[19] * xSh[i] * xSHaz[i] + 
                        BetaTsq[20] * xSh[i] * xStrike + 
                        BetaTsq[21] * xSh[i] * xDip + 
                        BetaTsq[22] * xSh[i] * xPf[i] + 
                    
                        BetaTsq[23] * xSHaz[i] * xStrike + 
                        BetaTsq[24] * xSHaz[i] * xDip + 
                        BetaTsq[25] * xSHaz[i] * xPf[i] + 
                    
                        BetaTsq[26] * xStrike * xDip + 
                        BetaTsq[27] * xStrike * xPf[i] + 
                    
                        BetaTsq[28] * xDip * xPf[i] +

                        BetaTsq[29] * xSV[i] * xSV[i] + 
                        BetaTsq[30] * xSH[i] * xSH[i] + 
                        BetaTsq[31] * xSh[i] * xSh[i] + 
                        BetaTsq[32] * xSHaz[i] * xSHaz[i] + 
                        BetaTsq[33] * xStrike * xStrike + 
                        BetaTsq[34] * xDip * xDip +
                        BetaTsq[35] * xPf[i] * xPf[i])
       
    #   calc CDF from all nMC runs for this fault segment  
    sortTs = np.sort(mcTsq)
    cumTs = (np.cumsum(sortTs) / np.sum(sortTs)) * 100.
    
    #   get the P value for the chosen critical Ts
    #   colour code this fault segment 
    if np.min(sortTs) <= muCrit and np.max(sortTs) >= muCrit:    
        PTsCrit = np.min(cumTs[np.ix_(sortTs > muCrit)])
        if PTsCrit > 99.:
            sColour.append('green')
            nGreen += 1 
        elif PTsCrit < 67.:
            sColour.append('red')
            nRed += 1 
        elif PTsCrit < 99. and PTsCrit > 67.:
            sColour.append('orange')
            nOrange += 1 
    elif np.min(sortTs) > muCrit:
        PTsCrit = 0.
        sColour.append('red')
        nRed += 1 
    else:
        PTsCrit = 100.
        sColour.append('green') 
        nGreen += 1 
        
    ax.plot(sortTs, cumTs, color=sColour[j], lw=0.5)
    

#   save CDF figure for all faults 
ax.plot([0., 1.], [67., 67.], '--r')
ax.plot([muCrit, muCrit], [0., 100.], '-r')
ax.fill_betweenx([0., 100.], 0.6, 0.85, color='pink', alpha=0.2)
ax.grid(True)
ax.set_xlim(0., 1.)
ax.set_ylim(0., 100.) 
ax.set_xlabel('Slip Tendency (Effective), T$_s$')
ax.set_ylabel('Conditional Probability, %')
ax.set_title(r'CDFs from MC simulation: $N_{MC}$=%i, $N_{faults}$=%i' % (nMC, nFaults))
plt.savefig('RuhrBochum fault CDFs.png', dpi=300)

print('\nProportions of red, orange & green faults:')
print(nRed/nFaults, nOrange/nFaults, nGreen/nFaults)

#   write out the CDF colour-code for each fault 
fOut = open("pfsRuhrBochum_faults_colourcoded.txt", "w")
for i in range(nFaults):
    sLine = str(int(FaultID[i])) + ", " + sColour[i] + '\n'
    fOut.write(sLine)
fOut.close() 

toc = time.time()
print('\n...finished pfsRuhrBochum_Ts, %3.2f sec elapsed.' % (toc-tic))
