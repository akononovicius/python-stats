#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##
## implementation of the MF-DFA algorithm as described in:
## http://mokslasplius.lt/rizikos-fizika/multifractality-time-series
##

import numpy as np
import matplotlib.pyplot as plt

def fluctuations(profile,q,scale):
    # segment profile into equal chunks
    segments=int(len(profile) // scale)
    stridedProfile=np.lib.stride_tricks.as_strided(profile,shape=(segments,scale))
    # fit all chunks and evaluate fluctations
    xVals=np.arange(scale)
    fqs=np.zeros(segments)
    for i, segVals in enumerate(stridedProfile):
        coef=np.polyfit(xVals,segVals,1)
        fitVals=np.polyval(coef,xVals)
        fqs[i]=np.mean((segVals-fitVals)**2)
    return fqs

def dfa(profile,q,scaleSample,showFqs=False):
    # sample fluctuations in given points
    fqs=np.zeros(len(scaleSample))
    for i, s in enumerate(scaleSample):
        if(q!=0):
            fqs[i]=np.mean(fluctuations(profile,q,s)**(q/2))**(1/q)
        else:
            fqs[i]=np.exp(0.5*np.mean(np.log(fluctuations(profile,q,s))))
    if(showFqs):
        plt.plot(scaleSample,fqs,label="q="+str(q))
    return fqs

def mfdfa(series,qSample,scaleSample,showFqs=False):
    # obtain profile
    profile=np.cumsum(series-np.mean(series))
    logScaleSample=np.log10(scaleSample)
    hq=np.zeros(len(qSample))
    if(showFqs):
        plt.figure()
        plt.loglog()
        plt.xlabel('s')
        plt.ylabel('Fq(s)')
    for i, q in enumerate(qSample):
        logFqs=np.log10(dfa(profile,q,scaleSample,showFqs=showFqs))
        hq[i]=np.polyfit(logScaleSample,logFqs,1)[0]
    if(showFqs):
        plt.show()
    return hq

def segMfdfa(series,qSample,scaleSample,segmentSize=None):
    if(segmentSize is None):
        return mfdfa(series,qSample,scaleSample)
    hqs=()
    for i in np.arange(0,len(series)-segmentSize+1,segmentSize):
        hqs=hqs+(mfdfa(series[i:i+segmentSize],qSample,scaleSample),)
    return np.mean(np.vstack(hqs),axis=0)
