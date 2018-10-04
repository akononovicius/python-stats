#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def MakeCdf(data,start=None,stop=None,outPoints=100):
    if(start==None):
        start=data.min()
    if(stop==None):
        stop=data.max()
    bins=np.linspace(start,stop,num=outPoints)
    histogram=np.histogram(data,bins=bins,normed=False)[0]
    cdf=np.vstack([bins,np.append([0],np.cumsum(histogram))]).T
    cdf[:,1]/=np.sum(histogram)
    return cdf

# Obtain CDF using Kaplan-Meier estimator.
# The CDF is the same as the obtaiend by MakeCDF (up to numerical
# rounding errors). In order for Kaplan-Meier estimator to yield
# different result censored data must be present. This function
# would need to be further modified to work with censored data
# (likely to be extended in the future).
def MakeKMCdf(data,start=None,stop=None,outPoints=100):
    def __KaplanMeierEstimate(data,boundaries):
        surviving=np.sum(data>=boundaries[0])
        d=data[(boundaries[0]<=data) & (data<boundaries[1])].copy()
        if(len(d)==0):
            return 1
        d=np.sort(d)
        result=1
        for p in np.unique(d):
            dead=np.sum(d==p)
            result*=(1-dead/surviving)
            surviving-=dead
        return result
    if(start is None):
        start=np.min(data)
    if(stop is None):
        stop=np.max(data)
    bins=np.linspace(start,stop,num=outPoints)
    bounds=np.vstack([bins[:-1],bins[1:]]).T
    kme=[1]
    for b in bounds:
        kme+=[kme[-1]*__KaplanMeierEstimate(data,b),]
    del b
    kme=np.array(kme)
    cdf=np.vstack([bins,1-kme]).T
    return cdf
