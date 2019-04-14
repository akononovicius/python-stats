#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

#make PDF with values eqi-sampled on lin-scale
# returns PDF in lin-lin scale    
def MakePdf(data,start=None,stop=None,outPoints=100):
    if(start==None):
        start=data.min()
    if(stop==None):
        stop=data.max()
    bins=np.linspace(start,stop,num=outPoints)
    histogram = np.histogram(data,bins=bins,normed=False)[0]
    pos=np.where(histogram==0)
    histogram=np.delete(histogram,pos)
    bins=np.delete(bins,pos)
    diffs=np.diff(bins)    
    bins=0.5*(bins[1:] + bins[:-1])
    pdf=np.array([bins,histogram/histogram.sum()/diffs]).T
    return pdf

#make PDF with values eqi-sampled on log-scale
# returns PDF in lin-lin scale
def MakeLogPdf(data,start=None,stop=None,outPoints=100):
    if(start==None):
        start=math.log10(data.min())
    else:
        start=math.log10(start)
    if(stop==None):
        stop=math.log10(data.max())
    else:
        stop=math.log10(stop)
    bins=np.logspace(start,stop,num=outPoints)
    histogram = np.histogram(data,bins=bins,normed=False)[0]
    pos=np.where(histogram==0)
    histogram=np.delete(histogram,pos)
    bins=np.delete(bins,pos)
    diffs=np.diff(bins)    
    bins=0.5*(bins[1:] + bins[:-1])
    pdf=np.array([bins,histogram/histogram.sum()/diffs]).T
    return pdf

#save PDF with values eqi-sampled on log-scale into a file
def SaveLogPdf(file,data,start=None,stop=None,outPoints=100,
               fmt="%.3f",returnData=False):
    pdf=MakeLogPdf(data,start=start,stop=stop,outPoints=outPoints)
    np.savetxt(file,np.log10(pdf),fmt=fmt)
    if(returnData):
        return pdf

#convert PDF to CDF
def PDF2CDF(pdf):
    delta=np.diff(pdf[:,0])
    tCdf1=np.cumsum(pdf[1:,1]*delta)
    tCdf2=np.cumsum(pdf[:-1,1]*delta)

    cdf=0.5*tCdf1+0.5*tCdf2
    cdf=cdf/cdf[-1]
    cdf=np.concatenate(([0],cdf))
    
    return np.transpose([pdf[:,0],cdf])
