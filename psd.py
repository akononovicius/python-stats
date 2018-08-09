#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy.signal as sp
from stats.averageOverLogLog import AverageOverLogLog

#make PSD with frequencies eqi-sampled on log-scale
# returns PSD in lin-lin scale
def MakeLogPsd(data,fs=1,outPoints=100):
    psd=np.array(sp.periodogram(data,fs=fs)).T
    ids=np.unique(
            np.logspace(0,math.log10(psd.shape[0]),outPoints).astype(int)
        )-1
    ids=np.vstack((ids[1:-1],ids[2:])).T
    return np.array([
            [(psd[i[0],0]+psd[i[1],0])/2,np.mean(psd[i[0]:i[1],1])]
        for i in ids])

#if time series is longer than a given segment length, then it is split
# to parts of segment length. spliting is done both from the start and from
# the end (so that firstmost and foremost values are included)
#for each of the segment psd is obtained and later averaged over all segments
def MakeSegLogPsd(data,fs=1,outPoints=100,segment=262144,
                  biDirectionalSplit=True):
    def _ConvertToPower2(num):
        if(num > 0 and ((num & (num - 1)))):
            num=int(2**(int(math.floor(math.log2(num))+1)))
        return num
    segment=_ConvertToPower2(segment)
    if(len(data)<segment):
        segment=int(_ConvertToPower2(len(data))/2)
    if(segment<4096):
        return MakeLogPsd(data,fs=fs,outPoints=outPoints)
    psds=()
    im=int(math.floor(len(data)/segment))
    #do spliting from the start
    for i in range(im):
        start=i*segment
        psd=MakeLogPsd(data[start:start+segment],fs=fs,outPoints=10*outPoints)
        psds=psds+(psd,)
    l=len(data)
    if(biDirectionalSplit):
        #do spliting from the end
        for i in range(im):
            start=l-i*segment
            psd=MakeLogPsd(data[start-segment:start],fs=fs,
                           outPoints=10*outPoints)
            psds=psds+(psd,)
    del start, psd, im, l
    return AverageOverLogLog(psds,outPoints=outPoints)

#make PSD with values eqi-sampled on log-scale into a file
def SaveLogPsd(file,data,fs=1,outPoints=100,fmt="%.3f",returnData=False):
    psd=MakeLogPsd(data,fs=fs,outPoints=outPoints)
    np.savetxt(file,np.log10(psd),fmt=fmt)
    if(returnData):
        return psd
        
def SaveSegLogPsd(file,data,fs=1,outPoints=100,segment=262144,
                  fmt="%.3f",returnData=False):
    psd=MakeSegLogPsd(data,fs=fs,outPoints=outPoints,segment=segment)
    np.savetxt(file,np.log10(psd),fmt=fmt)
    if(returnData):
        return psd
