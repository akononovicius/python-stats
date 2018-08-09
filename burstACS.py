#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##
## Burst statistics analysis as described in [Gontis et al., ACS, 2012].
##

import numpy as np

#
# Prepend and append series with fake data so that first and last bursts
# do not become lost
#
def __PrepSeries(s,thresh,delta=1):
    series=s.copy()
    if(series[0]<thresh):
        series=np.append([thresh-delta,thresh+delta],series)
    else:
        series=np.append([thresh+delta,thresh-delta],series)
    if(series[-1]<thresh):
        series=np.append(series,[thresh+delta,thresh-delta])
    else:
        series=np.append(series,[thresh-delta,thresh+delta])
    return series

#
# Various stats extraction functions
#
def __ExtractBurstMax(s,bst,bd,tr):
    def _peak(s,fr,n,tr):
        return np.max(s[fr:fr+n]-tr)
    return np.array([_peak(s,bst[i],bd[i],tr)
                    for i in range(len(bst))])

def __ExtractBurstSize(s,bst,bd,tr,dt):
    def _size(s,fr,n,tr,h):
        return np.sum(s[fr:fr+n]-tr)*h
    return np.array([_size(s,bst[i],bd[i],tr,dt)
                    for i in range(len(bst))])

def __ExtractIBurstMin(s,bst,ibd,tr):
    def _ipeak(s,to,n,tr):
        return np.max(tr-s[to-n:to])
    return np.array([_ipeak(s,bst[i],ibd[i],tr)
                    for i in np.arange(1,len(bst)-1)])

def __ExtractIBurstSize(s,bst,ibd,tr,dt):
    def _isize(s,to,n,tr,h):
        return np.sum(tr-s[to-n:to])*h
    return np.array([_isize(s,bst[i],ibd[i],tr,dt)
                    for i in np.arange(1,len(bst)-1)])

#
# The main public extraction function
#
def ExtractBurstData(ser,thresh,samplePeriod=1,returnBurst=True,
                     returnInterBurst=False,extractOther=False,
                     prepSeries=False):
    if((not returnBurst) and (not returnInterBurst)):
        raise ValueError("The function will not return anything")
    rez=()

    series=ser.copy().astype(float)
    if(prepSeries):
        series=__PrepSeries(series,thresh=thresh,delta=0.1*thresh)

    eventTimes=np.where(series>=thresh)[0]
    interEventPeriods=np.diff(eventTimes)

    iearr=np.where(interEventPeriods>1)[0]
    burstStartTimes=eventTimes[iearr[:-1]+1]
    del eventTimes
    interBurstDuration=None
    if(returnInterBurst):
        interBurstDuration=interEventPeriods[iearr[:-1]]-1
    del interEventPeriods
    burstDuration=None
    if(returnBurst):
        burstDuration=np.diff(iearr)
    del iearr

    if(returnBurst):
        rez=rez+(burstDuration*samplePeriod,)
        if(extractOther):
            burstMax=__ExtractBurstMax(series,burstStartTimes,burstDuration,
                                      thresh)
            burstSize=__ExtractBurstSize(series,burstStartTimes,burstDuration,
                                      thresh,samplePeriod)
            rez=rez+(burstMax,)
            rez=rez+(burstSize,)
    if(returnInterBurst):
        rez=rez+(interBurstDuration*samplePeriod,)
        if(extractOther):
            interBurstMin=__ExtractIBurstMin(series,burstStartTimes,
                                            interBurstDuration,thresh)
            interBurstSize=__ExtractIBurstSize(series,burstStartTimes,
                                            interBurstDuration,thresh,
                                            samplePeriod)
            rez=rez+(interBurstMin,)
            rez=rez+(interBurstSize,)

    return rez
