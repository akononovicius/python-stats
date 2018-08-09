#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

#averaging of log-log stats
def AverageOverLogLog(arrs,outPoints=100):
    def _GetBinWidth(arr,i):
        before=0
        if(i>0):
            before=arr[i-1,0]
        nex=arr[i,0]
        if(i<len(arr)-1):
            nex=arr[i+1,0]
        return (nex-before)*0.5
    def _ResampleArea(arr,bins):
        rez=[]    
        i=0
        for b in bins:
            integral=0
            tWidth=0
            while(i<len(arr) and arr[i,0]<b):
                width=_GetBinWidth(arr,i)
                tWidth=tWidth+width
                integral=integral+arr[i,1]*width
                i=i+1
            if(tWidth>0):
                rez=rez+[integral/tWidth,]
            else:
                rez=rez+[0,]
        return np.array(rez)
    minX=np.min([np.min(arrs[i][:,0]) for i in range(len(arrs))])
    maxX=np.max([np.max(arrs[i][:,0]) for i in range(len(arrs))])
    steps=np.logspace(math.log10(minX),math.log10(maxX),outPoints)
    rez=np.mean(np.array([_ResampleArea(arr,steps) for arr in arrs]),axis=0)
    steps=steps*math.sqrt(steps[0]/steps[1])
    rez=np.vstack((steps,rez)).T
    return rez[rez[:,1]>0]
