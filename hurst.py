#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

##
## Rescaled Range related functions
##
def __MeanRange(series, segmentSize, /, *, wrap=True):
    # Step 1: Splitting series into segments
    _nSegments = len(series) // segmentSize
    _segs = series[:_nSegments*segmentSize].reshape(_nSegments, segmentSize)
    if wrap:
        # wrap might be needed to account for the edge points too
        _segs = np.vstack((_segs, series[-_nSegments*segmentSize:].reshape(_nSegments, segmentSize)))
    # Step 2: Obtain standard deviation for each segment
    _stds = np.std(_segs, axis=1)
    # Step 3: Obtain profile (cumulative sum of values in the segment)
    _prof = np.cumsum(_segs, axis=1)
    del _segs
    # Step 5: Establish range
    _low = np.min(_prof, axis=1)
    _high = np.max(_prof, axis=1)
    _rng = _high - _low
    # Step 6: Calculate mean range / standard deviation
    return np.mean(_rng / _stds)

def __MeanRanges(series, segmentSizes, /, *, wrap=True):
    _ranges = np.zeros(len(segmentSizes))
    for _idx, _segmentSize in enumerate(segmentSizes):
        _ranges[_idx] = __MeanRange(series, _segmentSize, wrap=wrap)
    return _ranges

def RescaledRange(series, lowSegmentSize, highSegmentSize, /, *, wrap=True, points=100):
    # NOTE: We will work only if series is stationary (equivalent to fractional Gaussian noise)
    _lss = np.log10(lowSegmentSize)
    _hss = np.log10(highSegmentSize)
    _segmentSizes = np.unique(np.floor(np.logspace(_lss, _hss, num = points)).astype(int))
    _segmentSizes = _segmentSizes[ _segmentSizes > 1 ]
    _ranges = __MeanRanges(series, _segmentSizes, wrap=wrap)
    return np.polyfit(np.log10(_segmentSizes), np.log10(_ranges), 1)[0]

##
## Box Counting method
##
def BoxCount1D(series, nSegments, /, *, wrap=True):
    # NOTE: this implementation will work only for one dimensional series (e.g. Cantor set)
    _segmentSize = len(series) // nSegments
    _segs = series[:nSegments*_segmentSize].reshape(nSegments, _segmentSize)
    if wrap:
        # wrap might be needed to account for the edge points too
        _segs = np.vstack((_segs, series[-nSegments*_segmentSize:].reshape(nSegments, _segmentSize)))
    return np.sum(np.sum(_segs,axis=1)>0) // 2
