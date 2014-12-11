# -*- coding: utf-8 -*-
"""Module to compute the autocorrelation function of a field,
directly from spectra, rather than via the power spectrum.

The algorithms here implemented are formally O(n^2)!
"""

import numpy as np
import _autocorr_priv

def autocorr_spectra(slist, nout=100):
    """
    Find the autocorrelation function from a list of spectra
    Spectra are assumed to be along the same axis.
    slist - list of quantity along spectra to autocorrelate. npix * nspectra
    nout - number of bins in output autocorrelation function
    """
    nspec = int(np.sqrt(np.shape(slist)[0]))
    assert nspec*nspec == np.shape(slist)[0]
    auto = _autocorr_priv.crosscorr_spectra(slist, slist, nspec, nout)
    return auto

def autocorr_list(slist, thresh, nout=100):
    """Find the autocorrelation function from a sparse list of discrete tracer points.
       The field is assumed to be 1 at these points and zero elsewhere
       list - list of points to autocorrelate. A tuple length n of 1xP arrays:
       the output of an np.where on an n-d field
       nbins - number of bins in output autocorrelation function
       size - size of the original field (assumed square), so field has dimensions (size,size..) n times
       weight - weight each point has: use 1/(avg. density)
       norm - If true, normalise by the number of possible cells in each bin
    """
    (spec, npix) = np.shape(slist)
    nspec = int(np.sqrt(spec))
    assert nspec*nspec == spec
    ind = np.where(slist > thresh)
    return _autocorr_priv.autocorr_list(ind[0], ind[1], nspec, npix, nout)

def crosscorr_spectra(slist1, slist2, thresh, nout=100):
    """
    Cross-correlate two quantities evaluated along spectra.
    One is assumed to be a sparse tracer (such as a DLA).
    slist1 is treated as a field
    slist2 is a tracer, everything > thresh is 1, < thresh is 0.
    """
    assert np.all(np.shape(slist1) == np.shape(slist2))
    (spec, _) = np.shape(slist1)
    nspec = int(np.sqrt(spec))
    assert nspec*nspec == spec
    ind = np.where(slist2 > thresh)
    return _autocorr_priv.crosscorr_list_spectra(slist1, ind[0], ind[1], nspec, nout)


def modecount(nspec, npix, nout=100):
    """Count the modes in each bin for the supplied regularly spaced spectra."""
    return _autocorr_priv.modecount(nspec, npix, nout)
