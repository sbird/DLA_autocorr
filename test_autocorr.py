import _autocorr_priv
import numpy as np
import itertools as it

def testModecountSlow():
  """Test whether the slow 3D modecount correctly computes simple analytic tests.
     Note: no periodic wrapping!
  """
  modes = _autocorr_priv.modecount_slow(2,2,8)
  #Lower bin boundaries are at:
  #sqrt(3) / nout
  #0, 0.21, 0.42, ...
  #8 at 0 distance (bin 0)
  #24 at 1/2  (bin 2)
  #24 at 1/sqrt(2) (bin 3)
  #8 at sqrt(3/4) (bin 4)
  expected = np.array([ 8,  0, 24, 24,  8,  0,  0,  0], dtype=np.int32)
  assert np.all(modes  == expected)
  #Check different binning
  modes = _autocorr_priv.modecount_slow(2,2,12)
  expected = np.array([ 8,  0, 0, 24, 24, 0, 8,  0,  0,  0, 0, 0], dtype=np.int32)
  assert np.all(modes  == expected)

  #Check npix different from nspec
  modes = _autocorr_priv.modecount_slow(2,1,5)
  #4 at 0 distance (bin 0)
  #8 at 1/2  (bin 2)
  #4 at 1/sqrt(2) (bin 3)
  expected = np.array([ 4, 8, 4, 0, 0], dtype=np.int32)
  assert np.all(modes  == expected)
  modes = _autocorr_priv.modecount_slow(1,4,8)
  #4 at 0 distance (bin 0)
  #1+2+2+1 at 1/4  (bin 1)
  #4 at 1/2 (bin 2)
  #2 at 3/4 (bin 3)
  expected = np.array([ 4, 6, 4, 2, 0, 0, 0, 0], dtype=np.int32)
  assert np.all(modes  == expected)


def testModecountFastSlowBins():
  """Test whether the faster 3D modecount gives the same answer as the slow version.
     Use several different binnings.
  """
  values = (15, 14, 5, 18)
  for (spec, pix) in it.combinations(values, 2):
      for bins in (spec+pix, (spec+pix)*2, 42, 5):
          slow = _autocorr_priv.modecount_slow(spec,pix,bins)
          fast = _autocorr_priv.modecount(spec,pix,bins)
          assert np.all(slow == fast)

def testModecountFastSlow():
  """Test whether the faster 3D modecount gives the same answer as the slow version for larger values.
  This can be slow.
  """
  values = (16, 13, 23, 4, 34, 17)
  for (spec, pix) in it.combinations(values, 2):
      slow = _autocorr_priv.modecount_slow(spec,pix,35)
      fast = _autocorr_priv.modecount(spec,pix,35)
      assert np.all(slow == fast)

def testModecountFastStable():
  """Test against a pre-computed large value, to make sure we are getting the same answer."""
  pre_compute_slow = np.array([  2780032,  17127136,  46818968,  77370656, 123590832, 161636256,
       211220832, 252947240, 297074976, 327237040, 361562904, 383550416,
       397786512, 401343640, 399877208, 387034440, 363200264, 331814960,
       292657648, 238451600, 183780000, 131866488,  92345704,  60716952,
        38258760,  22054960,  11621688,   5485488,   2384512,    960648,
          331672,     92072,     16128,      1360,         8], dtype=np.int32)
  fast = _autocorr_priv.modecount(50,30,35)
  assert np.all(fast == pre_compute_slow)

def testOverflow():
    """
    Check that large values do not overflow the counters!
    """
    assert np.all( _autocorr_priv.modecount(500,550,5) > 0)


def testAutocorrAnalytic():
    """Correlation functions for a single spectrum with a few non-zero points."""
    slist = np.zeros((1,50),dtype=np.float64)
    slist[0,25] = 1
    auto = _autocorr_priv.crosscorr_spectra(slist, slist, 1, 30)
    assert(auto[0] == 1)
    assert(np.all(auto[1:] == 0))
    #Check that it doesn't matter which position this point is.
    slist[0,25] = 0
    slist[0,15] = 1
    auto = _autocorr_priv.crosscorr_spectra(slist, slist, 1, 30)
    assert(auto[0] == 1)
    assert(np.all(auto[1:] == 0))
    #Check two points
    slist[0,30] = 1
    auto = _autocorr_priv.crosscorr_spectra(slist, slist, 1, 30)
    assert(auto[0] == 2)
    assert(auto[5] == 2)
    assert(np.all(auto[1:5] == 0))
    assert(np.all(auto[6:] == 0))
    #Check for different inputs
    slist2 = np.zeros((1,50),dtype=np.float64)
    slist2[0,15] = 1
    auto = _autocorr_priv.crosscorr_spectra(slist, slist2, 1, 30)
    assert(auto[0] == 1)
    assert(auto[5] == 1)
    assert(np.all(auto[1:5] == 0))
    assert(np.all(auto[6:] == 0))

def testAutoCorrAgainstModecount():
    """Check that if autocorr is handed a constant array, it gives the same answer as modecount, ie,
    that it has the right limiting behaviour"""
    slist = np.ones((15*15, 8), dtype=np.float64)
    auto = _autocorr_priv.crosscorr_spectra(slist, slist, 15, 30)
    mode = _autocorr_priv.modecount(15, 8, 30)
    assert np.all(auto == mode)
    slist*=-2
    #Should now be four times mode
    auto = _autocorr_priv.crosscorr_spectra(slist, slist, 15, 30)
    assert np.all(auto == 4*mode)

def testAutoCorrList():
    """Check that the slow spectra computation gives the same answer as the list-based computation"""
    np.random.seed(23)
    values = (15, 14, 5, 18, 13, 23, 4, 34, 17)
    for (spec, pix) in it.combinations(values, 2):
        slist = np.random.randint(0,2,(spec*spec,pix))
        slist = slist.astype(np.float64)
        auto = _autocorr_priv.crosscorr_spectra(slist, slist, spec, 30)
        ind = np.where(slist == 1)
        auto2 = _autocorr_priv.autocorr_list(ind[0], ind[1], spec, pix,30)
        assert(np.all(auto == auto2))


def testCrossCorrList():
    """Check that the cross-correlation between a field and tracers gives the same answer as the list-based computation"""
    np.random.seed(23)
    values = (15, 14, 5, 18, 13, 23, 4, 34, 17)
    for (spec, pix) in it.combinations(values, 2):
        slist = np.random.randint(0,2,(spec*spec,pix))
        slist = slist.astype(np.float64)
        ind = np.where(slist == 1)
        auto = _autocorr_priv.crosscorr_list_spectra(slist, ind[0], ind[1], spec, 30)
        auto2 = _autocorr_priv.autocorr_list(ind[0], ind[1], spec, pix,30)
        assert(np.all(auto == auto2))
