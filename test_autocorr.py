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
    fast = _autocorr_priv.modecount(500,300,35)
    assert np.all(fast > 0)
