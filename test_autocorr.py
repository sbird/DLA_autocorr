import _autocorr_priv
import numpy as np

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