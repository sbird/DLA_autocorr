# -*- coding: utf-8 -*-
"""Module to load the output of the DLA autocorrelator and plot it"""

import numpy as np
import math
from scipy.special import j0

def autofrompower_3d(k, pk,rr):
    """Cmpute the autocorrelation function a 3D dimensionful power spectrum, such as you might get from CAMB.
    From Challinor's structure notes:
        P(k) =  < δ δ*>
        Δ^2 = P(k) k^3/(2π^2)
        ζ(r) = int dk/k Δ^2 j_0(kr)
             = int dk (k^2) P(k) j_0(kr) / (2π^2)
        Arguments:
            k - k values
            pk - power spectrum
            r - values of r = | x-y |
                at which to evaluate the autocorrelation
    """
    auto = np.array([np.sum(pk*j0(k*r)*k**2/2/math.pi**2)/np.size(k) for r in rr])
    return auto


class AutoCorr:
    """Load an autocorrelation output by the moments C code"""
    def __init__(self, filename):
        self.UnitLength_in_cm=3.085678e21
        self.load_file(filename)

    def load_file(self,filename):
        """Load the output of ./moments"""
        mom = open(filename)
        #Read original filename
        self.snapname = mom.readline()
        #Read simulation data
        nums = mom.readline().split()
        self.redshift = float(nums[0])
        self.grdsize = int(nums[1])
        self.hub = float(nums[2])
        self.box = float(nums[3])
        self.omega0 = float(nums[4])
        self.omegab = float(nums[5])

        #Read total density
        mom.readline()
        inn = mom.readline().split()
        self.rho_HI = inn [0]
        if np.size(inn) > 1:
            self.rho_DLA = inn[1]
        inline = mom.readline()
        bins = []
        hist = []
        inline = mom.readline()

        #Read cddf
        while inline != "" and inline != "==\n":
            ll = inline.split()
            bins.append(float(ll[0]))
            hist.append(float(ll[1]))
            inline = mom.readline()

        (self.cddf_bins, self.cddf) = self.convert_to_cddf(np.array(bins), np.array(hist), self.grdsize**3)

        #Read power spectrum
        keff = []
        power = []
        count = []

        inline = mom.readline()
        while inline != "" and inline != "==":
            ll = inline.split()
            keff.append(float(ll[0]))
            power.append(float(ll[1]))
            count.append(float(ll[1]))
            inline = mom.readline()
        #First convert power spectrum to CAMB units
        #Box is in kpc/h, we want Mpc/h
        scale = 2*math.pi/(self.box/1000.)
        self.keff = np.array(keff)[1:]*scale
        self.power = np.array(power)[1:]/scale**3
        self.count = np.array(count)[1:]
        #From r=1 (where shot noise is likely to matter) to the box
        self.auto_bins = np.logspace(0,np.log10(self.box/1000.),10)
        self.auto = autofrompower_3d(self.keff, self.power, self.auto_bins)


    def convert_to_cddf(self,bins, hist,size):
        """Convert a histogram to the cddf"""
        width =  np.array([bins[i+1]-bins[i] for i in range(0,np.size(bins)-1)])
        center = np.array([(bins[i]+bins[i+1])/2. for i in range(0,np.size(bins)-1)])
        return ( center, hist[:-1] / ( size * width * self.absorption_distance() ) )

    def absorption_distance(self):
        """Compute X(z), the absorption distance per sightline (eq. 9 of Nagamine et al 2003)
        in dimensionless units."""
        #h * 100 km/s/Mpc in h/s
        h100=3.2407789e-18
        # in cm/s
        light=2.9979e10
        #Units: h/s   s/cm                        kpc/h      cm/kpc
        return h100/light*(1+self.redshift)**2*self.box/self.grdsize*self.UnitLength_in_cm
