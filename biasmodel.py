# -*- coding: utf-8 -*-
"""This module calculates the halo bias from the fitting formula of Tinker 2010 (http://arxiv.org/abs/1001.3162)"""

import halo_mass_function as hm
import numpy as np
import halofit
import math
from save_figure import save_figure
import boxhi
import myname

class HaloBias(hm.HaloMassFunction):
    """Module for calculating the halo bias.
    """
    def __init__(self, redshift, omega_m=0.27, omega_b=0.045, omega_l=0.73, hubble=0.7, ns=0.95, sigma8=0.8, log_mass_lim=(6, 20) ):
        hm.HaloMassFunction.__init__(self,redshift,omega_m, omega_b,omega_l,hubble, ns,sigma8,log_mass_lim=log_mass_lim)
        self.omega_m = omega_m
        self.omega_l = omega_l

        #log Overdensity at which a halo is defined
        self.yover = np.log10(200)
        #From table 2 of Tinker 2010.
        self.A = 1. + 0.24 * self.yover * np.exp(-(4/self.yover)**4)
        self.a = 0.44 * self.yover
        self.B = 0.183
        self.b = 1.5
        self.C = 0.019 + 0.107*self.yover + 0.19* np.exp(-(4/self.yover)**4)
        self.c = 2.4

    def halo_bias(self, mass):
        """
        Compute the halo bias as a function of mass
        """
        nu = self.get_nu(mass)
        return self._halo_bias(nu)

    def _halo_bias(self, nu):
        """
        Compute the halo bias as a function of mass
        """
        return 1. - self.A * 1./(1. + (self.delta_c0/nu)**self.a) + self.B * nu**self.b + self.C*nu**self.c

    def get_nu(self, mass):
        """Return nu"""
        sigma = self.overden.sigmaof_M_z(mass)
        return self.delta_c0 / sigma


class DLABias(HaloBias):
    """Module for computing DLA bias"""
    def __init__(self, redshift, beta = 1.00, sig10 = 2.19, top = 12.9, bottom = 8.5, omega_m=0.273, omega_b=0.045, omega_l=0.727, hubble=0.7, ns=0.95, sigma8=0.8, log_mass_lim=(6, 20) ):
        HaloBias.__init__(self,redshift,omega_m, omega_b,omega_l,hubble, ns,sigma8,log_mass_lim=log_mass_lim)
        #Mean value of beta from the simulations. At z=4,3,2: 1.02, 1.04, 1.00
        self.beta = beta
        #Only tested for z=[4-2]
        self.sig10 = sig10 + (redshift - 2) * 0.33
        self.top = top
        self.bottom = bottom

    def sigma_DLA(self, mass):
        """Return the DLA cross-section from Bird et al 2014"""
        sDLA =  self.sig10 * (mass / 1e10)**self.beta
#         ind = np.where(mass > 1e13)
#         sDLA[ind] = sDLA[ind][0]
        return sDLA

    def dla_bias(self):
        """DLA bias as the convolution of the DLA cross-section with the number of halos:
            integral ( b (M)  n(M) sigma(M) )  / integral( n(M) sigma(M) )"""
        #Do integrals in log space
        npts = 50
        mass = 10**np.linspace(self.bottom, self.top,npts)
        #First evaluate the bias integral
        samples = self.halo_bias(mass) * self._integrand(mass)
        #Simple trapezium rule
        summ = (samples[0] + samples[-1] + 2*np.sum(samples[1:-1]))
        #Then the total number of halos
        hsamples = self._integrand(mass)
        hsum = (hsamples[0] + hsamples[-1] + 2*np.sum(hsamples[1:-1]))
        return summ / hsum

    def _integrand(self, mass):
        """Integrand for above with bias"""
        return mass * self.dndm(mass) * self.sigma_DLA(mass)

class NonLinearDLABias(DLABias):
    """Compute the DLA bias using a non-linear matter power spectrum"""
    def dla_bias(self):
        """DLA bias as the convolution of the DLA cross-section with the number of halos:
            integral ( b (M)  n(M) sigma(M) )  / integral( n(M) sigma(M) )"""
        (k, Delta, zz) = halofit.load_pkfile("/home/spb/data/Cosmo/ICs/codes/camb/spb_rescale/illustris_matterpower_2.3.dat")
        dla_biask = np.zeros_like(k)
        powerk = np.zeros_like(k)
        #Do integrals in log space
        npts = 50
        for mass in 10**np.linspace(self.bottom, self.top, npts):
            mbias = self.halo_bias(mass)
            hfit = halofit.HaloFit(k, mbias**2*Delta, zz, self.omega_m, self.omega_l)
            hfit_nobias = halofit.HaloFit(k, Delta, zz, self.omega_m, self.omega_l)
            tsig = (mass * self.dndm(mass) * self.sigma_DLA(mass))**2
            if mass == self.bottom or mass == self.top:
                dla_biask += tsig * hfit.do_nonlin()
                powerk += tsig * hfit_nobias.do_nonlin()
            else:
                dla_biask += 2*tsig * hfit.do_nonlin()
                powerk += 2*tsig * hfit_nobias.do_nonlin()
        return (k, np.sqrt(dla_biask / powerk))

def sigma_DLA_pontzen(mass, hubble = 0.7):
    """Return the DLA cross-section from Pontzen et al 2008"""
    #Correct physical kpc to comoving kpc/h for comparison with Bird 2014 sigma_DLA.
    s0 = 40*hubble**2/4.**2
    m0 = 10**(9.5)
    alpha = 0.5
    return s0*(mass / m0)**2 * (1+ mass/m0)**(alpha-2)

def nonlinearlyabias(bias, omega_m = 0.27, omega_l = 0.73):
    """Get the bias for the Lyman alpha forest as a non-linear function of k,
    assuming the lyman alpha forest is a delta function of scale"""
    (k, Delta, zz) = halofit.load_pkfile("/home/spb/data/Cosmo/ICs/codes/camb/spb_rescale/illustris_matterpower_2.3.dat")
    hfit = halofit.HaloFit(k, bias**2*Delta, zz, omega_m, omega_l)
#     hfit_nobias = halofit.HaloFit(k, Delta, zz, omega_m, omega_l)
    dla_biask = hfit.do_nonlin()
    #Use the linear theory power because that is what is used in FR 12
    powerk = Delta #hfit_nobias.do_nonlin()
    return (k, np.sqrt(dla_biask / powerk))

def plot_dla_halo_bias(sim, snap, Mmin=8, Mmax=13, nbins=40):
    """Plot a nonlinear bias model from histogram of the halo masses of DLA hosts. Each bin contains the fraction
       of DLA cells associated with halos in this mass bin"""
    bbb = NonLinearHaloBias(2.3, top=13, bottom=8)
    bbb.dla_bias()
    print "DLA bias is:", dla_bias
    return (kk, dla_biask, powerk, dla_bias)

def plot_dla_halo_bias(sim, snap, Mmin=8, Mmax=13, nbins=40, minpart = 0, dist=2.):
    """Plot a nonlinear bias model from histogram of the halo masses of DLA hosts. Each bin contains the fraction
       of DLA cells associated with halos in this mass bin"""
    hspec = boxhi.BoxHI(myname.get_name(sim), snap, nslice=10)
    hspec._get_sigma_DLA(minpart, dist)
    ind = np.where(hspec.sigDLA > 0)
    sigs = hspec.sigDLA[ind]/1e6
    (kk, Delta, zz) = halofit.load_pkfile("/home/spb/data/Cosmo/ICs/codes/camb/spb_rescale/illustris_matterpower_2.3.dat")
    bbb = HaloBias(zz, 0.273, 0.045, 0.727, 0.71, 0.97)
    ii = np.where(hspec.real_sub_mass[ind] > 1e8)
    biases = bbb.halo_bias(hspec.real_sub_mass[ind][ii])
    dla_bias = np.sum(sigs[ii]*biases)/ np.sum(sigs[ii])
    mbins = np.logspace(Mmin,Mmax,nbins)
    dla_biask = np.zeros_like(kk)
    hfit = halofit.HaloFit(kk, Delta, zz, 0.27, 0.73)
    powerk = np.sum(sigs[ii])*np.sqrt(hfit.do_nonlin())
    for mm in np.arange(np.size(mbins)-1):
        mbias = bbb.halo_bias(np.sqrt(mbins[mm]*mbins[mm+1]))
        hfit = halofit.HaloFit(kk, mbias**2*Delta, zz, 0.27, 0.73)
        ii = np.where(np.logical_and(hspec.real_sub_mass[ind] > mbins[mm], hspec.real_sub_mass[ind] < mbins[mm+1]))
        tsig = np.sum(sigs[ii])
        dla_biask += np.sqrt(hfit.do_nonlin())*tsig
    print "DLA bias is:", dla_bias
    return (kk, dla_biask, powerk, dla_bias)

def plot_dla_bias_z23(sim, Mmin=8, Mmax=13, nbins=40, minpart = 0, dist=2., interp=4, interpz=2.5):
    """Interpolate as a function of redshift"""
    (kk, dlabiask, powerk, dla_bias) = plot_dla_halo_bias(sim, interp)
    (kk, dlabiask, powerk, dla_bias) = plot_dla_halo_bias(sim, 5)
    a1 = 1./(1+2.5)
    a2 = 1./(1+2)
    a = 1./(1+2.3)
    dlabias_halo = 10**(np.log10(dlabiask)*(a2-a)/(a2-a1) + np.log10(dlabiask)*(a-a1)/(a2-a1))
    power_halo = 10**(np.log10(powerk)*(a2-a)/(a2-a1) + np.log10(powerk)*(a-a1)/(a2-a1))
    dla_bias = 10**(np.log10(dla_bias)*(a2-a)/(a2-a1) + np.log10(dla_bias)*(a-a1)/(a2-a1))
    return (kk, dlabias_halo/power_halo, dla_bias)


if __name__ == "__main__":

    from plot_bias import plot_bias,bias_data_avg,bias_data_scale
    import matplotlib.pyplot as plt
    #Measured b_F ( 1+ beta_F ) = -0.336
#     bF = -0.336/2
#     (k, lyabias) = nonlinearlyabias(bF)
    (kk,dlabias_halo, dla_bias) = plot_dla_bias_z23(7)
    plt.plot(kk,dlabias_halo, color="red", ls="--", label="Halofit")
    base="/home/spb/data/Cosmo/Cosmo7_V6/L25n512/output/snapdir_005/DLA_autocorr_snap_005"
    plot_bias(base, "blue", ls="-", label="2xUV")
#     (k,dlabias_halo, dla_bias) = plot_dla_bias_z23(1,interp=3, interpz=3.)
#     plt.plot(kk,dlabias_halo, color="purple", ls="-.", label="Hf-HVEL")
#     base="/home/spb/data/Cosmo/Cosmo1_V6/L25n512/output/snapdir_005/DLA_autocorr_snap_005"
#     plot_bias(base, "purple", ls="-", label="HVEL")
#     plt.plot(k, -dlabias_halo*lyabias/bF, color="green", ls="-.", label="Forest")
#     bb = NonLinearDLABias(2.3)
#     (k, dlabias) = bb.dla_bias()
#     plt.plot(k, dlabias, color="orange", ls="--")
    bias_data_scale()
    plt.legend(loc=1, ncol=1)
#     bias_data_avg()
    plt.xlim(0.06, 2.5)
    plt.ylim(1,4.5)
    plt.xticks([0.2, 0.5, 1.0, 2.0],["0.2","0.5","1.0","2.0"])
    save_figure("DLA_bias_nonlin")
    plt.clf()
