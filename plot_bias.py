#!/usr/bin env python
# -*- coding: utf-8 -*-
"""Make some plots of the DLA bias"""

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt

import bias
import numpy as np
import re
import math
import os.path as path
from biasmodel import plot_dla_bias_z23,NonLinearDLABias,DLABias
from save_figure import save_figure

#The 5Mpc cut-off used in Font-Ribera 2012
k_cut = 0.25*2*math.pi

#Colors and linestyles for the simulations
colors = {0:"red", 1:"purple", 2:"cyan", 3:"green", 4:"greenyellow", 5:"pink", 7:"blue", 6:"grey", 9:"orange"}
lss = {0:"--",1:":", 2:":",3:"-.", 4:"--", 5:"--",6:"--",7:"-",9:"-"}
labels = {0:"DEF",1:"HVEL", 2:"HVNOAGN",3:"NOSN", 4:"WMNOAGN", 5:"MVEL",6:"METAL",7:"2xUV", 9:"FAST"}

def interp_power(zz, z1, z2, datafile, other):
    """Load two snapshots and interpolate between them linearly a as log P(k)"""
    otherstring = re.sub("005", other, datafile)
    dd_other = bias.AutoCorr(otherstring)
    dd = bias.AutoCorr(datafile)
    a1 = 1./(1+z1)
    a2 = 1./(1+z2)
    a = 1./(1+zz)
    dd.power = 10**(np.log10(dd.power)*(a2-a)/(a2-a1) + np.log10(dd_other.power)*(a-a1)/(a2-a1))
    dd.redshift = zz
    return dd

def plot_bias(datafile,color="blue", ls="-", label="", interp = "004"):
    """Plot the bias from a snapshot"""
    dd = interp_power(2.3, 2.5, 2, datafile, interp)
    total = re.sub("DLA_", "total_",datafile)
    dd_total = interp_power(2.3, 2.5, 2, total, interp)
    plt.semilogx(dd.keff, np.sqrt(dd.power/dd_total.power),color=color,ls=ls, label=label)
    plt.ylim(1,4)
    return (dd.keff, np.sqrt(dd.power/dd_total.power))

def plot_bias_no_interp(datafile,color="blue", ls="-", label=""):
    """Plot the bias from a snapshot"""
    dd = bias.AutoCorr(datafile)
    total = re.sub("DLA_", "total_",datafile)
    dd_total = bias.AutoCorr(total)
    plt.semilogx(dd.keff, np.sqrt(dd.power/dd_total.power),color=color,ls=ls, label=label)
    plt.ylim(1,4)

def bias_data_scale():
    """Plot the data from Font-Ribera et al 2012"""
    data = np.loadtxt("dla_bias_r.txt")
    #Col 1 is r in Mpc, Col 2 is the error
    #Col 3 is the bias, col 4 the error on the bias
    kk = 2*math.pi/data[:,0]
    kerr = 1./(data[:,0]+data[:,1])-1./data[:,0]
    kerr*=2*math.pi
    plt.errorbar(kk,data[:,2],xerr=kerr, yerr=data[:,3],fmt="s",color="black",ms=10)

def bias_data_avg():
    """Plot the average scale-independent bias"""
    bval = 2.17
    berr = 0.2
#     plt.fill_between([0.01,100],bval-berr, bval+berr,color="lightgrey")
#     bval = 2.10
#     berr = 0.13
    plt.fill_between([0.01,100],bval-berr, bval+berr,color="grey")
#     plt.plot([0.01,100],[bval, bval],"-", color="black")


def get_avg_bias(datafile):
    """Plot the bias from a snapshot"""
    dd = bias.AutoCorr(datafile)
    total = re.sub("DLA_", "total_",datafile)
    dd_total = bias.AutoCorr(total)
    bb = np.sqrt(dd.power/dd_total.power)
    return np.mean(bb[np.where(dd.keff < k_cut)])

def plot_sim(sim, snaps, zzz, color="blue"):
    """Plot different redshifts for a simulation"""
    bb=[]
    for ss in snaps:
        sp = "snapdir_"+str(ss).rjust(3,'0')
        sdd = path.join(sim, sp)
        datafile = path.join(sdd,"DLA_autocorr_snap_"+str(ss).rjust(3,'0'))
        bb.append(get_avg_bias(datafile))
    plt.plot(zzz, bb,color=color)


def plot_data():
    """Plot the average determined bias from Font-Ribera 2012"""
    zz=np.linspace(2,2.5,100)
    bb = np.ones_like(zz)
    ax=plt.gca()
    ax.fill_between(zz, 1.97, 2.23,color='grey')
    plt.plot(zz, 2.10*bb,color='black')
    plt.plot(zz, 1.97*bb,color='black',ls="--")
    plt.plot(zz, 2.23*bb,color='black',ls="--")
    plt.plot(zz, 2.37*bb,color='black',ls="--")

def plot_all_sims():
    """Plot all the sims"""
    for ii in xrange(6):
        base="/home/spb/data/Cosmo/Cosmo"+str(ii)+"_V6/L25n512/output/"
        plot_sim(base,[1,3,5],[4,3,2], colors[ii])
    plot_data()

def plot_all_sims_scale_z2():
    """Plot all the sims"""
    for ii in (0,1,7,9):
        base="/home/spb/data/Cosmo/Cosmo"+str(ii)+"_V6/L25n512/output/snapdir_005/DLA_autocorr_snap_005"
        if ii == 7:
            (keff, dbias) = plot_bias(base, colors[ii], ls=lss[ii], label=labels[ii])
        else:
            plot_bias(base, colors[ii], ls=lss[ii], label=labels[ii])

    #3 gets special treatment as no z=2.5 snapshot
    base="/home/spb/data/Cosmo/Cosmo3_V6/L25n512/output/snapdir_005/DLA_autocorr_snap_005"
    plot_bias(base, colors[3], ls=lss[3], label=labels[3], interp="003")
    return (keff, dbias)

if __name__ == "__main__":
    bias_data_avg()
    (keff, dbias) = plot_all_sims_scale_z2()
    (kk,dlabias_halo, dla_bias) = plot_dla_bias_z23(7)
#     bbb = NonLinearDLABias(2.3, top=13, bottom=np.log10(3e8))
#     (kk, dlabias_halo) = bbb.dla_bias()
    plt.plot(kk,dlabias_halo, color="blue", ls=":", label="Halofit")
    bbb = DLABias(2.3, top=13, bottom=np.log10(3e8))
    print "Linear DLA bias",bbb.dla_bias()
    ind = np.where(np.logical_and(kk > 2*math.pi/100, kk < keff[0]))
    ind2 = np.where(np.logical_and(kk > keff[0], kk < k_cut))
    i2 = np.where(keff < k_cut)
    print "Extrapolated DLA bias for 2xUV", (np.mean(dlabias_halo[ind])*np.size(ind)+np.mean( dbias[i2])*np.size(ind2))/(np.size(ind)+np.size(ind2))
    plt.legend(loc=1, ncol=2)
    bias_data_scale()
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{b}_\mathrm{D}$")
    plt.ylim(1.0,5)
    plt.xlim(0.0715, 2)
#     plt.xticks([0.4, 0.6,0.8, 1.0, 1.3],["0.4","0.6","0.8","1.0","1.3"])
    plt.xticks([0.1, 0.2, 0.5, 1.0, 2.0],["0.1","0.2","0.5","1.0","2.0"])
    save_figure("DLA_bias_z2")
    plt.clf()
