#!/usr/bin env python
# -*- coding: utf-8 -*-
"""Make some plots of the DLA bias"""

import matplotlib
matplotlib.use('PDF')

import matplotlib.pyplot as plt

import bias
import numpy as np
import re
import os.path as path
from save_figure import save_figure

#The 5Mpc cut-off used in Font-Ribera 2012
k_cut = 0.2

colors = {0:"red", 1:"purple", 2:"blue", 3:"green", 4:"orange", 5:"cyan"}
lss = {0:"--",1:":", 2:"-",3:"-.", 4:"-", 5:"--"}

def plot_bias(datafile,color="blue", ls="-"):
    """Plot the bias from a snapshot"""
    dd = bias.AutoCorr(datafile)
    total = re.sub("DLA_", "total_",datafile)
    dd_total = bias.AutoCorr(total)
    plt.semilogx(dd.keff, np.sqrt(dd.power/dd_total.power),color=color,ls=ls)
    plt.ylim(1,4)

def bias_data_scale():
    """Plot the data from Font-Ribera et al 2012"""
    data = np.loadtxt("dla_bias_r.txt")
    #Col 1 is r in Mpc, Col 2 is the error
    #Col 3 is the bias, col 4 the error on the bias
    kk = 1./data[:,0]
    kerr = 1./(data[:,0]+data[:,1])-1./data[:,0]
    plt.errorbar(kk,data[:,2],xerr=kerr, yerr=data[:,3],fmt="s",color="black",ms=10)


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
    ax.fill_between(zz, 1.97, 2.37,color='grey')
    plt.plot(zz, 2.17*bb,color='black')
    plt.plot(zz, 1.97*bb,color='black',ls="--")
    plt.plot(zz, 2.37*bb,color='black',ls="--")

def plot_all_sims():
    """Plot all the sims"""
    for ii in xrange(6):
        base="/home/spb/data/Cosmo/Cosmo"+str(ii)+"_V6/L25n512/output/"
        plot_sim(base,[1,3,5],[4,3,2], colors[ii])
    plot_data()

def plot_all_sims_scale_z2():
    """Plot all the sims"""
    for ii in range(6):
        base="/home/spb/data/Cosmo/Cosmo"+str(ii)+"_V6/L25n512/output/snapdir_005/DLA_autocorr_snap_005"
        plot_bias(base, colors[ii], ls=lss[ii])
    bias_data_scale()
    plt.xlim(0.049, k_cut)

if __name__ == "__main__":
    plot_all_sims_scale_z2()
    plt.xlabel("k (h/Mpc)")
    plt.ylabel(r"$\mathrm{b}_\mathrm{D}$")
    plt.ylim(0.8,3.5)
    plt.xticks([0.05, 0.1,0.2],["0.05","0.1","0.2"])
    save_figure("DLA_bias_z2")
