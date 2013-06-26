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
    base="/home/spb/data/Cosmo/Cosmo2_V6/L25n512/"
    base3="/home/spb/data/Cosmo/Cosmo3_V6/L25n512/"
    base0="/home/spb/scratch/Cosmo/Cosmo0_V6_512/"
    plot_sim(base,[1,3,5],[4,3,2])
    plot_sim(base3,[1,2,3],[4,3,2],"red")
    plot_sim(base0,[54,60,68],[4,3,2],"green")
    plot_data()

def plot_all_sims_scale_z2():
    """Plot all the sims"""
    base="/home/spb/data/Cosmo/Cosmo2_V6/L25n512/snapdir_005/DLA_autocorr_snap_005"
    base3="/home/spb/data/Cosmo/Cosmo3_V6/L25n512/snapdir_003/DLA_autocorr_snap_003"
    base0="/home/spb/scratch/Cosmo/Cosmo0_V6_512/snapdir_068/DLA_autocorr_snap_068"
    plot_bias(base)
    plot_bias(base3,"red",ls="--")
    plot_bias(base0,"green",ls="-.")
    bias_data_scale()
    plt.xlim(0.049, k_cut)

if __name__ == "__main__":
    plot_all_sims_scale_z2()
    save_figure("DLA_bias_z2")
