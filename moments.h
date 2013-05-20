#ifndef _MOMENTS_H
#define _MOMENTS_H

#include <map>

#define XH 0.76
//Internal gadget mass unit: 1e10 M_sun/h in g/h
#define UnitMass_in_g 1.989e43
//Internal gadget length unit: 1 kpc/h in cm/h
#define UnitLength_in_cm 3.085678e21
//Internal velocity unit : 1 km/s in cm/s
#define UnitVelocity_in_cm_per_s 1e5
//proton mass in g
#define protonmass 1.66053886e-24


void multiply_by_tophat(double * field, int size, double thresh);
double find_total(double * field, int size);
std::map<double, int> pdf(double * field, int size, double xmin, double xmax, double xstep);

int load_hdf5_header(const char *ffname, double  *atime, double *redshift, double * Hz, double *box100, double *h100);
int load_hdf5_snapshot(const char *ffname, double *omegab, int fileno, double h100, double redshift, float *Pos, float * Mass, float * h);
#endif
