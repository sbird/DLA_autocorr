#ifndef _MOMENTS_H
#define _MOMENTS_H

#include <map>
#include <fftw3.h>

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
void discretize(double * field, int size);
double find_total(double * field, int size);
std::map<double, int> pdf(double * field, int size, double xmin, double xmax, double xstep);

#ifndef N_TYPE
        #define N_TYPE 6
        #define PARTTYPE 0
#endif

class H5Snap
{
    public:
        H5Snap(const char *ffname);
        double atime;
        double redshift;
        double box100;
        double h100;
        double omega0;
        double omegab;
        double mass[N_TYPE];
        int load_hdf5_snapshot(const char *ffname, int fileno, float **Pos, float ** Mass, float ** h);
};

int SPH_interpolate(double * field, double * comp, const int nx, float *pos, float *radii, float *value, float *weights, const double box, const int nval, const int periodic);

#ifdef __cplusplus
extern "C" {
#endif
int powerspectrum(int dims, fftw_plan* pl,fftw_complex *outfield, int nrbins, double *power, int *count,double *keffs);
#ifdef __cplusplus
}
#endif
#endif
