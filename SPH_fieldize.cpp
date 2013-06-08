#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdio.h>

#include "moments.h"

//Do allocate extra memory for FFTW
#define EXTRA 1L

/*Compute the SPH weighting for this cell.
 * rr is the smoothing length, r0 is the distance of the cell from the center*/
double compute_sph_cell_weight(double rr, double r0)
{
    if(r0 > rr){
        return 0;
    }
    double x = r0/rr;
    double geom = 8/M_PI/pow(rr,3);
    if(x > 0.5){
        return geom*2*pow(1-x,3);
    }
    else{
        return geom*(1-6*x*x+6*pow(x,3));
    }
}



#ifndef NO_KAHAN
/*Evaluate one iteration of Kahan Summation: sum is the current value of the field,
 *comp the compensation array, input the value to add this time.*/
inline void KahanSum(double* sum, double* comp, const double input, const int xoff,const int yoff, const int zoff, const long nx)
{
  const long off = (xoff*nx+yoff)*(2*(nx/2+EXTRA))+zoff;
  assert(off < nx*nx*(2*nx/2+1));
  const double yy = input - *(comp+off);
  const double temp = *(sum+off)+yy;     //Alas, sum is big, y small, so low-order digits of y are lost.
  *(comp+off) = (temp - *(sum+off)) -yy; //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
  *(sum+off) = temp;               //Algebraically, c should always be zero. Beware eagerly optimising compilers!
}

#else

inline void KahanSum(double* sum, double* comp, const double input, const int xoff,const int yoff, const int zoff, const long nx)
{
  const long off = (xoff*nx+yoff)*(2*(nx/2+EXTRA))+zoff;
  *(sum+off)+=input;
}
#endif

/**
 Do the hard work interpolating with an SPH kernel particles handed to us from python.
*/
int SPH_interpolate(double * field, double * comp, const int nx, float *pos, float *radii, float *value, float *weights, const double box, const long nval, const int periodic)
{
    assert(value);
    assert(pos);
    assert(radii);
    assert(field);
#ifndef NO_KAHAN
    assert(comp);
#endif
    //Convert to grid units
    const double conv = (nx-1)/box;
    for(int p=0;p<nval;p++){
        //Temp variables
        float pp[3];
        //Max size of kernel
        int upgx[3], lowgx[3];
        float rr= radii[p]*conv;
        float val= value[p];
        double weight = 1;
        if (weights != NULL){
            weight = weights[p];
        }
        for(int i=0; i<3; i++)
        {
            pp[i] = conv*pos[3*p+i];
            upgx[i] = floor(pp[i]+rr);
            lowgx[i] = floor(pp[i]-rr);
        }
        //Try to save some integrations if this particle is totally in this cell
        bool inside = true;
        for(int i=0; i<3; i++)
            inside *= (lowgx[i]==upgx[i] && lowgx[i] >= 0);

        if(inside){
                KahanSum(field, comp, val/weight, lowgx[0], lowgx[1], lowgx[2], nx);
                continue;
        }
        /*Array for storing cell weights*/
//         double sph_w[upgx[2]-lowgx[2]+1][upgx[1]-lowgx[1]+1][upgx[0]-lowgx[0]+1];
        double * sph_w = (double *) calloc((upgx[2]-lowgx[2]+1)*(upgx[1]-lowgx[1]+1)*(upgx[0]-lowgx[0]+1), sizeof(double));
        /*Total of cell weights*/
        double total=0;
        /* First compute the cell weights.
         * Subsample the cells if the smoothing length is O(1 cell).
         * This is more accurate, and also avoids edge cases where the particle can rest just between a cell.*/
        int nsub=2*((int)(3./rr))+1;
        double subs[nsub];
        /*Spread subsamples evenly across cell*/
        for(int i=0; i < nsub; i++)
            subs[i] = (i+1.)/(1.*nsub+1);
        #pragma omp parallel for reduction(+:total)
        for(int gz=lowgx[2];gz<=upgx[2];gz++)
            for(int gy=lowgx[1];gy<=upgx[1];gy++)
                for(int gx=lowgx[0];gx<=upgx[0];gx++){
                    double * cur_ptr = sph_w+(upgx[1]-lowgx[1]+1)*(upgx[0]-lowgx[0]+1)*(gz-lowgx[2])+(upgx[0]-lowgx[0]+1)*(gy-lowgx[1])+gx-lowgx[0];
                    for(int iz=0; iz< nsub; iz++)
                    for(int iy=0; iy< nsub; iy++)
                    for(int ix=0; ix< nsub; ix++){
                        double xx = gx-pp[0]+subs[ix];
                        double yy = gy-pp[1]+subs[iy];
                        double zz = gz-pp[2]+subs[iz];
                        double r0 = sqrt(xx*xx+yy*yy+zz*zz);
                        *cur_ptr+=compute_sph_cell_weight(rr,r0)/nsub/nsub;
                    }
                    total+=*cur_ptr;
                }
        if(total == 0){
           fprintf(stderr,"Massless particle detected! rr=%g gz=%d gy=%d gx=%d nsub = %d pp= %g %g \n",rr,upgx[2]-lowgx[2], upgx[1]-lowgx[1],upgx[0]-lowgx[0], nsub,-pp[0]+lowgx[0],-pp[1]+lowgx[1]);
        }
        //Deal with cells that have wrapped around the edges of the grid
        if(periodic){
            #pragma omp parallel for
            for(int gz=lowgx[2];gz<=upgx[2];gz++){
                //This deals with cells that have wrapped around the edges of the grid
                int gzm = (gz + (nx-1)) % (nx-1);
                for(int gy=lowgx[1];gy<=upgx[1];gy++){
                    int gym = (gy + (nx-1)) % (nx-1);
                    for(int gx=lowgx[0];gx<=upgx[0];gx++){
                        int gxm = (gx + (nx-1)) % (nx-1);
                        double * cur_ptr = sph_w+(upgx[1]-lowgx[1]+1)*(upgx[0]-lowgx[0]+1)*(gz-lowgx[2])+(upgx[0]-lowgx[0]+1)*(gy-lowgx[1])+gx-lowgx[0];
                        KahanSum(field, comp, *cur_ptr*val/total/weight,gxm,gym,gzm,nx);
                    }
                }
            }
        }
        else {
        /* Add cells to the array discarding parts that have wrapped around*/
            #pragma omp parallel for
            for(int gz=std::max(lowgx[2],0);gz<=std::min(upgx[2],nx-1);gz++)
                for(int gy=std::max(lowgx[1],0);gy<=std::min(upgx[1],nx-1);gy++)
                    for(int gx=std::max(lowgx[0],0);gx<=std::min(upgx[0],nx-1);gx++){
                        double * cur_ptr = sph_w+(upgx[1]-lowgx[1]+1)*(upgx[0]-lowgx[0]+1)*(gz-lowgx[2])+(upgx[0]-lowgx[0]+1)*(gy-lowgx[1])+gx-lowgx[0];
                        KahanSum(field, comp, *cur_ptr*val/total/weight,gx,gy,gz,nx);
                    }
        }
        free(sph_w);
    }
    return 0;
}
