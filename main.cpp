/* Copyright (c) 2013 Simeon Bird <spb41@ias.edu>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. */

#include "moments.h"
#include <math.h>
#include <cassert>
//For getopt
#include <unistd.h>
//For omp_get_num_procs
#include <omp.h>
#include <stdlib.h>
#include <hdf5.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <fftw3.h>

void help()
{
//TODO: write this
}

std::string find_first_hdf_file(const std::string& infname)
{
  /*Switch off error handling so that we can check whether a
   * file is HDF5 */
  /* Save old error handler */
  hid_t error_stack=0;
  herr_t (*old_func)(hid_t, void*);
  void *old_client_data;
  H5Eget_auto(error_stack, &old_func, &old_client_data);
  /* Turn off error handling */
  H5Eset_auto(error_stack, NULL, NULL);
  std::string fname = infname;

  /*Were we handed an HDF5 file?*/
  if(H5Fis_hdf5(fname.c_str()) <= 0){
     /*If we weren't, were we handed an HDF5 file without the suffix?*/
     fname = infname+std::string(".0.hdf5");
     if (H5Fis_hdf5(fname.c_str()) <= 0)
        fname = std::string();
  }

  /* Restore previous error handler */
  H5Eset_auto(error_stack, old_func, old_client_data);
  return fname;
}

/*Open a file for reading to check it exists*/
int file_readable(const char * filename)
{
     FILE * file;
     if ((file = fopen(filename, "r"))){
          fclose(file);
          return 1;
     }
     return 0;
}
/** Maximal size of FFT grid. 
 * In practice 1024 means we need just over 4GB, as sizeof(float)=4*/
#define FIELD_DIMS 1024

using namespace std;

void convert_units(double * field, int size, const double redshift, const double box, const double h100)
{
        const double UnitMassing = 1.989e43;
        //Internal gadget length unit: 1 kpc/h in cm/h
        const double UnitLengthincm=3.085678e21;
        //in g
        const double proton = 1.66053886e-24;
        //From M_sun/h/(kpc/h)^3 to g/cm^3 (comoving)
        //1 kpc/h in physical cm
        const double rscale = UnitLengthincm/(1+redshift)/h100;
        // 10^10 M_sun in g
        const double mscale = UnitMassing/h100;
        double conv = mscale/pow(rscale,3);
        //To atoms / cm^3
        conv/=proton;
        //To atoms / cm^2
        const double column = box/FIELD_DIMS*UnitLengthincm*(1+redshift)/h100;
        conv*=column;
        for(int i=0; i< size; i++)
            field[i]*=conv;
}
/** \file 
 * File containing main() */

/** Main function. Accepts arguments, uses GadgetReader to open the snapshot, prints some header info, 
 * allocates FFT memory and creates a plan, calls read_fieldize and powerspectrum, prints the P(k) 
 * to a file, and then frees the memory*/
int main(int argc, char* argv[]){
  int nrbins;
  double *power, *keffs;
  double *field, *comp;
  int *count; 
  string indir(""),outdir("");
  char c;
  int64_t Npart;
  float * Pos, *Mass, *hsml;
  fftw_plan pl;
  fftw_complex *outfield;
  while((c = getopt(argc, argv, "i:o:h")) !=-1){
    switch(c){
        case 'o':
           outdir=static_cast<string>(optarg);
           break;
        case 'i':
           indir=static_cast<string>(optarg);
           break;
        case 'h':
        default:
           help();
           return 0;
      }
  }
  //Open the snapshot
  /*ffname is a copy of input filename for extension*/
  /*First open first file to get header properties*/
  std::string fname = find_first_hdf_file(indir);
  std::string ffname = fname;
  unsigned i_fileno=0;
  int fileno=0;
  if( fname.empty() || outdir.empty()){
          help();
          return 0;
  }

  H5Snap snap( fname.c_str() );
  /*See if we have been handed the first file of a set:
   * our method for dealing with this closely mirrors
   * HDF5s family mode, but we cannot use this, because
   * our files may not all be the same size.*/
  i_fileno = fname.find(".0.hdf5")+1;

  //Get the header and print out some useful things
  nrbins=floor(sqrt(3)*((FIELD_DIMS+1.0)/2.0)+1);
  const size_t size = 2*FIELD_DIMS*FIELD_DIMS*(FIELD_DIMS/2+1);
  //Memory for the field
  /* Allocating a bit more memory allows us to do in-place transforms.*/
  field=(double *)fftw_malloc(size*sizeof(double));
  //For the compensation array: extra mem not necessary but simplifies things
  comp=(double *)fftw_malloc(size*sizeof(double));
  if( !comp || !field ) {
  	fprintf(stderr,"Error allocating memory for field\n");
  	return 1;
  }
  //Initialise
  for(int i=0; i< 2*FIELD_DIMS*FIELD_DIMS*(FIELD_DIMS/2+1); i++)
      field[i] = comp[i] = 0;
  string filename=outdir;
  size_t last=indir.find_last_of("/\\");
  //Set up FFTW
  outfield=(fftw_complex *) &field[0];
  if(!fftw_init_threads()){
          std::cerr<<"Error initialising fftw threads\n";
  		  return 0;
  }
  int threads = std::min(omp_get_num_procs(),6);
  omp_set_num_threads(threads);
  fftw_plan_with_nthreads(threads);
  pl=fftw_plan_dft_r2c_3d(FIELD_DIMS,FIELD_DIMS,FIELD_DIMS,&field[0],outfield, FFTW_ESTIMATE);
  //Allocate memory for output
  power=(double *) malloc(nrbins*sizeof(double));
  count=(int *) malloc(nrbins*sizeof(int));
  keffs=(double *) malloc(nrbins*sizeof(double));
  if(!power || !count || !keffs){
  	fprintf(stderr,"Error allocating memory for power spectrum.\n");
        return 1;
  }

  /*Loop over files. Keep going until we run out, skipping over broken files.
   * The call to file_readable is an easy way to shut up HDF5's error message.*/
  while(1){
          /* P is allocated inside load_snapshot*/
          /*If we ran out of files, we're done*/
          if(!(file_readable(ffname.c_str()) && H5Fis_hdf5(ffname.c_str()) > 0))
                  break;
          Npart=snap.load_hdf5_snapshot(ffname.c_str(), fileno,&Pos, &Mass, &hsml);
          if(Npart > 0){
             /*Do the hard SPH interpolation*/
             if(SPH_interpolate(field, comp, FIELD_DIMS, Pos, hsml, Mass, NULL, snap.box100, Npart, 1))
                 exit(1);
             /*Free the particle list once we don't need it*/
             free(Pos);
             free(Mass);
             free(hsml);
          }
          fileno++;
          if(i_fileno && i_fileno != std::string::npos){
		    std::ostringstream convert;
		    convert<<fileno;
            ffname = fname.replace(i_fileno, 1, convert.str());
		  }
          else
           break;
  }
  fftw_free(comp);
  std::cout<< "Done interpolating"<<std::endl;
  //Find totals and pdf
  //BE CAREFUL WITH FFTW!
  convert_units(field, size, snap.redshift, snap.box100, snap.h100);
  double total_HI = find_total(field, size);
  multiply_by_tophat(field, size, pow(10, 20.3));
  double total_DLA = find_total(field, size);
  std::map<double, int> hist = pdf(field, size, 17, 23, 0.2);
  /*Now make a power spectrum*/
  discretize(field, size);
  powerspectrum(FIELD_DIMS,&pl,outfield,nrbins, power,count,keffs);
  filename=outdir;
  filename+="/DLA_autocorr_"+indir.substr(last+1);
  std::ofstream file;
  file.open(filename.c_str());
  if(!file.is_open()){
      std::cerr<<"Could not open "<<filename<<std::endl;
      exit(1);
  }
  file << indir <<std::endl;
  //z a h box H(z) Omega_0 Omega_b
  file << snap.redshift << " " << snap.atime <<" " << snap.h100 << " " << snap.box100 << " " << snap.omega0 << " " << snap.omegab << std::endl;
  file << "==" <<std::endl;
  file << total_HI << " " << total_DLA <<std::endl;
  file << "==" <<std::endl;
  for (std::map<double,int>::iterator it=hist.begin(); it!=hist.end(); ++it)
    file << it->first << " " << it->second << std::endl;
  file << "==" <<std::endl;
  for(int i=0;i<nrbins;i++)
  {
    if(count[i])
      file<<keffs[i]<<" "<<power[i]<<" "<<count[i]<<std::endl;
  }
  file.close();
  //Free memory
  free(power);
  free(count);
  free(keffs);
  fftw_free(field);
  fftw_destroy_plan(pl);
  return 0;
}

