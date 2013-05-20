#include <hdf5.h>
#include <hdf5_hl.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "moments.h"

#ifndef N_TYPE
        #define N_TYPE 6
        #define PARTTYPE 0
#endif

const double gamma_UVB[6] = {3.99e-14, 3.03e-13, 6e-13, 5.53e-13, 4.31e-13, 3.52e-13};
//Photoion rate if T = 1e4K.
double rah_photo_rate(double nH, int redshift)
{
    double nSSh = 0.0050004114759671396;
    double photUVBratio= 0.98*pow(1+nH/pow(nSSh,1.64),-2.28)+0.02*pow(1+nH/nSSh,-0.84);
    int zz = redshift > 6 ? 6 : redshift;
    return photUVBratio * gamma_UVB[zz];
}

//Find the Rahmati neutral fraction at 10^4 K
double rah_neut_frac(double dens, int redshift)
{
    double alpha_A = 4.2969536156103855e-13;
    double lambda_T = 1.2453660557199207e-15;
    double A = alpha_A + lambda_T;
    double B = 2*alpha_A + rah_photo_rate(dens, redshift)/dens + lambda_T;
    return (B - sqrt(B*B-4*A*alpha_A))/(2*A);
}

/*Routine that is a wrapper around HDF5's dataset access routines to do error checking. Returns the length on success, 0 on failure.*/
hsize_t get_single_dataset(const char *name, float * data_ptr,  hsize_t data_length, hid_t * hdf_group,int fileno){
          int rank;
          hsize_t vlength;
          size_t type_size;
          H5T_class_t class_id;
          if (H5LTget_dataset_ndims(*hdf_group, name, &rank) < 0 || rank != 1){
             fprintf(stderr, "File %d: Rank of %s is %d !=1\n",fileno,name, rank);
             return 0;
          }
          H5LTget_dataset_info(*hdf_group, name, &vlength, &class_id, &type_size);
          if(type_size != 4 || class_id != H5T_FLOAT  || vlength > data_length || H5LTread_dataset_float(*hdf_group, name, data_ptr) < 0 ){
              fprintf(stderr, "File %d: Failed reading %s (%lu)\n",fileno,name, (uint64_t)vlength);
              return 0;
          }
          return vlength;
}

/*A similar wrapper around HDF5's dataset access routines to do error checking. Returns the length on success, 0 on failure.*/
hsize_t get_triple_dataset(const char *name, float * data_ptr, hsize_t data_length, hid_t * hdf_group,int fileno){
          int rank;
          hsize_t vlength[2];
          size_t type_size;
          H5T_class_t class_id;
          if (H5LTget_dataset_ndims(*hdf_group, name, &rank) < 0 || rank != 2){
             fprintf(stderr, "File %d: Rank of %s is %d !=2\n",fileno,name, rank);
             return 0;
          }
          H5LTget_dataset_info(*hdf_group, name, &vlength[0], &class_id, &type_size);
          if(type_size != 4 || class_id != H5T_FLOAT || vlength[1] != 3 || vlength[0] > data_length || H5LTread_dataset_float(*hdf_group, name, data_ptr) < 0 ){
              fprintf(stderr, "File %d: Failed reading %s (%lu)\n",fileno,name, (uint64_t)vlength[0]);
              return 0;
          }
          return vlength[0];
}

/* this routine loads header data from the first file of an HDF5 snapshot.*/
int load_hdf5_header(const char *ffname, double  *atime, double *redshift, double * Hz, double *box100, double *h100, double *Omega0)
{
  int i;
  int npart[N_TYPE];
  double mass[N_TYPE];
  int flag_cooling;
  int64_t npart_all[N_TYPE];
  double OmegaLambda;
  hid_t hdf_group,hdf_file;
  hdf_file=H5Fopen(ffname,H5F_ACC_RDONLY,H5P_DEFAULT);
  if(hdf_file < 0){
        return -1;
  }
  if ( (hdf_group=H5Gopen(hdf_file,"/Header",H5P_DEFAULT)) < 0) {
        H5Fclose(hdf_file);
        return -1;
  }
  /* Read some header functions */
  
  if(H5LTget_attribute_double(hdf_group,".","Time",atime) ||
     H5LTget_attribute_double(hdf_group,".","Redshift", redshift) ||
     H5LTget_attribute_double(hdf_group,".","BoxSize", box100) ||
     H5LTget_attribute_double(hdf_group,".","HubbleParam", h100) ||
     H5LTget_attribute_double(hdf_group,".","Omega0", Omega0) ||
     H5LTget_attribute_double(hdf_group,".","OmegaLambda", &OmegaLambda) ||
     H5LTget_attribute_int(hdf_group,".","Flag_Cooling",&flag_cooling)){
          fprintf(stderr,"Failed to read some header value\n");
      H5Gclose(hdf_group);
      H5Fclose(hdf_file);
      return -1;
  }
  (*Hz)=100.0*(*h100) * sqrt(1.+*Omega0*(1./(*atime)-1.)+OmegaLambda*((pow(*atime,2)) -1.))/(*atime);
  /*Get the total number of particles*/
  H5LTget_attribute(hdf_group,".","NumPart_Total",H5T_NATIVE_INT, &npart);
  for(i = 0; i< N_TYPE; i++)
          npart_all[i]=npart[i];
  H5LTget_attribute(hdf_group,".","NumPart_Total_HighWord",H5T_NATIVE_INT, &npart);
  for(i = 0; i< N_TYPE; i++)
          npart_all[i]+=(1L<<32)*npart[i];
  H5LTget_attribute(hdf_group,".","MassTable",H5T_NATIVE_DOUBLE, mass);
  
  /*Close header*/
  H5Gclose(hdf_group);
  H5Fclose(hdf_file);
  
  if(npart_all[PARTTYPE] <=0)
          return -1;
  printf("NumPart=[%ld,%ld,%ld,%ld,%ld,%ld], ",npart_all[0],npart_all[1],npart_all[2],npart_all[3],npart_all[4],npart_all[5]);
  printf("Masses=[%g %g %g %g %g %g], ",mass[0],mass[1],mass[2],mass[3],mass[4],mass[5]);
  printf("Redshift=%g, Ω_M=%g Ω_L=%g\n",(*redshift),*Omega0,OmegaLambda);
  printf("Expansion factor = %f\n",(*atime));
  printf("Hubble = %g Box=%g \n",(*h100),(*box100));
  return 0;
}
  
/* This routine loads particle data from a single HDF5 snapshot file.
 * A snapshot may be distributed into multiple files. */
int load_hdf5_snapshot(const char *ffname, double *omegab, int fileno, double h100, double redshift, float *Pos, float * Mass, float * h)
{
  size_t i;
  int npart[N_TYPE];
  double mass[N_TYPE];
  char name[16];
  float *fraction, *density;
  double Omega0;
  hid_t hdf_group,hdf_file;
  hsize_t length;
  hdf_file=H5Fopen(ffname,H5F_ACC_RDONLY,H5P_DEFAULT);
  if(hdf_file < 0){
        return -1;
  }
  if ( (hdf_group=H5Gopen(hdf_file,"/Header",H5P_DEFAULT)) < 0) {
        H5Fclose(hdf_file);
        return -1;
  }
  if( H5LTget_attribute(hdf_group,".","NumPart_ThisFile",H5T_NATIVE_INT, &npart) ||
      H5LTget_attribute_double(hdf_group,".","Omega0", &Omega0) ||
      H5LTget_attribute(hdf_group,".","MassTable",H5T_NATIVE_DOUBLE, mass)) {
      fprintf(stderr,"Failed to read some header value\n");
      H5Gclose(hdf_group);
      H5Fclose(hdf_file);
      return -1;
  }
  const int np = npart[PARTTYPE];
  int success = ((Pos=(float *)malloc(np*3*sizeof(float))) &&
    (Mass=(float *) malloc(np*sizeof(float))) &&
   (fraction=(float *)malloc(np*sizeof(float))) &&
   (density=(float *)malloc(np*sizeof(float))) &&
  ((h=(float *)malloc(np*sizeof(float)))));
  if(!success) {
    fprintf(stderr,"Failed to allocate memory.\n\n");
    exit(1);
  }
  H5Gclose(hdf_group);
  /*Open particle data*/
  snprintf(name,16,"/PartType%d",PARTTYPE);

  if ( (hdf_group=H5Gopen(hdf_file,name,H5P_DEFAULT)) < 0) {
        H5Fclose(hdf_file);
        return -1;
  }

  /* Read position and velocity*/
  length = get_triple_dataset("Coordinates",Pos,npart[PARTTYPE],&hdf_group,fileno);
  if(length == 0)
          goto exit;
  printf("Reading File %d (%lu particles)\n", fileno,(uint64_t)length);

  /* Particle masses  */
  if(mass[PARTTYPE])
        for(i=0; i< length;i++)
           Mass[i] = mass[PARTTYPE];
  else
     if (length != get_single_dataset("Masses",Mass,length,&hdf_group,fileno))
             goto exit;
  (*omegab) = Mass[0]/(Mass[0]+mass[1])*Omega0;
  /* The HI fraction, nHI/nH */
  if (length != get_single_dataset("NeutralHydrogenAbundance",fraction,length,&hdf_group,fileno))
     goto exit;
  if (length != get_single_dataset("Density",density,length,&hdf_group,fileno))
     goto exit;
  /*Get HI mass: 0.76 is the hydrogen mass fraction, which could be configurable*/
  for(i = 0; i< length; i++){
      /*Above the star formation density use the Rahmati fitting formula directly,
        as Arepo reports values for the eEOS.*/
      double dens = density[i]*UnitMass_in_g/pow(UnitLength_in_cm,3)/protonmass*h100*h100;
      dens*=XH*pow(1+redshift,3);
      if(dens > 0.1)
          fraction[i] = rah_neut_frac(dens, redshift);
      Mass[i]*=fraction[i]*XH;
  }
  /*Are we arepo? If we are we should have this array.*/
  if ( H5LTfind_dataset(hdf_group, "Volume")){
      /*Read in density*/
      if (length != get_single_dataset("Volume",h,length,&hdf_group,fileno))
              goto exit;
      /*Find cell length from volume, assuming a sphere.
       * Note that 4 pi/3**1/3 ~ 1.4, so the geometric 
       * factors nearly cancel and the cell is almost a cube.*/
      for(i=0;i<length;i++)
              h[i] = pow(3*h[i]/4/M_PI,0.33333333);
   }
  else{
      /* The smoothing length for gadget*/
      if (length != get_single_dataset("SmoothingLength",h,length,&hdf_group,fileno))
          goto exit;
      }
exit:
  H5Gclose(hdf_group);
  H5Fclose(hdf_file);
  if(fileno < 1){
        printf("\nP[%d].Pos = [%g %g %g]\n", 0, Pos[0], Pos[1],Pos[2]);
        printf("P[-1].Mass = %e\n", Mass[0]);
        printf("P[-1].NH0 = %e\n", fraction[length-1]);
        printf("P[-1].h = %f\n", h[length-1]);
  }
  free(fraction);
  free(density);
  return length;
}
