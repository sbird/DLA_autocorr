#include <H5Cpp.h>
#include <cassert>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "moments.h"

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
hsize_t get_single_dataset(const char *name, float * data_ptr,  hsize_t data_length, H5::Group * hdf_group,int fileno){
    H5::DataSet dataset = hdf_group->openDataSet(name);
    if(dataset.getTypeClass() != H5T_NATIVE_FLOAT)
          return 0;
    H5::DataSpace dataspace = dataset.getSpace();
    int rank = dataspace.getSimpleExtentNdims();
    if (rank > 2){
       fprintf(stderr, "File %d: Rank of %s is %d !=2\n",fileno,name, rank);
       return 0;
    }
    hsize_t vlength[2];
    rank = dataspace.getSimpleExtentDims(&vlength[0]);
    if( (rank > 1 && vlength[1] != 3) || vlength[0] > data_length) {
        fprintf(stderr, "File %d: Bad array shape %s (%lu)\n",fileno,name, (uint64_t)vlength[0]);
        return 0;
    }
    dataset.read(data_ptr, H5T_NATIVE_FLOAT);
    return vlength[0];
}

/* this routine loads header data from the first file of an HDF5 snapshot.*/
H5Snap::H5Snap(const char *ffname)
{
  int npart[N_TYPE];
  int64_t npart_all[N_TYPE];
  H5::H5File hdf_file(ffname,H5F_ACC_RDONLY);
  H5::Group hdf_group(hdf_file.openGroup("/Header"));
  /* Read some header functions */
  hdf_group.openAttribute("Time").read(H5T_NATIVE_DOUBLE,&atime); 
  hdf_group.openAttribute("Redshift").read(H5T_NATIVE_DOUBLE,&redshift); 
  hdf_group.openAttribute("BoxSize").read(H5T_NATIVE_DOUBLE,&box100); 
  hdf_group.openAttribute("HubbleParam").read(H5T_NATIVE_DOUBLE,&h100); 
  hdf_group.openAttribute("Omega0").read(H5T_NATIVE_DOUBLE,&omega0); 
  /*Get the total number of particles*/
  hdf_group.openAttribute("NumPart_Total").read(H5T_NATIVE_INT,&npart); 
  for(int i = 0; i< N_TYPE; i++)
          npart_all[i]=npart[i];
  hdf_group.openAttribute("NumPart_Total_HighWord").read(H5T_NATIVE_INT,&npart); 
  for(int i = 0; i< N_TYPE; i++)
          npart_all[i]+=(1L<<32)*npart[i];
  hdf_group.openAttribute("MassTable").read(H5T_NATIVE_DOUBLE,&mass);
  
  omegab = mass[0]/(mass[0]+mass[1])*omega0;
  printf("NumPart=[%ld,%ld,%ld,%ld,%ld,%ld], ",npart_all[0],npart_all[1],npart_all[2],npart_all[3],npart_all[4],npart_all[5]);
  printf("Masses=[%g %g %g %g %g %g], ",mass[0],mass[1],mass[2],mass[3],mass[4],mass[5]);
  printf("Redshift=%g, Ω_M=%g\n",redshift,omega0);
  printf("Expansion factor = %f\n",atime);
  printf("Hubble = %g Box=%g \n",h100,box100);
}
  
/* This routine loads particle data from a single HDF5 snapshot file.
 * A snapshot may be distributed into multiple files. */
int H5Snap::load_hdf5_snapshot(const char *ffname, int fileno, float **Pos_out, float **Mass_out, float **h_out)
{
  size_t i;
  int npart[N_TYPE];
  char name[16];
  hsize_t length;
  H5::H5File hdf_file(ffname,H5F_ACC_RDONLY);

  {
    H5::Group hdf_group(hdf_file.openGroup("/Header"));
    hdf_group.openAttribute("NumPart_ThisFile").read(H5T_NATIVE_INT,&npart);
  }

  const int np = npart[PARTTYPE];
  float * Pos=(float *)malloc(np*3*sizeof(float));
  float * Mass=(float *) malloc(np*sizeof(float));
  float * h=(float *)malloc(np*sizeof(float));
  float * fraction=(float *)malloc(np*sizeof(float));
  float * density=(float *)malloc(np*sizeof(float));
  assert(Mass);
  assert(Pos);
  assert(fraction);
  assert(density);
  assert(h);
  /*Open particle data*/
  snprintf(name,16,"/PartType%d",PARTTYPE);

  H5::Group hdf_group(hdf_file.openGroup(name));
  /* Read position and velocity*/
  length = get_single_dataset("Coordinates",Pos,npart[PARTTYPE],&hdf_group,fileno);
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
  omegab = Mass[0]/(Mass[0]+mass[1])*omega0;
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
  try {  // to determine if the dataset exists in the group
      H5::DataSet( hdf_group.openDataSet( "Volume" ));
      if (length != get_single_dataset("Volume",h,length,&hdf_group,fileno))
          goto exit;
      /*Find cell length from volume, assuming a sphere.
       * Note that 4 pi/3**1/3 ~ 1.4, so the geometric 
       * factors nearly cancel and the cell is almost a cube.*/
      for(i=0;i<length;i++)
              h[i] = pow(3*h[i]/4/M_PI,0.33333333);
  }
  catch( H5::GroupIException not_found_error ) {
      /* The smoothing length for gadget*/
      if (length != get_single_dataset("SmoothingLength",h,length,&hdf_group,fileno))
          goto exit;
  }
  if(fileno < 1){
        printf("\nP[%d].Pos = [%g %g %g]\n", 0, Pos[0], Pos[1],Pos[2]);
        printf("P[-1].Mass = %e\n", Mass[0]);
        printf("P[-1].NH0 = %e\n", fraction[length-1]);
        printf("P[-1].h = %f\n", h[length-1]);
  }
exit:
  *Pos_out = Pos;
  *Mass_out = Mass;
  *h_out = h;
  free(fraction);
  free(density);
  return length;
}
