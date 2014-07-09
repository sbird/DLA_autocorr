#include <H5Cpp.h>
#include "moments.h"
#include <map>
#include <valarray>
#include <cmath>
#include <string>
#include <sstream>


/** Functions for working with an interpolated field*/

/* Multiply all elements of size field by a top hat function H(x > thresh)*/
void multiply_by_tophat(FloatType * field, size_t size, FloatType thresh)
{
    #pragma omp parallel for
    for(size_t i=0; i< size; i++)
    {
        field[i] *= (field[i] > thresh);
    }
}

/* Make all positive array values 1*/
void discretize(FloatType * field, size_t size)
{
    #pragma omp parallel for
    for(size_t i=0; i< size; i++)
    {
        if(field[i] > 0)
            field[i] = 1;
    }
}


/* Normalise by the mean of the field, so that the field has mean 0,
 * sd. 1.
 * ie, calculate delta = rho / rho_bar - 1 */
void calc_delta(FloatType * field, size_t size, long realsize)
{
    const double mean = find_total(field, size) / realsize;
    #pragma omp parallel for
    for(size_t i=0; i< size; i++)
    {
        field[i] = field[i]/ mean -1;
    }
}

/* Find the total value of all elements of a field*/
double find_total(FloatType * field, size_t size)
{
    double total=0;
    #pragma omp parallel for reduction(+:total)
    for(size_t i=0; i< size; i++)
    {
        total += field[i];
    }
    return total;
}

/** Count the values of a field to find the pdf:
 * arguments: field - values to operate on
 * size: number of elements in the field
 * xmin: smallest histogram bin
 * xmax: largest histogram bin
 * nxbins: number of bins to cover the above range. Bins are log spaced.
 *  */
std::map<double, int> histogram(const FloatType * field, const size_t size, const double xmin, const double xmax, const int nxbins)
{
    std::map<double, int> hist;
    
    //Initialise the key values
    for (double i = log10(xmin); i < log10(xmax); i+=(log10(xmax)-log10(xmin))/nxbins)
        hist[pow(10.,i)] = 0;
    
    //Count values: note this means that the last entry will contain the 
    //no. of elements past the end
    for (size_t i = 0; i < size; ++i)
    {
        //Lower bound = first key equal to or greater than.
        std::map<double,int>::iterator it = hist.lower_bound(*(field+i));
        if (it != hist.begin()){
            --it;
            (it->second)++;
        }
    }

    return hist;

}

/** Convert a histogram to a pdf, adjusting the bins to be the bin centroids.
 */
std::map<double, double> pdf(std::map<double, int> hist, const size_t size)
{
    //Convert to a pdf
    //First, calculate bins
    std::map<double, double> pdf;
    for (std::map<double,int>::iterator it=hist.begin(); it!=hist.end()--;)
    {
        double start = it->first;
        int val = it->second;
        ++it;
        double finish = it->first;
        double width = finish - start;
        pdf[(start + finish)/2.] = (val)/(width*size);
    }

    return pdf;
}

//Test whether a particle with position (xcoord, ycoord, zcoord)
//is within the virial radius of halo j.
inline bool is_halo_close(const int j, const double xcoord, const double ycoord, const double zcoord, float * sub_cofm, float * sub_radii, const double box, const double grid)
{
            double xpos = fabs(sub_cofm[3*j] - xcoord);
            double ypos = fabs(sub_cofm[3*j+1] - ycoord);
            double zpos = fabs(sub_cofm[3*j+2] - zcoord);
            //Periodic wrapping
            if (xpos > box/2.)
                xpos = box-xpos;
            if (ypos > box/2.)
                ypos = box-ypos;
            if (zpos > box/2.)
                zpos = box-zpos;

            //Distance
            const float dd = xpos*xpos + ypos*ypos + zpos*zpos;
            //Is it close?
            const float rvir = sub_radii[j]*sub_radii[j];//+grid*grid/4.;
            //We will only be within the virial radius for one halo
            if (dd <= rvir) {
                return true;
            }
            else
                return false;
}

/*Routine that is a wrapper around HDF5's dataset access routines to do error checking. Returns the length on success, 0 on failure.*/
hsize_t get_single_int_dataset(const char *name, int * data_ptr,  hsize_t data_length, H5::Group * hdf_group,int fileno){
    H5::DataSet dataset = hdf_group->openDataSet(name);
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
    dataset.read(data_ptr, H5::PredType::NATIVE_INT);
    return vlength[0];
}

#define EXTRA 1L

FindHalo::FindHalo(std::string halo_file)
{
    int numfiles;
    {
        //Load the halo catalogues from a file.
        H5::H5File hdf_file(halo_file,H5F_ACC_RDONLY);
        //Read header
        {
            H5::Group hdf_group(hdf_file.openGroup("/Header"));
            hdf_group.openAttribute("Ngroups_Total").read(H5::PredType::NATIVE_INT,&nhalo);
            hdf_group.openAttribute("Nsubgroups_Total").read(H5::PredType::NATIVE_INT,&nsubhalo);
            hdf_group.openAttribute("NumFiles").read(H5::PredType::NATIVE_INT,&numfiles);
        }
        //Allocate memory
        sub_mass = (float *)malloc(nhalo*sizeof(float));
        sub_radii = (float *)malloc(nhalo*sizeof(float));
        sub_pos = (float *)malloc(nhalo*sizeof(float));
        subsub_radii = (float *)malloc(nsubhalo*sizeof(float));
        subsub_pos = (float *)malloc(nsubhalo*sizeof(float));
        subsub_index = (int *)malloc(nsubhalo*sizeof(int));
        if (subsub_index == NULL){
            fprintf(stderr, "Could not allocate halo memory\n");
            exit(1);
        }
    }
    int i_fileno = halo_file.find(".0.hdf5")+1;

    int cur_halo=0;
    int cur_subhalo = 0;
    for(int fileno = 0; fileno< numfiles; fileno++)
    {
        //Alter string for next file
        std::ostringstream convert;
        convert<<fileno;
        int replace = convert.str().length();
        if (fileno == 10)
            replace = 1;
        std::string this_file = halo_file.replace(i_fileno, replace, convert.str());
        //Load the halo catalogues from a file.
        H5::H5File hdf_file(this_file,H5F_ACC_RDONLY);
        int nhalo_file;
        int nsubhalo_file;
        //Read file header
        {
            H5::Group hdf_group(hdf_file.openGroup("/Header"));
            hdf_group.openAttribute("Ngroups_ThisFile").read(H5::PredType::NATIVE_INT,&nhalo_file);
            hdf_group.openAttribute("Nsubgroups_ThisFile").read(H5::PredType::NATIVE_INT,&nsubhalo_file);
        }
        //Read
        {
            H5::Group hdf_group(hdf_file.openGroup("/Group"));
            int length = get_single_dataset("GroupMass",sub_mass+cur_halo,nhalo_file,&hdf_group,fileno);
            length = get_single_dataset("GroupPos",sub_pos+cur_halo,3*nhalo_file,&hdf_group,fileno);
            length = get_single_dataset("Group_R_Crit200",sub_radii+cur_halo,nhalo_file,&hdf_group,fileno);
            if (length != nhalo_file){
              fprintf(stderr, "Error reading data from %d\n", fileno);
              exit(1);
            }
            H5::Group sub_group(hdf_file.openGroup("/Subhalo"));
            length = get_single_dataset("SubhaloPos",subsub_pos+cur_subhalo,3*nsubhalo_file,&sub_group,fileno);
            length = get_single_dataset("SubhaloHalfmassRad",subsub_radii+cur_subhalo,nsubhalo_file,&sub_group,fileno);
            length = get_single_int_dataset("SubhaloGrNr",subsub_index+cur_subhalo,nsubhalo_file,&sub_group,fileno);
            if (length != nsubhalo_file){
              fprintf(stderr, "Error reading subhalo data from %d\n", fileno);
              exit(1);
            }
        }
        cur_halo += nhalo_file;
        cur_subhalo += nsubhalo_file;
    }
}

/** Find a histogram of the number of DLAs around each halo
*/
std::valarray<int> FindHalo::get_halos(const FloatType * field, const size_t field_size, const int field_dims, const double Mmin, const double Mmax, const int nbins, const double box)
{
    //Each entry i is mass between (i,i+1) / nbins * (log(Mmax) - log(Mmin)) + log(Mmin)
    std::valarray<int> halo_hist(0,nbins);
    const double grid = box/field_dims;
    int field_dlas = 0, tot_dlas=0;

    //Store index in a map as the easiest way of sorting it
    std::map<const double, const int> sort_mass;
    //Insert - the mass into the map, so that the largest halo comes first.
    for (int i=0; i< nhalo; ++i){
        sort_mass.insert(std::pair<const double, const int>(-1*sub_mass[i],i));
    }

    //Store index in a map as the easiest way of sorting it
    std::multimap<const int32_t, const int> sort_sub_index;
    //Insert - the index into the map, so that the subhalos of the largest halo comes first.
    for (int i=0; i< nsubhalo; ++i){
        sort_sub_index.insert(std::pair<const int32_t, const int>(subsub_index[i],i));
    }
    //field_size can be the true allocated array size since all the extra bits are zero at this point
    #pragma omp parallel for reduction(+:tot_dlas) reduction(+:field_dlas)
    for (size_t i = 0; i< field_size; i++)
    {
        //Don't care about not DLAs.
        if (field[i] <= 0.)
            continue;
        //Convert index back into position
        const int zoff = (i % (2*field_dims/2+EXTRA));
        const double zcoord = (box/field_dims) * zoff;
        const int yoff = ( (i - zoff)/(2*field_dims/2+EXTRA) % field_dims);
        const double ycoord = (box/field_dims) * yoff;
        const double xcoord = (box/field_dims) * (((i - zoff)/(2*field_dims/2+EXTRA) - yoff ) / field_dims);
        if(xcoord < 0 || xcoord > box || ycoord < 0 || ycoord > box || zcoord < 0 || zcoord > box){
            printf("Bad coord: %g %g %g\n",xcoord,ycoord, zcoord);
            exit(1);
        }
        tot_dlas++;
        // Largest halo where the particle is within r_vir.
        int nearest_halo=-1;
        for (std::map<const double,const int>::const_iterator it = sort_mass.begin(); it != sort_mass.end(); ++it)
        {
            if (is_halo_close(it->second, xcoord, ycoord, zcoord, sub_pos, sub_radii, box, grid)) {
//                 printf("(%g %g %g) halo: %d rad %g pos %g %g %g\n",xcoord, ycoord, zcoord, it->second, sub_radii[it->second], sub_pos[3*it->second],sub_pos[3*it->second+1], sub_pos[3*it->second+2]);
                nearest_halo = it->second;
                break;
            }
        }
        //If no halo found, loop over subhalos.
        if (nearest_halo < 0){
          for (std::multimap<const int32_t,const int>::const_iterator it = sort_sub_index.begin(); it != sort_sub_index.end(); ++it){
            //If close to a subhalo, assign to the parent halo.
            if (is_halo_close(it->second, xcoord, ycoord, zcoord, subsub_pos, subsub_radii, box, grid)) {
                nearest_halo = it->first;
                break;
            }
          }
        }
        if (nearest_halo >= 0){
            double mass = sub_mass[nearest_halo];
            //The +10 converts from gadget 1e10 solar mass units to solar masses.
            int bin = nbins*(log10(mass)+10 - log10(Mmin))/(log10(Mmax) - log10(Mmin));
            if ((bin >= nbins) || (bin < 0)){
                fprintf(stderr,"Suggested bin %d (mass %g) out of array\n",bin, mass);
                exit(1);
            }
            #pragma omp critical (_dla_cross_)
            {
                halo_hist[bin] += 1;
            }
        }
        else{
            field_dlas++;
        }
    }
    printf("Field DLAs: %d tot_dlas %d\n",field_dlas, tot_dlas);
    return halo_hist;
}
