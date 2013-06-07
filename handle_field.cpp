#include "moments.h"
#include <map>
#include <cmath>
/** Functions for working with an interpolated field*/

/* Multiply all elements of size field by a top hat function H(x > thresh)*/
void multiply_by_tophat(double * field, int size, double thresh)
{
    #pragma omp parallel for
    for(int i=0; i< size; i++)
    {
        field[i] *= (field[i] > thresh);
    }
}

/* Make all positive array values 1*/
void discretize(double * field, int size)
{
    #pragma omp parallel for
    for(int i=0; i< size; i++)
    {
        if(field[i] > 0)
            field[i] = 1;
    }
}


/* Normalise by the mean of the field, so that the field has mean 0,
 * sd. 1.
 * ie, calculate delta = rho / rho_bar - 1 */
void calc_delta(double * field, int size, int realsize)
{
    const double mean = find_total(field, size) / realsize;
    #pragma omp parallel for
    for(int i=0; i< size; i++)
    {
        field[i] = field[i]/ mean -1;
    }
}

/* Find the total value of all elements of a field*/
double find_total(double * field, int size)
{
    double total=0;
    #pragma omp parallel for reduction(+:total)
    for(int i=0; i< size; i++)
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
std::map<double, int> histogram(const double * field, const int size, const double xmin, const double xmax, const int nxbins)
{
    std::map<double, int> hist;
    
    //Initialise the key values
    for (double i = log10(xmin); i < log10(xmax); i+=(log10(xmax)-log10(xmin))/nxbins)
        hist[pow(10.,i)] = 0;
    
    //Count values: note this means that the last entry will contain the 
    //no. of elements past the end
    for (int i = 0; i < size; ++i)
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
std::map<double, double> pdf(std::map<double, int> hist, const int size)
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
