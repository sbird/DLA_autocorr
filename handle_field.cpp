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

/*Count the values of a field to find the pdf*/
std::map<double, int> pdf(double * field, int size, double xmin, double xmax, double xstep)
{
    std::map<double, int> hist;
    
    //Initialise the key values
    for (double i = xmin; i < xmax; i+=xstep)
        hist[pow(10,i)] = 0;
    hist[xmax] = 0;
    
    //Count values: note this means that the last entry will contain the 
    //no. of elements past the end
    for (int i = 0; i < size; ++i)
    {
        //Lower bound = first key equal to or greater than.
        std::map<double,int>::iterator it = hist.lower_bound(*(field+i));
        if (it != hist.begin())
            *(it--)++;
    }
    
    return hist;
}

