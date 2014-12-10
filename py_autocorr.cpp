/* Python module to calculate the autocorrelation function of a field*/
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include "numpy/arrayobject.h"
#include <omp.h>
#include <cassert>
#include <gsl/gsl_rng.h>

//Compute the absolute distance between two points
double distance(const int * a, const int * b, const npy_intp dims)
{
    double total=0;
    for(int i=0; i< dims; i++){
        total+=pow(*(a+i)-*(b+i),2);
    }
    return sqrt(total);
}


//Compute the absolute distance between two points
double distance2(const int * a, const int * b)
{
    const int dif1 = (*(a)-*(b));
    const int dif2 = (*(a+1)-*(b+1));
    return sqrt(dif1*dif1+dif2*dif2);
}

/*Check whether the passed array has type typename. Returns 1 if it doesn't, 0 if it does.*/
int check_type(PyArrayObject * arr, int npy_typename)
{
  return !PyArray_EquivTypes(PyArray_DESCR(arr), PyArray_DescrFromType(npy_typename));
}

/*Find the cross-correlation function of two quantities computed along a list of spectra
   Spectra are assumed to be along the same axis and a regular grid.
   slist - list of quantity along spectra to correlate. (nspectra, npix)
   slist2 - Second quantity to correlate with the first
   nspec - number of spectra in each direction
   nout - Number of bins in output function
*/
PyObject * _crosscorr_spectra(PyObject *self, PyObject *args)
{
    PyArrayObject *slist, *slist2;
    int nout, nspec;
    if(!PyArg_ParseTuple(args, "O!O!ii",&PyArray_Type, &slist, &PyArray_Type, &slist2, &nspec, &nout) )
    {
        PyErr_SetString(PyExc_AttributeError, "Incorrect arguments: use slist, slist2, nspec, nbins\n");
        return NULL;
    }
    if(check_type(slist, NPY_DOUBLE) || check_type(slist2, NPY_DOUBLE) )
    {
          PyErr_SetString(PyExc_AttributeError, "slist and slist2 must be float64.\n");
          return NULL;
    }
    if( PyArray_NDIM(slist) != 2 || PyArray_DIM(slist,0) != nspec*nspec)
    {
        PyErr_SetString(PyExc_ValueError, "slist must have dimensons (nspec^2, npix).\n");
        return NULL;
    }
    if( PyArray_NDIM(slist2) != 2 || PyArray_DIM(slist2,0) != nspec*nspec)
    {
        PyErr_SetString(PyExc_ValueError, "slist2 must have dimensons (nspec^2, npix).\n");
        return NULL;
    }
    const int npix = PyArray_DIM(slist,1);
    if( PyArray_DIM(slist2,1) != npix )
    {
        PyErr_SetString(PyExc_ValueError, "slist2 must have same dimensions as slist1 (nspec^2, npix).\n");
        return NULL;
    }
    //Allocate output memory
    npy_intp npnbins = nout;
    PyArrayObject *autocorr = (PyArrayObject *) PyArray_SimpleNew(1,&npnbins,NPY_DOUBLE);
    PyArray_FILLWBYTE(autocorr, 0);
    //Bin autocorrelation, must cover sqrt(dims)*size
    //so each bin has size sqrt(dims)*size /nbins
    #pragma omp parallel
    {
        double autocorr_C[nout] = {0};
        #pragma omp for nowait
        for (int x2=0; x2<nspec;x2++)
            for (int y2=0; y2<nspec;y2++)
                for (int x1=0; x1<nspec;x1++)
                    for (int y1=0; y1<nspec;y1++)
                        for (int z1=0; z1<npix;z1++)
                            for (int z2=0; z2<npix;z2++)
                            {
                                //Total distance between each point.
                                //Each dimension is normalised to 1.
                                double rr2 = ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))/(1.*nspec*nspec)+(z1-z2)*(z1-z2)/(1.*npix*npix);
                                //Note that for 3D we need sqrt(3), for 2D sqrt(2)
                                int cbin = floor(sqrt(rr2) * nout / (1.*sqrt(3.)));
                                autocorr_C[cbin]+=*(double *) PyArray_GETPTR2(slist, x1*nspec+y1, z1) * *(double *) PyArray_GETPTR2(slist2, x2*nspec+y2, z2);
                            }
       #pragma omp critical
       {
           for(int nn=0; nn< nout; nn++){
               *(double *)PyArray_GETPTR1(autocorr,nn)+=autocorr_C[nn];
           }
       }
    }
    return Py_BuildValue("O", autocorr);
}

/* Find the autocorrelation function from a sparse list of discrete tracer points.
   The field is assumed to be 1 at these points and zero elsewhere
   plist1. plist2 - list of points to autocorrelate. A tuple length n (n=2) of 1xP arrays.
   the first and second components of output of an np.where on a field of shape (nspec, npix)
   nspec - number of spectra in each direction
   npix - number of pixels in each direction
   nbins - number of bins in output autocorrelation function
*/
PyObject * _autocorr_list(PyObject *self, PyObject *args)
{
    PyArrayObject *plist1, *plist2;
    int nout, nspec, npix;
    if(!PyArg_ParseTuple(args, "O!O!iii",&PyArray_Type, &plist1, &PyArray_Type, &plist2, &nspec, &npix, &nout) )
        return NULL;
    if(check_type(plist2, NPY_INT64) || check_type(plist2, NPY_INT64))
    {
          PyErr_SetString(PyExc_AttributeError, "Input list is not 32-bit integer.\n");
          return NULL;
    }
    /*In practice assume this is 2*/
    if( PyArray_NDIM(plist2) != 1 || PyArray_NDIM(plist2) != 1)
    {
        PyErr_SetString(PyExc_AttributeError, "plist and plist2 must each be single dimensional arrays.\n");
        return NULL;
    }
    npy_intp points = PyArray_DIM(plist1,0);
    if( points !=  PyArray_DIM(plist2,0))
    {
        PyErr_SetString(PyExc_AttributeError, "plist1 and plist2 must have same length.\n");
        return NULL;
    }
    npy_intp npnbins = nout;
    //Array for output
    PyArrayObject *autocorr = (PyArrayObject *) PyArray_SimpleNew(1,&npnbins,NPY_DOUBLE);
    PyArray_FILLWBYTE(autocorr, 0);
    //Avg. density of the field: rho-bar
    #pragma omp parallel
    {
        //Bin autocorrelation, must cover sqrt(dims)*size
        //so each bin has size sqrt(dims)*size /nbins
        int64_t autocorr_C[nout] = {0};
        #pragma omp for nowait
        for(int b=0; b<points; b++){
            for(int a=0; a<points; a++){
                // plist contains position indices. Position of each point is (n, m) = (plist[0, a], plist[1,a])
                // In physical coordinates that is (x, y, z) = (n / nspec, n % nspec, m)
                // So distance is (x**2 + y**2)/nspec**2 + z**2/npix**2
                int na = *(int *)PyArray_GETPTR1(plist1,a);
                int ma = *(int *)PyArray_GETPTR1(plist2,a);
                int nb = *(int *)PyArray_GETPTR1(plist1,b);
                int mb = *(int *)PyArray_GETPTR1(plist2,b);
                //x difference
                double rr2 = (na / nspec - nb/nspec)*(na / nspec - nb/nspec)/(1.*nspec*nspec);
                //y difference
                rr2 += (na % nspec - nb % nspec)*(na % nspec - nb % nspec)/(1.*nspec*nspec);
                //z difference
                rr2 += (ma-mb)*(ma-mb)/(1.*npix*npix);
                //Which bin to add this one to?
                int cbin = floor(sqrt(rr2) * nout / (1.*sqrt(3.)));
                autocorr_C[cbin]+=1;
            }
        }
        #pragma omp critical
        {
           for(int nn=0; nn< nout; nn++){
               *(double *)PyArray_GETPTR1(autocorr,nn)+=autocorr_C[nn];
           }
        }
    }
    return Py_BuildValue("O", autocorr);
}

//Computes the number of modes in spectra spaced regularly on a grid using brute-force O(n^2) computations.
PyObject * _modecount_slow(PyObject *self, PyObject *args)
{
    int nspec;
    int nout;
    int npix;
    //Arguments: nspec is number of spectra within the nspec, 
    //npix is number of pixels along the line of sight.
    //nout is number of bins in the resulting output correlation function
    if(!PyArg_ParseTuple(args, "iii",&nspec, &npix, &nout) )
    {
        PyErr_SetString(PyExc_AttributeError, "Incorrect arguments: \n"
              "nspec: number of spectra in x and y, \n"
              "npix: number of pixels in z, \n"
              "nout: nunber of output bins\n");
        return NULL;
    }
    int64_t count[nout]={0};
    npy_intp npnout = nout;
    for (int x2=0; x2<nspec;x2++)
    for (int y2=0; y2<nspec;y2++)
    for (int x1=0; x1<nspec;x1++)
    for (int y1=0; y1<nspec;y1++)
    for (int z1=0; z1<npix;z1++)
    for (int z2=0; z2<npix;z2++)
    {
       //Total distance between each point. 
       //Each dimension is normalised to 1.
       double rr2 = ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))/(1.*nspec*nspec)+(z1-z2)*(z1-z2)/(1.*npix*npix);
       //Note that for 3D we need sqrt(3), for 2D sqrt(2)
       int cbin = floor(sqrt(rr2) * nout / (1.*sqrt(3.)));
       count[cbin]++;
    }
    PyArrayObject *pycount = (PyArrayObject *) PyArray_SimpleNew(1,&npnout,NPY_INT64);
    for(int nn=0; nn< nout; nn++){
        *(int64_t *)PyArray_GETPTR1(pycount,nn)=count[nn];
    }
    return Py_BuildValue("O", pycount);
}


PyObject * _modecount(PyObject *self, PyObject *args)
{
    int nspec;
    int nout;
    int npix;
    //Arguments: nspec is number of spectra within the nspec, 
    //npix is number of pixels along the line of sight.
    //nout is number of bins in the resulting output correlation function
    if(!PyArg_ParseTuple(args, "iii",&nspec, &npix, &nout) )
    {
        PyErr_SetString(PyExc_AttributeError, "Incorrect arguments: \n"
        "nspec: number of spectra in x and y, \n"
        "npix: number of pixels in z, \n"
        "nout: nunber of output bins\n");
        return NULL;
    }
    //Thread-private variables are stack allocated.
    //In practice each stack gets 8MB ~ 1e6 64 int64s, which we are unlikely to need,
    //but check just in case.
    if (nout > 1000000 || nout <= 0){
        PyErr_SetString(PyExc_ValueError, "Number of output bins is <= 0 or too large (would overflow the stack)\n");
        return NULL;
    }
    if (npix < 0 || nspec < 0){
        PyErr_SetString(PyExc_ValueError, "Number of x, y, or z pixels is < 0\n");
        return NULL;
    }
    npy_intp npnout = nout;
    //Allocate memory for output
    PyArrayObject *pycount = (PyArrayObject *) PyArray_SimpleNew(1,&npnout,NPY_INT64);
    PyArray_FILLWBYTE(pycount, 0);
    // Special treatment for x=y=z=0 mode
    *(int64_t *)PyArray_GETPTR1(pycount,0)= nspec*nspec*npix;
    #pragma omp parallel
    {
        int64_t count[nout]={0};
        //x=y=z=0 already done
        //x=y=0
        #pragma omp for nowait
        for (int z=1; z<npix;z++){
            int cbin = floor(z * nout / (1.*npix*sqrt(3.)));
            count[cbin]+=2*(npix-z)*nspec*nspec;
        }
        #pragma omp for nowait
        for (int x=1; x<nspec;x++){
            //This might look like we are double-counting for the y=0 case.
            //However, this is not so; because x and y are symmetric,
            //the modes for (x=0, y!=0) are the same
            //as those for (y=0, x!=0), so we just do both at ones.
            //x!=0 y!=0 z!=0
            for (int y=0; y<nspec;y++){
                double rr2 = x*x+y*y;
                int cbin = floor(sqrt(rr2) * nout / (1.*nspec*sqrt(3.)));
                count[cbin]+=4*(nspec-y)*(nspec-x)*npix;
                //Amount to scale each dimension by so that box is cube of side 1.
                for (int z=1; z<npix;z++){
                    double rr2_inner = rr2+pow(z*nspec/1./npix,2);
                    int cbin = floor(sqrt(rr2_inner) * nout / (1.*nspec*sqrt(3.)));
                    count[cbin]+=8*(nspec-y)*(nspec-x)*(npix-z);
                }
            }
        }
        #pragma omp critical
        {
            for(int nn=0; nn< nout; nn++){
                *(int64_t *)PyArray_GETPTR1(pycount,nn)+=count[nn];
            }
        }
    }
    return Py_BuildValue("O", pycount);
}

PyObject * _modecount_2d(PyObject *self, PyObject *args)
{
    int box;
    int nbins;
    if(!PyArg_ParseTuple(args, "ii",&box, &nbins) )
        return NULL;
    int count[nbins]={0};
    npy_intp npnbins = nbins;
    // Special treatment for x=0 mode which would otherwise be double-counted
    // x=y=0
    count[0] = box*box;
    for (int y=1; y<box;y++){
       int cbin = floor(y * nbins / (1.*box*sqrt(2.)));
       count[cbin]+=2*(box-y)*box;
    }
    for (int x=1; x<box;x++){
        // Special treatment for y=0 mode which would otherwise be double-counted
        int cbin = floor(x * nbins / (1.*box*sqrt(2.)));
        count[cbin]+=2*(box-x)*box;
        for (int y=1; y<box;y++){
           double rr = sqrt(x*x+y*y);
           int cbin = floor(rr * nbins / (1.*box*sqrt(2.)));
           count[cbin]+=4*(box-y)*(box-x);
        }
    }
    PyArrayObject *pycount = (PyArrayObject *) PyArray_SimpleNew(1,&npnbins,NPY_INT);
    for(int nn=0; nn< nbins; nn++){
        *(int *)PyArray_GETPTR1(pycount,nn)=count[nn];
    }
    return Py_BuildValue("O", pycount);
}

//Compute the number of modes present in a bin using monte carlo techniques
//Note as this is a histogram we have not computed error; just call it twice with differing samples and see the difference.
PyObject * _modecount_monte_carlo_2d(PyObject *self, PyObject *args)
{
    int box;
    int nbins;
    int nsamples;
    if(!PyArg_ParseTuple(args, "iii",&box, &nbins, &nsamples) )
        return NULL;
    int count[nbins];
    memset(count,0,nbins*sizeof(int));
    npy_intp npnbins = nbins;
    gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(r, 23);
    assert(gsl_rng_min(r) == 0);
    for (int i=0; i<nsamples; i++){
        int a = gsl_rng_uniform_int(r, box);
        int b = gsl_rng_uniform_int(r, box);
        int x = gsl_rng_uniform_int(r, box);
        int y = gsl_rng_uniform_int(r, box);
        double rr = sqrt((x-a)*(x-a)+(y-b)*(y-b));
        int cbin = floor(rr * nbins / (1.*box*sqrt(2.)));
        count[cbin]++;
    }
    gsl_rng_free(r);
    //Copy back into python array
    PyArrayObject *pycount = (PyArrayObject *) PyArray_SimpleNew(1,&npnbins,NPY_INT);
    //Total number of modes
    long long total = pow(box,4);
    for(int nn=0; nn< nbins; nn++){
        //Correct for samples
        *(int *)PyArray_GETPTR1(pycount,nn)=(total*count[nn])/nsamples;
//         if (count[nn] < 10)
//             printf("Not enough samples count[%d] = %d\n", nn, count[nn]);
    }
    //Note: the largest-scale modes near the box should be computed 
    //directly
    //as getting a fair sample there will be hard
    return Py_BuildValue("O", pycount);
}


static PyMethodDef __autocorr[] = {
  {"crosscorr_spectra", _crosscorr_spectra, METH_VARARGS,
   "Find the autocorrelation function from a list of spectra"
   "Spectra are assumed to be along the same axis and on a regular grid."
   "slist - First quantity along spectrum to correlate. (nspectra, npix)"
   "slist2 - Second quantity to correlate. (nspectra, npix)"
   "nspec - number of spectra in each direction"
   "nout - Number of bins in output function"
   "    "},
  {"autocorr_list", _autocorr_list, METH_VARARGS,
   "Calculate the autocorrelation function"
   "    Arguments: plist1, plist2, spec, pix, nbins"
   "    "},
  {"modecount_2d", _modecount_2d, METH_VARARGS,
   "Calculate the number of modes in each bin"
   "    Arguments: box, nbins"
   "    "},
  {"modecount_mc_2d", _modecount_monte_carlo_2d, METH_VARARGS,
   "Calculate the number of modes in each bin with monte carlo"
   "    Arguments: box, nbins, nsamples"
   "    "},
  {"modecount_slow", _modecount_slow, METH_VARARGS,
   "Calculate the number of modes in 3D, binned, assuming a regular grid, a really slow way."
   "    Arguments: nspec: number of spectra in x and y, npix: number of pixels in z, nout: nunber of output bins"
   "    "},
  {"modecount", _modecount, METH_VARARGS,
       "Calculate the number of modes in 3D, binned, assuming a regular grid, a faster way."
       "    Arguments: nspec: number of spectra in x and y, npix: number of pixels in z, nout: nunber of output bins"
       "    "},
  {NULL, NULL, 0, NULL},
};

PyMODINIT_FUNC
init_autocorr_priv(void)
{
  Py_InitModule("_autocorr_priv", __autocorr);
  import_array();
}
