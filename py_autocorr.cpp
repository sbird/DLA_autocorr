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

//Compute square of distance between two sightlines, assuming same axis
double spec_distance2(const double * a, const double * b)
{
    const double dif[2] = {(*(a)-*(b)), (*(a+1)-*(b+1))};
    return dif[0]*dif[0]+dif[1]*dif[1];
}

/*Find the autocorrelation function from a list of spectra
   Spectra are assumed to be along the same axis.
   slist - list of quantity along spectra to autocorrelate. npix * nspectra
   spos -  positions of the spectra: 2x nspectra: (x, y).
   nbins - number of bins in output autocorrelation function
   pixsz - Size of a pixel in units of the cofm.
*/
PyObject * _autocorr_spectra(PyObject *self, PyObject *args)
{
    PyArrayObject *slist, *spos;
    int nbins;
    double pixsz;
    if(!PyArg_ParseTuple(args, "O!O!di",&PyArray_Type, &slist, &PyArray_Type, &spos, &pixsz, &nbins) )
    {
        PyErr_SetString(PyExc_AttributeError, "Incorrect arguments: use slist, spos, pixsz, nbins\n");
        return NULL;
    }
    if(check_type(slist, NPY_DOUBLE) || check_type(spos, NPY_DOUBLE))
    {
          PyErr_SetString(PyExc_AttributeError, "Input arrays are not float64.\n");
          return NULL;
    }
    npy_intp npix = PyArray_DIM(slist,0);
    npy_intp nspectra = PyArray_DIM(slist,1);
    npy_intp npnbins = nbins;
    //Bin autocorrelation, must cover sqrt(dims)*size
    //so each bin has size sqrt(dims)*size /nbins
    const int nproc = omp_get_num_procs();
    double autocorr_C[nproc][nbins];
    int modecount_C[nproc][nbins];
    memset(modecount_C,0,nproc*nbins*sizeof(int));
    memset(autocorr_C,0,nproc*nbins*sizeof(int));
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const double binsz = nbins / (npix * pixsz*sqrt(3));
        #pragma omp for
        for(int b=0; b<nspectra; b++){
            for(int a=0; a<nspectra; a++){
                double sdist2 = spec_distance2((double *) PyArray_GETPTR2(spos,0, a), (double *) PyArray_GETPTR2(spos,0, b));
                double * speca = (double *) PyArray_GETPTR2(slist,0, a);
                double * specb = (double *) PyArray_GETPTR2(slist,0, b);
                for(int bb=0; bb<npix; bb++){
                    for(int aa=0; aa<npix; aa++){
                        double rr = sqrt(sdist2+ (bb-aa)*(bb-aa)*pixsz);
                        //Which bin to add this one to?
                        int cbin = floor(rr * binsz);
                        autocorr_C[tid][cbin]+=speca[aa]*specb[bb];
                        modecount_C[tid][cbin]+=1;
                    }
                }
            }
       }
    }
    PyArrayObject *autocorr = (PyArrayObject *) PyArray_SimpleNew(1,&npnbins,NPY_DOUBLE);
    PyArray_FILLWBYTE(autocorr, 0);
    PyArrayObject *modecount = (PyArrayObject *) PyArray_SimpleNew(1,&npnbins,NPY_INT);
    PyArray_FILLWBYTE(modecount, 0);

    for(int tid=0; tid < nproc; tid++){
        for(int nn=0; nn< nbins; nn++){
            *(double *)PyArray_GETPTR1(autocorr,nn)+=autocorr_C[tid][nn];
	    *(int *)PyArray_GETPTR1(modecount,nn)+=modecount_C[tid][nn];
        }
    }
    return Py_BuildValue("OO", modecount, autocorr);
}

/*Find the autocorrelation function from a sparse list of discrete tracer points.
   The field is assumed to be 1 at these points and zero elsewhere
   list - list of points to autocorrelate. A tuple length n (n=2) of 1xP arrays:
   the output of an np.where on an n-d field
   nbins - number of bins in output autocorrelation function
   size - size of the original field (assumed square), so field has dimensions (size,size..) n times
   norm - If true, normalise by the number of possible cells in each bin
*/
PyObject * _autocorr_list(PyObject *self, PyObject *args)
{
    PyArrayObject *plist;
    int nbins, size;
    if(!PyArg_ParseTuple(args, "O!iii",&PyArray_Type, &plist, &nbins, &size) )
        return NULL;
    if(check_type(plist, NPY_INT32))
    {
          PyErr_SetString(PyExc_AttributeError, "Input list is not 32-bit integer.\n");
          return NULL;
    }
    /*In practice assume this is 2*/
    npy_intp dims = PyArray_DIM(plist,0);
    assert(dims == 2);
    npy_intp points = PyArray_DIM(plist,1);
    npy_intp npnbins = nbins;
    //Bin autocorrelation, must cover sqrt(dims)*size
    //so each bin has size sqrt(dims)*size /nbins
    const int nproc = omp_get_num_procs();
    int autocorr_C[nproc][nbins];
    memset(autocorr_C,0,nproc*nbins*sizeof(int));
    //Avg. density of the field: rho-bar
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        #pragma omp for
        for(int b=0; b<points; b++){
            for(int a=0; a<points; a++){
                //TODO: Check actually int
                double rr = distance2((int *)PyArray_GETPTR2(plist,0,a), (int *) PyArray_GETPTR2(plist,0,b));
                //Which bin to add this one to?
                int cbin = floor(rr * nbins / (size*sqrt(dims)));
                autocorr_C[tid][cbin]+=1;
            }
        }
    }
    PyArrayObject *autocorr = (PyArrayObject *) PyArray_SimpleNew(1,&npnbins,NPY_DOUBLE);
    PyArray_FILLWBYTE(autocorr, 0);
    for(int tid=0; tid < nproc; tid++){
        for(int nn=0; nn< nbins; nn++){
            *(double *)PyArray_GETPTR1(autocorr,nn)+=autocorr_C[tid][nn];
        }
    }
    return Py_BuildValue("O", autocorr);
}

/* Count the modes present in a list of spectra
   spos -  positions of the spectra: 2x nspectra: (x, y).
   nbins - number of bins in output autocorrelation function
   pixsz - Size of a pixel in units of the cofm.
   npix - number of pixels
*/
PyObject * _modecount_spectra(PyObject *self, PyObject *args)
{
    PyArrayObject *spos;
    int nbins, npix;
    double pixsz;
    if(!PyArg_ParseTuple(args, "O!dii",&PyArray_Type, &spos, &pixsz, &nbins, &npix) )
    {
        PyErr_SetString(PyExc_AttributeError, "Incorrect arguments: use spos, pixsz, nbins\n");
        return NULL;
    }
    if(check_type(spos, NPY_DOUBLE))
    {
          PyErr_SetString(PyExc_AttributeError, "Input arrays are not float64.\n");
          return NULL;
    }
    npy_intp nspectra = PyArray_DIM(spos,1);
    //Bin autocorrelation, must cover sqrt(dims)*size
    //so each bin has size sqrt(dims)*size /nbins
    const int nproc = omp_get_num_procs();
    int modecount_C[nproc][nbins];
    memset(modecount_C,0,nproc*nbins*sizeof(int));
    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const double binsz = nbins / (npix * pixsz*sqrt(3));
        #pragma omp for
        for(int b=0; b<nspectra; b++){
            for(int a=0; a<nspectra; a++){
                double sdist2 = spec_distance2((double *) PyArray_GETPTR2(spos,0, a), (double *) PyArray_GETPTR2(spos,0, b));
                for(int bb=0; bb<npix; bb++){
                    for(int aa=0; aa<npix; aa++){
                        double rr = sqrt(sdist2+ (bb-aa)*(bb-aa)*pixsz);
                        //Which bin to add this one to?
                        int cbin = floor(rr * binsz);
                        modecount_C[tid][cbin]+=1;
                    }
                }
            }
        }
    }
    npy_intp npnbins = nbins;
    PyArrayObject *modecount = (PyArrayObject *) PyArray_SimpleNew(1,&npnbins,NPY_INT);
    PyArray_FILLWBYTE(modecount, 0);

    for(int tid=0; tid < nproc; tid++){
        for(int nn=0; nn< nbins; nn++){
            *(int *)PyArray_GETPTR1(modecount,nn)+=modecount_C[tid][nn];
        }
    }
    return Py_BuildValue("O", modecount);
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
        return NULL;
    int count[nout];
    memset(count,0,nout*sizeof(int));
    npy_intp npnout = nout;
    //Amount to scale each dimension by so that box is cube of side 1.
    const double specscale= 1./nspec/nspec;
    const double pixscale = 1./npix/npix;
    for (int x2=0; x2<nspec;x2++)
    for (int y2=0; y2<nspec;y2++)
    for (int x1=0; x1<nspec;x1++)
    for (int y1=0; y1<nspec;y1++)
    for (int z1=0; z1<npix;z1++)
    for (int z2=0; z2<npix;z2++)
    {
       //Total distance between each point. 
       //Each dimension is normalised to 1.
       double rr2 = ((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))*specscale+(z1-z2)*(z1-z2)*pixscale;
       //Note that for 3D we need sqrt(3), for 2D sqrt(2)
       int cbin = floor(sqrt(rr2) * nout / (1.*sqrt(3.)));
       count[cbin]++;
    }
    PyArrayObject *pycount = (PyArrayObject *) PyArray_SimpleNew(1,&npnout,NPY_INT);
    for(int nn=0; nn< nout; nn++){
        *(int *)PyArray_GETPTR1(pycount,nn)=count[nn];
    }
    return Py_BuildValue("O", pycount);
}


PyObject * _modecount_3d(PyObject *self, PyObject *args)
{
    int nspec;
    int nout;
    int npix;
    //Arguments: nspec is number of spectra within the nspec, 
    //npix is number of pixels along the line of sight.
    //nout is number of bins in the resulting output correlation function
    if(!PyArg_ParseTuple(args, "iii",&nspec, &npix, &nout) )
        return NULL;
    int count[nout];
    memset(count,0,nout*sizeof(int));
    npy_intp npnout = nout;
    //Amount to scale each dimension by so that box is cube of side 1.
    const double specscale= 1./nspec/nspec;
    const double pixscale = 1./npix/npix;
    // Special treatment for x=0 mode which would otherwise be double-counted
    // x=y=z=0
    count[0] = nspec*nspec*npix;
    //x=y=0
    for (int z=1; z<npix;z++){
        int cbin = floor(z * nout / (1.*npix*sqrt(3.)));
        count[cbin]+=2*(npix-z)*nspec*nspec;
    }
    for (int x=1; x<nspec;x++){
        //This might look like we are double-counting for the y=0 case.
        //However, this is not so; because x and y are symmetric, 
        //the modes for (x=0, y!=0) are the same 
        //as those for (y=0, x!=0), so we just do both at ones.
        //x!=0 y!=0 z!=0
        for (int y=0; y<nspec;y++){
            double rr = sqrt(x*x+y*y);
            int cbin = floor(rr * nout / (1.*nspec*sqrt(3.)));
            count[cbin]+=4*(nspec-y)*(nspec-x)*npix;
            for (int z=1; z<npix;z++){
                double rr2 = (x*x+y*y)*specscale+z*z*pixscale;
                int cbin = floor(sqrt(rr2) * nout / (1.*sqrt(3.)));
                count[cbin]+=8*(nspec-y)*(nspec-x)*(npix-z);
            }
        }
    }
    PyArrayObject *pycount = (PyArrayObject *) PyArray_SimpleNew(1,&npnout,NPY_INT);
    for(int nn=0; nn< nout; nn++){
        *(int *)PyArray_GETPTR1(pycount,nn)=count[nn];
    }
    return Py_BuildValue("O", pycount);
}

PyObject * _modecount(PyObject *self, PyObject *args)
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
PyObject * _modecount_monte_carlo(PyObject *self, PyObject *args)
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
  {"autocorr_spectra", _autocorr_spectra, METH_VARARGS,
   "Find the autocorrelation function from a list of spectra"
   "Spectra are assumed to be along the same axis."
   "slist - list of quantity along spectra to autocorrelate. npix * nspectra"
   "spos -  positions of the spectra: 2x nspectra: (x, y). "
   "pixsz - Size of a pixel in units of the cofm."
   "nbins - number of bins in output autocorrelation function"
   "    "},
  {"autocorr_list", _autocorr_list, METH_VARARGS,
   "Calculate the autocorrelation function"
   "    Arguments: plist, nbins, size, norm"
   "    "},
  {"modecount", _modecount, METH_VARARGS,
   "Calculate the number of modes in each bin"
   "    Arguments: box, nbins"
   "    "},
  {"modecount_mc", _modecount_monte_carlo, METH_VARARGS,
   "Calculate the number of modes in each bin with monte carlo"
   "    Arguments: box, nbins, nsamples"
   "    "},
  {"modecount_slow", _modecount_slow, METH_VARARGS,
   "Calculate the number of modes in 3D, binned, assuming a regular grid, a really slow way."
   "    Arguments: nspec: number of spectra in x and y, npix: number of pixels in z, nout: nunber of output bins"
   "    "},
  {"modecount_3d", _modecount_3d, METH_VARARGS,
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
