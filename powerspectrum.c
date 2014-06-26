/* Copyright (c) 2009, Simeon Bird <spb41@cam.ac.uk>
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

#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/** \file 
 * Defines powerspectrum() wrapper around FFTW*/
//extern float invwindow(int kx, int ky, int kz, int n);
#define invwindow( kx, ky, kz, n) (1)

/*Note we need some contiguous memory space after the actual data in field. *The real input data has size
 *dims*dims*dims
 *The output has size dims*dims*(dims/2+1) *complex* values
 * So need dims*dims*dims+2 float space.
 * Also the field needs to be stored carefully to make the 
 * extra space be in the right place. */

/**Little macro to work the storage order of the FFT.*/
#define KVAL(n) ((n)<=dims/2 ? (n) : ((n)-dims))

int powerspectrum(int dims, fftwf_plan* pl,fftwf_complex *outfield, int nrbins, double *power, int *count,double *keffs)
{
    const size_t dims2=dims*dims;
    const size_t dims3=dims2*dims;
    const size_t fdims = (dims/2+1);
    const size_t fdims2 = dims*fdims;
    /*How many bins per unit interval in k?*/
    const int binsperunit=nrbins/(floor(sqrt(3)*abs((dims+1.0)/2.0)+1));
    /*Half the bin width*/
    const float bwth=1.0/(2.0*binsperunit);
    fftwf_execute(*pl);
    memset(power, 0, nrbins*sizeof(double));
    memset(count, 0, nrbins*sizeof(int));
	/* Now we compute the powerspectrum in each direction.
	 * FFTW is unnormalised, so we need to scale by the length of the array
	 * (we do this later). */
	for(int i=0; i< nrbins/2; i++){
		/* bin center (k) is i+a.
		 * a is bin width/2, is 0.5
		 * k_eff is k+ 2a^2k/(a^2+3k^2) */
		float k=i*2.0*bwth;
		keffs[i]=(k+bwth)+2*pow(bwth,2)*(k+bwth)/(pow(bwth,2)+3*pow((k+bwth),2));
	}
	/*After this point, the number of modes is decreasing.*/
	for(int i=nrbins/2; i< nrbins; i++){
		/* bin center (k) is i+a.
		 * a is bin width/2, is 0.5
		 * k_eff is k+ 2a^2k/(a^2+3k^2) */
		float k=i*2.0*bwth;
		keffs[i]=(k+bwth)-2*pow(bwth,2)*(k+bwth)/(pow(bwth,2)+3*pow((k+bwth),2));
	}
	#pragma omp parallel 
	{
		float powerpriv[nrbins];
		int countpriv[nrbins];
        memset(powerpriv, 0, nrbins*sizeof(float));
        memset(countpriv, 0, nrbins*sizeof(int));
		/* Want P(k)= F(k).re*F(k).re+F(k).im*F(k).im
		 * Use the symmetry of the real fourier transform to half the final dimension.*/
		#pragma omp for schedule(static, 128) nowait
		for(int i=0; i<dims;i++){
			size_t indx=i*fdims2;
			for(int j=0; j<dims; j++){
				size_t indy=j*fdims;
				/* The k=0 and N/2 mode need special treatment here, 
				 * as they alone are not doubled.*/
				/*Do k=0 mode.*/
				size_t index=indx+indy;
				float kk=sqrt(pow(KVAL(i),2)+pow(KVAL(j),2));
				int psindex=floor(binsperunit*kk);
				powerpriv[psindex]+=(pow(outfield[index][0],2)+pow(outfield[index][1],2))*pow(invwindow(KVAL(i),KVAL(j),0,dims),2);
				countpriv[psindex]++;
				/*Now do the k=N/2 mode*/
				index=indx+indy+dims/2;
				kk=sqrt(pow(KVAL(i),2)+pow(KVAL(j),2)+pow(KVAL(dims/2),2));
				psindex=floor(binsperunit*kk);
				powerpriv[psindex]+=(pow(outfield[index][0],2)+pow(outfield[index][1],2))*pow(invwindow(KVAL(i),KVAL(j),KVAL(dims/2),dims),2);
				countpriv[psindex]++;
				/*Now do the rest. Because of the symmetry, each mode counts twice.*/
				for(int k=1; k<dims/2; k++){
					index=indx+indy+k;
					kk=sqrt(pow(KVAL(i),2)+pow(KVAL(j),2)+pow(KVAL(k),2));
					psindex=floor(binsperunit*kk);
					/* Correct for shot noise and window function in IDL. 
					 * See my notes for the reason why.*/
					powerpriv[psindex]+=2*(pow(outfield[index][0],2)+pow(outfield[index][1],2))*pow(invwindow(KVAL(i),KVAL(j),KVAL(k),dims),2);
					countpriv[psindex]+=2;
				}
			}
		}
		#pragma omp critical
		{
			for(int i=0; i< nrbins;i++){
				power[i]+=powerpriv[i];
				count[i]+=countpriv[i];
			}
		}
	}
	for(int i=0; i< nrbins;i++){
		if(count[i]){
			/* I do the division twice to avoid any overflow.*/
			power[i]/=dims3;
			power[i]/=dims3;
			power[i]/=count[i];
		}
	}
	return 0;
}

