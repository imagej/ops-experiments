#include<stdio.h>

#include "MKLFFTW.h"
#include "mkl.h"
#include "mkl_vml_functions.h"
#include "mkl_vml.h"
#include "mkl_dfti.h"
#include "mkl_cblas.h"

#include "fftw/fftw3.h"
#include "fftw/fftw3_mkl.h"

int main() {
	int w=512;
	int h=512;

	printf("entering\n");

	float * x=(float*)malloc(sizeof(float)*w*h);
	float * y=(float*)malloc(sizeof(float)*2*(w/2+1)*h);

	testMKLFFTW(x,y,w,h);

	printf("success\n");
	free(x);
	free(y);
		 
}

extern "C" EXPORT void testMKLFFTW(float * x_, float * y_, int width, int height) {
	
	printf("starting mkl fftwf\n");
	
	fftwf_plan plan = fftwf_plan_dft_r2c_2d(width, height, x_, (fftwf_complex*)y_,
		(int)FFTW_ESTIMATE);

	fftwf_execute(plan);

	fftwf_destroy_plan(plan);

}

extern "C" EXPORT void mklConvolve(float * x, float *h, float *y, float * X_, float * H_, const int width, const int height, bool conj) {

	fftwf_plan forward1 = fftwf_plan_dft_r2c_2d(width, height, x, (fftwf_complex*)X_,
		(int)FFTW_ESTIMATE);

	fftwf_plan forward2 = fftwf_plan_dft_r2c_2d(width, height, h, (fftwf_complex*)H_,
		(int)FFTW_ESTIMATE);

	fftwf_plan inverse = fftwf_plan_dft_c2r_2d(width, height, (fftwf_complex*)X_, y, (int)FFTW_ESTIMATE);

	fftwf_execute(forward1);
	fftwf_execute(forward2);

	const MKL_INT n = (width/2+1)*height;

	if (conj) {
		// multiply X_, H_ for convolution
		vcMulByConj(n, (MKL_Complex8*)X_, (MKL_Complex8*)H_, (MKL_Complex8*)X_);
	}
	else {
		// multiply X_, H_ for convolution
		vcMul(n, (MKL_Complex8*)X_, (MKL_Complex8*)H_, (MKL_Complex8*)X_);
	}

	fftwf_execute(inverse);

	fftwf_destroy_plan(forward1);
	fftwf_destroy_plan(forward2);
	fftwf_destroy_plan(inverse);

}

extern "C" EXPORT void mklConvolve3D(float * x, float *h, float *y, float * X_, float * H_, const int n0, const int n1, const int n2, bool conj) {

	fftwf_plan forward1 = fftwf_plan_dft_r2c_3d(n0, n1, n2, x, (fftwf_complex*)X_,
		(int)FFTW_ESTIMATE);

	fftwf_plan forward2 = fftwf_plan_dft_r2c_3d(n0, n1, n2, h, (fftwf_complex*)H_,
		(int)FFTW_ESTIMATE);

	fftwf_plan inverse = fftwf_plan_dft_c2r_3d(n0, n1, n2, (fftwf_complex*)X_, y, (int)FFTW_ESTIMATE);

	fftwf_execute(forward1);
	fftwf_execute(forward2);

	const MKL_INT imageSize = n0*n1*n2;
	const MKL_INT fftSize = n0*n1*(n2/2+1);

	if (conj) {
		// conjugate multiply X_, H_ for correlation
		vcMulByConj(fftSize, (MKL_Complex8*)X_, (MKL_Complex8*)H_, (MKL_Complex8*)X_);
	}
	else {
		// multiply X_, H_ for convolution
		vcMul(fftSize, (MKL_Complex8*)X_, (MKL_Complex8*)H_, (MKL_Complex8*)X_);
	}

	fftwf_execute(inverse);

	cblas_sscal(imageSize, 1. / (imageSize), y, 1);

	fftwf_destroy_plan(forward1);
	fftwf_destroy_plan(forward2);
	fftwf_destroy_plan(inverse);

}

extern "C" EXPORT void mklRichardsonLucy3D(int iterations, float * x, float *h, float*y, fftwf_complex* FFT_, fftwf_complex* H_,const int n0, const int n1, const int n2, float * normal) {

	if (normal == NULL) {
		printf("The normal is NULL!\n");
	}
	else {
		printf("We have recieved the normal!");
	}

	printf("starting mklrl 3D - ImageJ Version\n");
    
    const int imageSize=n0*n1*n2;
    const int fftSize=n0*n1*(n2/2+1);
    
    float * temp = (float*)malloc(sizeof(float)*n0*n1*n2);

	fftwf_plan forward1 = fftwf_plan_dft_r2c_3d(n0,n1,n2, y, (fftwf_complex*)FFT_,
		(int)FFTW_ESTIMATE);  

    // create FFT plan for PSF
	fftwf_plan forwardH = fftwf_plan_dft_r2c_3d(n0,n1,n2,h,(fftwf_complex*)H_,
		(int)FFTW_ESTIMATE);
    
    // execute FFT plan for PSF
    fftwf_execute(forwardH);

    fftwf_plan forward3 = fftwf_plan_dft_r2c_3d(n0,n1,n2, temp, (fftwf_complex*)FFT_,
		(int)FFTW_ESTIMATE);

	fftwf_plan inverse = fftwf_plan_dft_c2r_3d(n0,n1,n2, (fftwf_complex*)FFT_, temp, (int)FFTW_ESTIMATE);

    // iterations

	float delta = 0.00001;
    
    for (int i=0;i<iterations;i++) {
        // create reblurred
        
        printf("iteration %d\n", i);
        fflush(stdout);
       
        fftwf_execute(forward1);

        // multiply X_, H_ for convolution
        vcMul(fftSize, (MKL_Complex8*)FFT_, (MKL_Complex8*)H_, (MKL_Complex8*)FFT_);

        fftwf_execute(inverse);    
        cblas_sscal(imageSize, 1./(imageSize), temp, 1);
        
        // divide original image by temp
        //vsDiv(imageSize, x, temp, temp);
		for (int j = 0; j < imageSize; j++) {
			
			if (temp[j] > 0) {
				temp[j] = x[j] / temp[j];
			}
			else {
				temp[j] = 0;
			}
		}

      //  cblas_scopy(imageSize, temp, 1, y, 1);

        // correlate with PSF
        fftwf_execute(forward3);

        // multiply X_, H_* for correllation
        vcMulByConj(fftSize, (MKL_Complex8*)FFT_, (MKL_Complex8*)H_, (MKL_Complex8*)FFT_);

        fftwf_execute(inverse);
        cblas_sscal(imageSize, 1./imageSize, temp, 1);

        // multiply by y
        vsMul(imageSize, y, temp, y);

		if (normal != NULL) {
			//vsDiv(imageSize, y, normal, y);
			for (int j = 0; j < imageSize; j++) {

				if (normal[j] > 0) {
					y[j] = y[j] / normal[j];
				}
				else {
					y[j] = y[j]/1.;
				}
			}
		}
 
    }
     
    //cblas_scopy(width*height, temp, 1, y, 1);
    
    fftwf_destroy_plan(forward1);
	fftwf_destroy_plan(forwardH);
	fftwf_destroy_plan(inverse);
    
    free(temp);

}

void testMKLFFT()
{

	//float _Complex x[32][100];
	float x[32][100];
	float y[34][102];

	printf("starting");
	
	DFTI_DESCRIPTOR_HANDLE my_desc1_handle;
	DFTI_DESCRIPTOR_HANDLE my_desc2_handle;
	MKL_LONG status, l[2];
	//...put input data into x[j][k] 0<=j<=31, 0<=k<=99
	//...put input data into y[j][k] 0<=j<=31, 0<=k<=99
	l[0] = 32; l[1] = 100;
	status = DftiCreateDescriptor( &my_desc1_handle, DFTI_SINGLE,
			  DFTI_COMPLEX, 2, l);
	status = DftiCommitDescriptor( my_desc1_handle);
	status = DftiComputeForward( my_desc1_handle, x);
	status = DftiFreeDescriptor(&my_desc1_handle);
	/* result is the complex value x[j][k], 0<=j<=31, 0<=k<=99 */
	status = DftiCreateDescriptor( &my_desc2_handle, DFTI_SINGLE,
			  DFTI_REAL, 2, l);
	status = DftiCommitDescriptor( my_desc2_handle);
	status = DftiComputeForward( my_desc2_handle, y);
	status = DftiFreeDescriptor(&my_desc2_handle);
	/* result is the complex value z(j,k) 0<=j<=31; 0<=k<=99
	/* and is stored in CCS format*/
	
	printf("finishing");
}
