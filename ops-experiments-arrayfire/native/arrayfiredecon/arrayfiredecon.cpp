#include <stdio.h>
#include <iostream>
#include "arrayfiredecon.h"

#include <arrayfire.h>
#include <af/util.h>

void test() {
  printf("Test arrayfire entry point\n");
}

int arrayTest( int n, float * f) {
    
    af::array a = af::array(n,f);
    // Sum the values and copy the result to the CPU:
    double sum = af::sum<float>(a);
    printf("sum: %g\n", sum);

}

int conv(size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_out) {
  printf("\nEntering Convolution");

  af::array a_image = af::array(N1, N2, N3, h_image);
  af::array a_psf = af::array(N1, N2, N3, h_psf);
  
  float sum_image = af::sum<float>(a_image);
  float sum_psf = af::sum<float>(a_psf);
  
  printf("sum image: %g\n", sum_image);
  printf("sum psf: %g\n", sum_psf);
 
  af::array convolved=af::fftConvolve3(a_image, a_psf);
  float sum_convolved = af::sum<float>(convolved);
  printf("sum convolved: %g\n", sum_convolved);
 
  convolved.host(h_out);
  
  return 0;
}

int conv2(size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_out) {
  printf("\nEntering Convolution 2\n");

  af::array a_image = af::array(N1, N2, N3, h_image);
  af::array a_psf = af::array(N1, N2, N3, h_psf);
  
  float sum_image = af::sum<float>(a_image);
  float sum_psf = af::sum<float>(a_psf);
  
  printf("sum image: %g\n", sum_image);
  printf("sum psf: %g\n", sum_psf);
  
  af::array fft1=af::fftR2C<3>(a_image);
  af::array fft2=af::fftR2C<3>(a_psf);
  fft1 = fft1*fft2;
  af::array ifft = af::fftC2R<3>(fft1,false, 1./(double)(N1*N2*N3));
  ifft.host(h_out);

  return 0;
}

int deconv(unsigned int iter, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_object, float * h_normal) {
    printf("\nEntering Decon\n");

    af::array a_image = af::array(N1, N2, N3, h_image);
    af::array a_object = af::array(N1, N2, N3, h_object);
    af::array a_psf = af::array(N1, N2, N3, h_psf);
  
    af::array fft2=af::fftR2C<3>(a_psf);
    af::array fft2_ = af::conjg(fft2); 

    printf("fft dims %d %d %d\n",fft2.dims()[0], fft2.dims()[1], fft2.dims()[2]);

    printf("fft2 is complex %d\n",fft2.iscomplex());
    //printf("fft2 is complex %d\n",fft2.);
    
    for (int i=0;i<iter;i++) {
      printf("Array fire RL iteration %d\n", i);
      
      // reblur current estimate 
      af::array fft1=af::fftR2C<3>(a_object);
      fft1 = fft1*fft2;
      af::array reblurred = af::fftC2R<3>(fft1,false,1./(double)(N1*N2*N3));

      // divide observed image by reblurred
      af::array div = a_image/reblurred;
      
      // correlate with PSF to get update factor
      fft1=af::fftR2C<3>(div);
      fft1= fft1*fft2_;
      af::array update = af::fftC2R<3>(fft1,false, 1./(double)(N1*N2*N3));
      
      // update object 
      a_object=update*a_object;
    }
    
    a_object.host(h_object);
    
    return 0;
}

  
