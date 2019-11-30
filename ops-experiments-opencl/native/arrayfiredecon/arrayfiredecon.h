#pragma once

extern "C" {
  void test();
  int arrayTest( int n, float * a);
  int conv(size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_out);
  int conv2(size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_out);
  
  int deconv(unsigned int iter, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_object, float * h_normal);
}

