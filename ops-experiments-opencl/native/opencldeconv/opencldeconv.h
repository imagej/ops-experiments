#pragma once

extern "C" {
  void test();
  int conv(size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_out);
  int deconv(int iterations, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_out, float * normal);
  int deconv_long(int iterations, size_t N0, size_t N1, size_t N2, long d_image, long d_psf, long d_update, long d_normal, long l_context, long l_queuee, long l_device); 
  int fft2d(size_t N1, size_t N2, float *h_image, float * h_out);
  int fft2d_long(long N1, long N2, long h_image, long h_out, long l_context, long l_queue);
  int fftinv2d(size_t N1, size_t N2, float *h_fft, float * h_out);
}


