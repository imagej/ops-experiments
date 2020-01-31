#pragma once

#ifdef _WIN64
  __declspec(dllexport) void test();
  __declspec(dllexport) void arrayTest( int n, float * a);
  __declspec(dllexport) int conv(size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_out);
  __declspec(dllexport)int conv2(size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_out);
  __declspec(dllexport) int deconv(unsigned int iter, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_object, float * h_normal);
#else
  extern "C" {
    void test();
    void arrayTest( int n, float * a);
    int conv(size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_out);
    int conv2(size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_out);
    int deconv(unsigned int iter, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_object, float * h_normal);
}
#endif

