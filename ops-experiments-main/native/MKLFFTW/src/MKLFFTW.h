#pragma once

#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

#include "fftw/fftw3.h"
#include "fftw/fftw3_mkl.h"

extern "C" EXPORT void testMKLFFTW(float * x_, float * y_, int width, int height);

extern "C" EXPORT void mklConvolve(float * x, float *h, float * y, float * X_, float * H_, const int width, const int height, bool conj);

extern "C" EXPORT void mklConvolve3D(float * x, float *h, float * y, float * X_, float * H_, const int n0, const int n1, const int n2, bool conj);

extern "C" EXPORT void mklRichardsonLucy3D(int iterations, float * x, float *h, float*y, fftwf_complex* FFT_, fftwf_complex* H_, const int n0, const int n1, const int n2, float * normal);

void testMKLFFT();
