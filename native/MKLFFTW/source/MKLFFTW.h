#pragma once

#include "fftw/fftw3.h"
#include "fftw/fftw3_mkl.h"

__declspec(dllexport) void testMKLFFTW(float * x_, float * y_, int width, int height);

__declspec(dllexport) void mklConvolve(float * x, float *h, float * y, float * X_, float * H_, const int width, const int height, bool conj);

__declspec(dllexport) void mklConvolve3D(float * x, float *h, float * y, float * X_, float * H_, const int n0, const int n1, const int n2, bool conj);

__declspec(dllexport) void mklRichardsonLucy3D(int iterations, float * x, float *h, float*y, fftwf_complex* FFT_, fftwf_complex* H_, const int n0, const int n1, const int n2, float * normal);

void testMKLFFT();
