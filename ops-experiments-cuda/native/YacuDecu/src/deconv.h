#pragma once

extern "C" {
	int deconv_device(unsigned int iter, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_object, float * h_normal);
	int deconv_host(unsigned int iter, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_object, float * h_normal);
	int deconv_stream(unsigned int iter, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_object, float * h_normal);
	int conv_device(size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_out, unsigned int correlate); 
	int setDevice(int device);
	int getDeviceCount();
	long long getWorkSize(size_t N1, size_t N2, size_t N3);
	long long getTotalMem();
	long long getFreeMem();
	void removeSmallValues(float * in, long long size);
}
