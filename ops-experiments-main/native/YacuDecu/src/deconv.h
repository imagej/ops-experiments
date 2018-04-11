#pragma once

extern "C" {
	int deconv_device(unsigned int iter, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_object, float * h_normal);
	int deconv_host(unsigned int iter, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_object, float * h_normal);
	int deconv_stream(unsigned int iter, size_t N1, size_t N2, size_t N3, float *h_image, float *h_psf, float *h_object, float * h_normal);
}
