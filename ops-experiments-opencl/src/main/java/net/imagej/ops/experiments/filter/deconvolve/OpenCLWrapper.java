
package net.imagej.ops.experiments.filter.deconvolve;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(value = { @Platform(include = "opencldeconv.h", link = {
	"opencldeconv", "clFFT" }), 
	@Platform(value = "windows-x86_64", 
	linkpath = {"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/lib/x64/", "C:/Users/bnort/OpenCL/clFFT-2.12.2-Windows-x64/lib64/import/"},
	preloadpath = {"C:/Users/bnort/OpenCL/clFFT-2.12.2-Windows-x64/bin/"},
	preload = {"clFFT" }
			),
	@Platform(value = "linux-x86_64",
		includepath = "/usr/local/cuda-10.0/include/", linkpath = {
			"/usr/local/cuda-10.0/lib64/" }) })
public class OpenCLWrapper {

	static {
		Loader.load();
	}

	public static native long fft2d_long(long N1, long N2, long inPointer,
		long outPointer, long contextPointer, long queuePointer);

	public static native int deconv_long(int iterations, long N0, long N1,
		long N2, long d_image, long d_psf, long d_update, long d_normal,
		long l_context, long l_queuee, long l_device);

	public static void load() {
		Loader.load();
	};

}
