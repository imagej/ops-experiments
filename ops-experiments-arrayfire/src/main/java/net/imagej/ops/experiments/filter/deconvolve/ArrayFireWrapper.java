
package net.imagej.ops.experiments.filter.deconvolve;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(value = { @Platform(include = "arrayfiredecon.h", linkpath = {"/opt/arrayfire/lib64/"}, link = {
	"arrayfiredecon_cpu" },
//preload = {"iomp5", "mkl_avx", "mkl_avx2", "mkl_avx512", "mkl_def", "mkl_mc", "mkl_mc3", "mkl_core", "mkl_gnu_thread", "mkl_intel_lp64"}) 
preload = { "afcpu", "mkl_avx", "mkl_avx2", "mkl_avx512", "mkl_def", "mkl_mc", "mkl_mc3", "mkl_core", "mkl_gnu_thread", "mkl_intel_lp64"}) 
})
public class ArrayFireWrapper {

	static {
		Loader.load();
	}

	public static native int conv2(long N1, long N2, long N3,
		FloatPointer h_image, FloatPointer h_psf, FloatPointer h_out);

	public static void load() {
		Loader.load();
	};

}
