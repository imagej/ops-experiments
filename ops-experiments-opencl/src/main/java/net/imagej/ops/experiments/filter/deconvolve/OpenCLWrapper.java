
package net.imagej.ops.experiments.filter.deconvolve;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(value = { @Platform(include = "opencldeconv.h", link = { "opencldeconv"}), @Platform(value = "windows-x86_64", linkpath = {
		"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/lib/x64/" }),
	@Platform(value = "linux-x86_64",
		includepath = "/usr/local/cuda-10.0/include/", linkpath = {
			"/usr/local/cuda-10.0/lib64/" }) })
public class OpenCLWrapper {

	static {
		Loader.load();
	}

	public static native long fft2d_long(long N1, long N2, long inPointer, long outPointer, long contextPointer, long queuePointer);

	public static void load() {
		Loader.load();
	};

}
