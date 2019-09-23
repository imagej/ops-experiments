
package net.imagej.ops.experiments.filter.deconvolve;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(value = { @Platform(include = "deconv.h", link = { "YacuDecu",
	"cudart", "cufft" }), @Platform(value = "windows-x86_64", linkpath = {
		"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/lib/x64/" }),
	@Platform(value = "linux-x86_64",
		includepath = "/usr/local/cuda-10.0/include/", linkpath = {
			"/usr/local/cuda-10.0/lib64/" }) })
public class YacuDecuRichardsonLucyWrapper {

	static {
		Loader.load();
	}

	public static native int deconv_device(int iter, int n1, int n2, int n3,
		FloatPointer image, FloatPointer psf, FloatPointer object,
		FloatPointer normal);

	public static native int conv_device(int n1, int n2, int n3,
		FloatPointer image, FloatPointer psf, FloatPointer out, int correlate);

	public static native void setDevice(int device);
	
	public static native int getDeviceCount();

	public static native long getWorkSize(int N1, int N2, int N3);
	
	public static native long getTotalMem();
	
	public static native long getFreeMem();

	public static native void removeSmallValues(FloatPointer in, long size);
	


	public static void load() {
		Loader.load();
	};

}
