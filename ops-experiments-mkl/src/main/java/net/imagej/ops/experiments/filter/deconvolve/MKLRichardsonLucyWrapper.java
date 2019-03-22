package net.imagej.ops.experiments.filter.deconvolve;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(value = {
	    @Platform(include = {"MKLFFTW.h"},
	              compiler = "fastfpu", includepath = "/opt/intel/mkl/include/", linkpath = {"/opt/intel/lib/", "/opt/intel/mkl/lib/"}, link = {"mkl_rt", "MKLFFTW"},
	              preload = {"iomp5", "mkl_avx", "mkl_avx2", "mkl_avx512", "mkl_avx512_mic", "mkl_def", "mkl_mc", "mkl_mc3", "mkl_core", "mkl_gnu_thread", "mkl_intel_lp64", "mkl_intel_thread"}),
	    @Platform(value = "linux-x86",    linkpath = {"/opt/intel/lib/ia32/", "/opt/intel/mkl/lib/ia32/"}),
	    @Platform(value = "linux-x86_64", linkpath = {"/opt/intel/lib/intel64/", "/opt/intel/mkl/lib/intel64/"}),
	    @Platform(value = "windows", preload = {"libiomp5md", "mkl_avx", "mkl_avx2", "mkl_avx512", "mkl_avx512_mic", "mkl_def", "mkl_mc", "mkl_mc3", "mkl_core", "mkl_intel_lp64", "mkl_intel_thread"},
	                                     includepath = "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/include/"),
	    @Platform(value = "windows-x86",    linkpath = "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/ia32/",
	                                     preloadpath = {"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/ia32/compiler/",
	                                                    "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/ia32/mkl/"}),
	    @Platform(value = "windows-x86_64", linkpath = "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64/",
	                                     preloadpath = {"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/compiler/",
	                                                    "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/redist/intel64/mkl/"}) })
public class MKLRichardsonLucyWrapper {

	static {
		Loader.load();
	}

	public static native void mklRichardsonLucy3D(int iterations, FloatPointer x, FloatPointer h, FloatPointer y, @Cast("fftwf_complex*") FloatPointer FFT_,
			@Cast("fftwf_complex*") FloatPointer H_, int n0, int n1, int n2, FloatPointer normal);

	public static void load() {
		Loader.load();
	};

}
