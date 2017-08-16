package net.imagej.ops.experiments.fft;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;


/*
@Properties( value = {
		@Platform(include="MKLFFTW.h", link={"MKLFFTW", "libiomp5md", "mkl_core", "mkl_intel_lp64", "mkl_intel_thread"},
				preload = {"iomp5", "mkl_avx", "mkl_avx2", "mkl_avx512_mic", "mkl_def", "mkl_mc3", "mkl_core", "mkl_gnu_thread", "mkl_intel_lp64", "mkl_intel_thread"}),
	   
	    @Platform(value = "windows-x86_64", 
	    
	    linkpath = {"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64/", 
	    		"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2017/windows/compiler/lib/intel64_win"}) 
		})*/
@Properties( value = {
		@Platform(include="MKLFFTW.h", link={"MKLFFTW", "libiomp5md", "mkl_core", "mkl_intel_lp64", "mkl_intel_thread"}),
	    @Platform(value = "windows-x86_64", 
	    linkpath = {"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64/", 
	    		"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2017/windows/compiler/lib/intel64_win"}
	    ,includepath = "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/include") 
		})
public class MKLFFTWFloatRealForward2DWrapper {
	static {
		Loader.load();
	}
	
	public static native void testMKLFFTW
	(FloatPointer in, FloatPointer out, int width, int height);
	
	/*public static native void mklConvolve
	(FloatPointer x, FloatPointer h, FloatPointer X_, FloatPointer H_, int width, int height);*/
	
	public static void load() {
		Loader.load();
	};

}
