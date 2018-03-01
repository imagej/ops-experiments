package net.imagej.ops.experiments.filter.convolve;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(value = {
	    @Platform(include = {"MKLFFTW.h","mkl.h", "mkl_version.h", "mkl_types.h", /*"mkl_blas.h",*/ "mkl_trans.h", "mkl_cblas.h", "mkl_spblas.h", /*"mkl_lapack.h",*/ "mkl_lapacke.h",
	        "mkl_dss.h", "mkl_pardiso.h", "mkl_sparse_handle.h", "mkl_service.h", "mkl_rci.h", "mkl_vml.h", "mkl_vml_defines.h", "mkl_vml_types.h", "mkl_vml_functions.h",
	        "mkl_vsl.h", "mkl_vsl_defines.h", "mkl_vsl_types.h", "mkl_vsl_functions.h", "mkl_df.h", "mkl_df_defines.h", "mkl_df_types.h", "mkl_df_functions.h",
	        "mkl_dfti.h", "mkl_trig_transforms.h", "mkl_poisson.h", "mkl_solvers_ee.h", /*"mkl_direct_types.h", "mkl_direct_blas.h", "mkl_direct_lapack.h", "mkl_direct_call.h",*/
	        "mkl_dnn_types.h", "mkl_dnn.h", /*"mkl_blacs.h", "mkl_pblas.h", "mkl_scalapack.h", "mkl_cdft_types.h", "mkl_cdft.h", "i_malloc.h" */},
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
public class MKLConvolveWrapper {
	static {
		Loader.load();
	}
	
	public static native void mklConvolve
	(FloatPointer x, FloatPointer h, FloatPointer y, FloatPointer X_, FloatPointer H_, int width, int height, boolean conj);
	
	public static void load() {
		Loader.load();
	};
}
