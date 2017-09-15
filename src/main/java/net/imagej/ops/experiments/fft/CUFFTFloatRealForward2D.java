package net.imagej.ops.experiments.fft;

import static org.bytedeco.javacpp.cuda.cudaMalloc;
import static org.bytedeco.javacpp.cuda.cudaFree;
import static org.bytedeco.javacpp.cufftw.FFTW_ESTIMATE;
import static org.bytedeco.javacpp.cufftw.fftwf_destroy_plan;
import static org.bytedeco.javacpp.cufftw.fftwf_execute_dft_r2c;
import static org.bytedeco.javacpp.cufftw.fftwf_plan_dft_r2c_2d;

import static net.imagej.ops.experiments.CudaUtility.checkCudaErrors;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.cufftw;
import org.bytedeco.javacpp.cufftw.fftwf_plan;
import org.scijava.Priority;
import org.scijava.plugin.Plugin;

import net.imagej.ops.Contingent;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.experiments.CudaUtility;
import net.imagej.ops.special.function.AbstractUnaryFunctionOp;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.complex.ComplexFloatType;
import net.imglib2.view.Views;



@Plugin(type = Ops.Filter.IFFT.class, priority = Priority.LOW_PRIORITY)
public class CUFFTFloatRealForward2D<C extends ComplexType<C>>
		extends AbstractUnaryFunctionOp<RandomAccessibleInterval<C>, Img<ComplexFloatType>>
		implements Ops.Filter.FFT, Contingent {

	/**
	 * Compute an 2D forward FFT using CUFFT (GPU)
	 */
	@Override
	public Img<ComplexFloatType> calculate(final RandomAccessibleInterval<C> in) {

		try {
			
			String libPathProperty = System.getProperty("java.library.path");
	        System.out.println(libPathProperty);

			// convert to device (cuda) FloatPointer
			final FloatPointer p = ConvertersUtility.ii2DToDeviceFloatPointer(Views.zeroMin(in));

			// output size of FFT see
			// http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html
			final long[] fftSize = new long[] { in.dimension(0) / 2 + 1, in.dimension(1) };

			final FloatPointer pout = new FloatPointer();

			// mall memory on device
			checkCudaErrors(cudaMalloc(pout, 2 * fftSize[0] * fftSize[1] * Float.BYTES));

			// create FFT plan
			final fftwf_plan plan = fftwf_plan_dft_r2c_2d((int) in.dimension(0), (int) in.dimension(1), p, pout,
					(int) FFTW_ESTIMATE);

			// fftwf_execute(plan);
			fftwf_execute_dft_r2c(plan, p, pout);
			fftwf_destroy_plan(plan);

			final float[] out = new float[(int) (2 * (fftSize[0]) * fftSize[1])];

			final FloatPointer host = ConvertersUtility.floatPointerDeviceToHost(pout,
					(int) (2 * (fftSize[0]) * fftSize[1]));

			host.get(out);
			
			checkCudaErrors( cudaFree(p));
			checkCudaErrors( cudaFree(pout));

			FloatPointer.free(host);

			return ArrayImgs.complexFloats(out, fftSize);

		} catch (final Exception e) {
			System.out.println(e);
			return null;
		}

	}

	@Override
	public boolean conforms() {
		try {
			Loader.load(cufftw.class);
		} catch (Exception e) {
			return false;
		}

		if (this.in().numDimensions() != 2) {
			return false;
		}

		return true;
	}

}
