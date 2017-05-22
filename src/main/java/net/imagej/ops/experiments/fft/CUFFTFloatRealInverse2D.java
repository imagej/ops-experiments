package net.imagej.ops.experiments.fft;

import static net.imagej.ops.experiments.CudaUtility.checkCudaErrors;
import static org.bytedeco.javacpp.cuda.cudaFree;
import static org.bytedeco.javacpp.cuda.cudaMalloc;
import static org.bytedeco.javacpp.cufftw.FFTW_ESTIMATE;
import static org.bytedeco.javacpp.cufftw.fftwf_destroy_plan;
import static org.bytedeco.javacpp.cufftw.fftwf_execute_dft_c2r;
import static org.bytedeco.javacpp.cufftw.fftwf_plan_dft_c2r_2d;

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
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

@Plugin(type = Ops.Filter.IFFT.class, priority = Priority.LOW_PRIORITY)
public class CUFFTFloatRealInverse2D<C extends ComplexType<C>> extends
		AbstractUnaryFunctionOp<RandomAccessibleInterval<C>, Img<FloatType>>implements Ops.Filter.FFT, Contingent {

	/**
	 * Compute an 2D forward FFT using CUFFT (GPU)
	 */
	@Override
	public Img<FloatType> calculate(final RandomAccessibleInterval<C> in) {

		try {
			

			// get data as a float pointer
			final FloatPointer data = ConvertersUtility.ii2DComplexToFloatPointer(Views.zeroMin(in));

			// move to device
			final FloatPointer p = ConvertersUtility.floatPointerHostToDevice(data,
					(int) (in.dimension(0) * in.dimension(1) * 2));

			// compute size of real signal
			final int[] realSize = new int[] { ((int) in.dimension(0) - 1) * 2, (int) in.dimension(1) };

			final FloatPointer pout = new FloatPointer();

			// malloc memory on device
			CudaUtility.checkCudaErrors(cudaMalloc(pout, realSize[0] * realSize[1] * Float.BYTES));

			// create FFT plan
			final fftwf_plan plan = fftwf_plan_dft_c2r_2d(realSize[0], realSize[1], p, pout, (int) FFTW_ESTIMATE);

			// fftwf_execute(plan);
			fftwf_execute_dft_c2r(plan, p, pout);

			fftwf_destroy_plan(plan);

			final float[] out = new float[(int) (realSize[0] * realSize[1])];

			// move memory back to host
			final FloatPointer host = ConvertersUtility.floatPointerDeviceToHost(pout,
					(int) ((realSize[0]) * realSize[1]));

			host.get(out);
			
			checkCudaErrors( cudaFree(p));
			checkCudaErrors( cudaFree(pout));
			
			FloatPointer.free(host);

			return ArrayImgs.floats(out, new long[] { realSize[0], realSize[1] });

		} catch (final Exception e) {
			System.out.println(e);
			return null;
		}

	}

	@Override
	public boolean conforms() {
		
		try {
			Loader.load(cufftw.class);
		}
		catch(Exception e) {
			return false;
		}

		if (this.in() == null) {
			return true;
		}

		if (this.in().numDimensions() != 2) {
			return false;
		}

		return true;
	}

}
