package net.imagej.ops.experiments.filter.convolve;

import static org.bytedeco.javacpp.fftw3.FFTW_ESTIMATE;
import static org.bytedeco.javacpp.fftw3.fftwf_destroy_plan;
import static org.bytedeco.javacpp.fftw3.fftwf_execute;
import static org.bytedeco.javacpp.fftw3.fftwf_plan_dft_c2r_2d;
import static org.bytedeco.javacpp.fftw3.fftwf_plan_dft_r2c_2d;

import net.imagej.ops.Ops;
import net.imagej.ops.experiments.filter.AbstractNativeFFTFilterF;
import net.imglib2.Interval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;

import org.apache.commons.math3.complex.Complex;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.fftw3.fftwf_plan;
import org.scijava.Priority;
import org.scijava.plugin.Plugin;

@Plugin(type = Ops.Filter.Convolve.class, priority = Priority.LOW_PRIORITY)
public class JavaCPPConvolveOp<I extends RealType<I>, O extends RealType<O> & NativeType<O>, K extends RealType<K>, C extends ComplexType<C> & NativeType<C>>
		extends AbstractNativeFFTFilterF<I, O, K, C> {

	@Override
	protected void loadNativeLibraries() {
		MKLConvolveWrapper.load();
	}

	@Override
	protected void runNativeFilter(Interval inputInterval, Interval outputInterval, FloatPointer input, FloatPointer kernel, FloatPointer output) {

		FloatPointer X_ = null;
		FloatPointer H_ = null;
		long n = -1;

		if (inputInterval.numDimensions() == 2) {
			final long[] fftSize = new long[] { inputInterval.dimension(0) / 2 + 1, inputInterval.dimension(1) };

			n = fftSize[0] * fftSize[1];

			X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1]));

			H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1]));

		} else {
			final long[] fftSize = new long[] { inputInterval.dimension(0) / 2 + 1, inputInterval.dimension(1),
					inputInterval.dimension(2) };

			n = fftSize[0] * fftSize[1] * fftSize[2];

			X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));

			H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));
		}

		// create FFT plan
		final fftwf_plan forward1 = fftwf_plan_dft_r2c_2d((int) inputInterval.dimension(0), (int) inputInterval.dimension(1),
				input, X_, (int) FFTW_ESTIMATE);

		final fftwf_plan forward2 = fftwf_plan_dft_r2c_2d((int) inputInterval.dimension(0), (int) inputInterval.dimension(1),
				kernel, H_, (int) FFTW_ESTIMATE);

		final fftwf_plan inverse = fftwf_plan_dft_c2r_2d((int) inputInterval.dimension(0), (int) inputInterval.dimension(1),
				X_, output, (int) FFTW_ESTIMATE);

		fftwf_execute(forward1);
		fftwf_execute(forward2);

		// use MKL to perform complex multiply
		// vcMul((int)n, X_, H_, X_);
		
		// for some reason vcMul crashed... until we figure that out
		// do a quick (to implement but slow to run) and dirty complex multiply
		for (int i = 0; i < 2 * n; i += 2) {
			Complex img = new Complex(X_.get(i), X_.get(i+1));
			Complex h = new Complex(H_.get(i), H_.get(i + 1));

			Complex y=img.multiply(h);

			X_.put(i, (float) y.getReal());
			X_.put(i + 1, (float) y.getImaginary());
		}

		fftwf_execute(inverse);

		// Pointer.free(fpInput);
		// Pointer.free(fpKernel);
		// Pointer.free(fpOutput);

		fftwf_destroy_plan(forward1);
		fftwf_destroy_plan(forward2);
		fftwf_destroy_plan(inverse);

	}

}
