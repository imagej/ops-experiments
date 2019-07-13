
package net.imagej.ops.experiments.filter.convolve;

import static org.bytedeco.javacpp.fftw3.FFTW_ESTIMATE;
import static org.bytedeco.javacpp.fftw3.fftwf_destroy_plan;
import static org.bytedeco.javacpp.fftw3.fftwf_execute;
import static org.bytedeco.javacpp.fftw3.fftwf_plan_dft_c2r_2d;
import static org.bytedeco.javacpp.fftw3.fftwf_plan_dft_r2c_2d;

import net.imagej.ops.OpService;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.filter.pad.DefaultPadInputFFT;
import net.imagej.ops.filter.pad.DefaultPadShiftKernelFFT;
import net.imagej.ops.special.computer.AbstractBinaryComputerOp;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.outofbounds.OutOfBoundsConstantValueFactory;
import net.imglib2.outofbounds.OutOfBoundsFactory;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory.Boundary;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

import org.apache.commons.math3.complex.Complex;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.fftw3.fftwf_plan;
import org.scijava.Priority;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

// WIP work in progress showing how to convolve using JavaCpp FFTW wrappers....
@Plugin(type = Ops.Filter.Convolve.class, priority = Priority.LOW_PRIORITY)
public class JavaCPPConvolveOp<I extends RealType<I>, O extends RealType<O> & NativeType<O>, K extends RealType<K>, C extends ComplexType<C> & NativeType<C>>
	extends
	AbstractBinaryComputerOp<RandomAccessibleInterval<I>, RandomAccessibleInterval<K>, RandomAccessibleInterval<O>>
{

	@Parameter
	OpService ops;

	@Parameter
	boolean correlate = false;

	@Parameter(required = false)
	OutOfBoundsFactory<I, RandomAccessibleInterval<I>> obfInput;

	@Parameter(required = false)
	OutOfBoundsFactory<I, RandomAccessibleInterval<I>> obfKernel;

	@Override
	public void compute(RandomAccessibleInterval<I> input,
		RandomAccessibleInterval<K> psf, RandomAccessibleInterval<O> output)
	{
		// compute extended size of the image based on PSF dimensions
		final long[] extendedSize = new long[output.numDimensions()];

		for (int d = 0; d < output.numDimensions(); d++) {
			extendedSize[d] = output.dimension(d) + psf.dimension(d);
		}

		// if non circulant extend image with zeros
		if (obfInput == null) {
			obfInput = new OutOfBoundsConstantValueFactory<>(Util.getTypeFromInterval(
				input).createVariable());
		}

		if (obfKernel == null) {
			obfKernel = new OutOfBoundsMirrorFactory<I, RandomAccessibleInterval<I>>(
				Boundary.SINGLE);
		}

		RandomAccessibleInterval<I> paddedInput = (RandomAccessibleInterval<I>) ops
			.run(DefaultPadInputFFT.class, input, new FinalDimensions(extendedSize),
				false, obfInput);

		RandomAccessibleInterval<K> paddedPSF = (RandomAccessibleInterval<K>) ops
			.run(DefaultPadShiftKernelFFT.class, psf, new FinalDimensions(
				extendedSize), false);

		long n;
		FloatPointer X_, H_;

		if (paddedInput.numDimensions() == 2) {
			final long[] fftSize = new long[] { paddedInput.dimension(0) / 2 + 1,
				paddedInput.dimension(1) };

			n = fftSize[0] * fftSize[1];

			FloatPointer fpInput = null;
			FloatPointer fpPSF = null;

			// convert image to FloatPointer
			fpInput = ConvertersUtility.ii2DToFloatPointer(Views.iterable((Views
				.zeroMin(paddedInput))));

			// convert PSF to FloatPointer
			fpPSF = ConvertersUtility.ii2DToFloatPointer(Views.zeroMin(paddedPSF));

			// create output
			FloatPointer fpOutput = new FloatPointer(n);

			X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1]));

			H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1]));

			// create FFT plan
			final fftwf_plan forward1 = fftwf_plan_dft_r2c_2d((int) paddedInput
				.dimension(0), (int) paddedInput.dimension(1), fpInput, X_,
				(int) FFTW_ESTIMATE);

			final fftwf_plan forward2 = fftwf_plan_dft_r2c_2d((int) paddedInput
				.dimension(0), (int) paddedInput.dimension(1), fpPSF, H_,
				(int) FFTW_ESTIMATE);

			final fftwf_plan inverse = fftwf_plan_dft_c2r_2d((int) paddedInput
				.dimension(0), (int) paddedInput.dimension(1), X_, fpOutput,
				(int) FFTW_ESTIMATE);

			fftwf_execute(forward1);
			fftwf_execute(forward2);

			// use MKL to perform complex multiply
			// vcMul((int)n, X_, H_, X_);

			// for some reason vcMul crashed... until we figure that out
			// do a quick (to implement but slow to run) and dirty complex multiply
			for (int i = 0; i < 2 * n; i += 2) {
				Complex img = new Complex(X_.get(i), X_.get(i + 1));
				Complex h = new Complex(H_.get(i), H_.get(i + 1));

				Complex y = img.multiply(h);

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
		else {
			final long[] fftSize = new long[] { paddedInput.dimension(0) / 2 + 1,
				paddedInput.dimension(1), paddedInput.dimension(2) };

			n = fftSize[0] * fftSize[1] * fftSize[2];

			X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));

			H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));
		}

	}

}
