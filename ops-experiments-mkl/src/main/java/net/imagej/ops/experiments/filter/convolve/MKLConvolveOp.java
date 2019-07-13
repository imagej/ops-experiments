
package net.imagej.ops.experiments.filter.convolve;

import net.imagej.ops.OpService;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.filter.pad.DefaultPadInputFFT;
import net.imagej.ops.filter.pad.DefaultPadShiftKernelFFT;
import net.imagej.ops.special.computer.AbstractBinaryComputerOp;
import net.imglib2.FinalDimensions;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.outofbounds.OutOfBoundsConstantValueFactory;
import net.imglib2.outofbounds.OutOfBoundsFactory;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory.Boundary;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

import org.bytedeco.javacpp.FloatPointer;
import org.scijava.Priority;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

@Plugin(type = Ops.Filter.Convolve.class, priority = Priority.LOW_PRIORITY)
public class MKLConvolveOp<I extends RealType<I>, O extends RealType<O> & NativeType<O>, K extends RealType<K>, C extends ComplexType<C> & NativeType<C>>
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

		MKLConvolveWrapper.load();

		FloatPointer X_ = null;

		FloatPointer H_ = null;

		int paddedSize;
		FloatPointer fpOutput = null;

		if (input.numDimensions() == 2) {
			final long[] fftSize = new long[] { input.dimension(0) / 2 + 1, input
				.dimension(1) };

			X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1]));

			H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1]));

			FloatPointer fpInput = null;
			FloatPointer fpPSF = null;

			// convert image to FloatPointer
			fpInput = ConvertersUtility.ii2DToFloatPointer(Views.iterable((Views
				.zeroMin(paddedInput))));

			// convert PSF to FloatPointer
			fpPSF = ConvertersUtility.ii2DToFloatPointer(Views.zeroMin(paddedPSF));

			paddedSize = (int) (paddedInput.dimension(0) * paddedInput.dimension(1));

			// create output
			fpOutput = new FloatPointer(paddedSize);

			// Call the MKL wrapper
			MKLConvolveWrapper.mklConvolve(fpInput, fpPSF, fpOutput, X_, H_,
				(int) paddedInput.dimension(1), (int) paddedInput.dimension(0),
				correlate);

		}
		else {
			final long[] fftSize = new long[] { input.dimension(0) / 2 + 1, input
				.dimension(1), input.dimension(2) };

			FloatPointer fpInput = null;
			FloatPointer fpPSF = null;

			// convert image to FloatPointer
			fpInput = ConvertersUtility.ii2DToFloatPointer(Views.iterable((Views
				.zeroMin(paddedInput))));

			// convert PSF to FloatPointer
			fpPSF = ConvertersUtility.ii2DToFloatPointer(Views.zeroMin(paddedPSF));

			paddedSize = (int) (paddedInput.dimension(0) * paddedInput.dimension(1) *
				paddedInput.dimension(2));

			// create output
			fpOutput = new FloatPointer(paddedSize);

			// Call the MKL wrapper
			MKLConvolve3DWrapper.mklConvolve3D(fpInput, fpPSF, fpOutput,
				(int) paddedInput.dimension(2), (int) paddedInput.dimension(1),
				(int) paddedInput.dimension(0), correlate);

		}

		// copy output to array
		final float[] arrayOutput = new float[paddedSize];
		fpOutput.get(arrayOutput);

		final Img<FloatType> convolved = ArrayImgs.floats(arrayOutput, new long[] {
			paddedInput.dimension(0), paddedInput.dimension(1), paddedInput.dimension(
				2) });

		Interval interval = Intervals.createMinMax(-paddedInput.min(0), -paddedInput
			.min(1), -paddedInput.min(2), -paddedInput.min(0) + input.dimension(0) -
				1, -paddedInput.min(1) + input.dimension(1) - 1, -paddedInput.min(2) +
					input.dimension(2) - 1);

		// -- copy (crop) padded back to original size
		ops().run(Ops.Copy.RAI.class, output, Views.zeroMin(Views.interval(
			convolved, interval)));

		// Pointer.free(fpInput);
		// Pointer.free(fpKernel);
		// Pointer.free(fpOutput);

	}
}
