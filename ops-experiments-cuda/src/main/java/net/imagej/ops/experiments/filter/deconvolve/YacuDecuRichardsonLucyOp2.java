
package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.ops.OpService;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.experiments.filter.AbstractNativeFFTFilterF;
import net.imagej.ops.filter.pad.DefaultPadInputFFT;
import net.imagej.ops.filter.pad.DefaultPadShiftKernelFFT;
import net.imagej.ops.special.function.AbstractBinaryFunctionOp;
import net.imagej.ops.special.function.BinaryFunctionOp;
import net.imagej.ops.special.function.Functions;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.outofbounds.OutOfBoundsConstantValueFactory;
import net.imglib2.outofbounds.OutOfBoundsFactory;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory.Boundary;
import net.imglib2.type.NativeType;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

import org.bytedeco.javacpp.FloatPointer;
import org.scijava.Priority;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

/**
 * Implements Cuda version of Richardson Lucy deconvolution.
 * 
 * @author bnorthan
 * @param <I>
 * @param <O>
 * @param <K>
 * @param <C>
 */
@Plugin(type = Ops.Deconvolve.RichardsonLucy.class, priority = Priority.LOW)
public class YacuDecuRichardsonLucyOp2<I extends RealType<I>, O extends RealType<O> & NativeType<O>, K extends RealType<K>, C extends ComplexType<C> & NativeType<C>>
	extends
	AbstractBinaryFunctionOp<RandomAccessibleInterval<I>, RandomAccessibleInterval<K>, RandomAccessibleInterval<FloatType>>
{

	@Parameter
	LogService log;

	@Parameter
	OpService ops;

	@Parameter
	boolean powerOfTwo;

	/**
	 * Defines the out of bounds strategy for the extended area of the input
	 */
	@Parameter(required = false)
	private OutOfBoundsFactory<I, RandomAccessibleInterval<I>> obfInput;

	/**
	 * Defines the out of bounds strategy for the extended area of the kernel
	 */
	@Parameter(required = false)
	private OutOfBoundsFactory<K, RandomAccessibleInterval<K>> obfKernel;

	@Parameter
	int iterations;

	@Parameter(required = false)
	boolean nonCirculant = false;

	/**
	 * Op used to pad the input
	 */
	private BinaryFunctionOp<RandomAccessibleInterval<I>, Dimensions, RandomAccessibleInterval<I>> padOp;

	/**
	 * Op used to pad the kernel
	 */
	private BinaryFunctionOp<RandomAccessibleInterval<K>, Dimensions, RandomAccessibleInterval<K>> padKernelOp;

	@Override
	public void initialize() {
		YacuDecuRichardsonLucyWrapper.load();
		super.initialize();
	}

	@Override
	public RandomAccessibleInterval<FloatType> calculate(
		RandomAccessibleInterval<I> input, RandomAccessibleInterval<K> kernel)
	{
		// the out of bounds factory will be different depending on whether we
		// are
		// using circulant or non-circulant
		if (!nonCirculant) {
			obfInput = new OutOfBoundsMirrorFactory<>(Boundary.SINGLE);
		}
		else if (nonCirculant) {
			obfInput = new OutOfBoundsConstantValueFactory<>(Util.getTypeFromInterval(
				in()).createVariable());
		}

		obfKernel = new OutOfBoundsConstantValueFactory<>(Util.getTypeFromInterval(
			kernel).createVariable());

		// create output of float type
		Img<FloatType> output = ops().create().img(input, new FloatType());

		final long[] paddedSize = new long[input.numDimensions()];

		// extend based on kernel size
		for (int d = 0; d < input.numDimensions(); ++d) {
			paddedSize[d] = (int) input.dimension(d) + (int) kernel.dimension(d) - 1;
		}

		log.info("original size " + output.dimension(0) + " " + output.dimension(
			1) + " " + output.dimension(2));
		log.info("padded to border size " + paddedSize[0] + " " + paddedSize[1] +
			" " + paddedSize[2]);
		log.info("padded to fft size " + input.dimension(0) + " " + input.dimension(
			1) + " " + input.dimension(2));

		padOp = (BinaryFunctionOp) Functions.binary(ops(), DefaultPadInputFFT.class,
			RandomAccessibleInterval.class, RandomAccessibleInterval.class,
			Dimensions.class, powerOfTwo, obfInput);

		padKernelOp = (BinaryFunctionOp) Functions.binary(ops(),
			DefaultPadShiftKernelFFT.class, RandomAccessibleInterval.class,
			RandomAccessibleInterval.class, Dimensions.class, powerOfTwo);

		// create extended input and kernel
		RandomAccessibleInterval<I> paddedInput = padOp.calculate(input,
			new FinalDimensions(paddedSize));

		RandomAccessibleInterval<K> paddedKernel = padKernelOp.calculate(kernel,
			new FinalDimensions(paddedSize));

		FloatPointer fpInput = null;
		FloatPointer fpKernel = null;
		FloatPointer fpOutput = null;

		// convert image to FloatPointer
		fpInput = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(paddedInput));

		// convert PSF to FloatPointer
		fpKernel = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(
			paddedKernel));

		// get a FloatPointer for the output
		// fpOutput =
		// ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(paddedInput));

		final float mean = ops().stats().mean(Views.iterable(paddedInput))
			.getRealFloat();

		final long bufferSize = paddedInput.dimension(0) * paddedInput.dimension(
			1) * paddedInput.dimension(2);
		fpOutput = new FloatPointer(bufferSize);

		for (int i = 0; i < bufferSize; i++) {
			fpOutput.put(i, mean);
		}

		final long[] fftSize = new long[] { paddedInput.dimension(0) / 2 + 1, paddedInput
			.dimension(1), paddedInput.dimension(2) };

		final FloatPointer X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] *
			fftSize[2]));

		final FloatPointer H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] *
			fftSize[2]));

		FloatPointer normalFP = null;

		// create the normalization factor needed for non-circulant mode
		if (nonCirculant == true) {

			normalFP = CudaDeconvolutionUtility.createNormalizationFactor(ops,
				paddedInput, input, fpKernel, X_, H_);
		}

		final long startTime = System.currentTimeMillis();

		// Call the Cuda wrapper
		YacuDecuRichardsonLucyWrapper.deconv_device(iterations,
			(int) paddedInput.dimension(2), (int) paddedInput.dimension(1),
			(int) paddedInput.dimension(0), fpInput, fpKernel, fpOutput, normalFP);

		final long endTime = System.currentTimeMillis();

		log.info("Total execution time (decon) is: " + (endTime - startTime));
		
		
		
		int imageSize = 1;

		for (int d = 0; d < input.numDimensions(); d++) {
			imageSize *= input.dimension(d);
		}

		// copy output to array
		final float[] arrayOutput = new float[imageSize];
		fpOutput.get(arrayOutput);

		// Pointer.free(fpInput);
		// Pointer.free(fpKernel);
		// Pointer.free(fpOutput);

		long[] imgSize = null;

		if (paddedInput.numDimensions() == 2) {
			imgSize = new long[] { paddedInput.dimension(0), paddedInput.dimension(1) };
		} else {
			imgSize = new long[] { paddedInput.dimension(0), paddedInput.dimension(1), paddedInput.dimension(2) };
		}
		final Img<FloatType> outPadded = ArrayImgs.floats(arrayOutput, imgSize);

		// compute output interval within the padded paddedInput
		final long[] start = new long[paddedInput.numDimensions()];
		final long[] end = new long[paddedInput.numDimensions()];

		for (int d = 0; d < output.numDimensions(); d++) {
			final long offset = (paddedInput.dimension(d) - output.dimension(d)) / 2;
			start[d] = offset;
			end[d] = start[d] + output.dimension(d) - 1;
		}

		// -- copy (crop) padded back to original size
		output = (Img<FloatType>) ops().run(Ops.Copy.RAI.class, output,
				Views.zeroMin(Views.interval(outPadded, new FinalInterval(start, end))));
		
		return output;



	}
}
