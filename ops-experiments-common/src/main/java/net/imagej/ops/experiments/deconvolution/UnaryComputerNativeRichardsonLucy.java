
package net.imagej.ops.experiments.deconvolution;

import net.imagej.ops.OpService;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.filter.pad.DefaultPadInputFFT;
import net.imagej.ops.filter.pad.DefaultPadShiftKernelFFT;
import net.imagej.ops.special.computer.AbstractUnaryComputerOp;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.outofbounds.OutOfBoundsConstantValueFactory;
import net.imglib2.outofbounds.OutOfBoundsFactory;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory.Boundary;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

import org.bytedeco.javacpp.FloatPointer;
import org.scijava.Priority;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

/**
 * This class implements the workflow to call a native (ie c, or c++ implementation using MKL, Cuda, OpenCL etc.)
 * version of Richardson Lucy.  
 * 
 * Caller must Set 'NativeRichardsonLucy' Parameter to desired implementation.  
 *
 * Documentation for the (Optionally) Non-circulant version of Richardson Lucy can be found here
 * http://bigwww.epfl.ch/deconvolution/challenge/index.html?p=documentation/theory/richardsonlucy
 * 
 * @author bnorthan
 */
@Plugin(type = Ops.Deconvolve.RichardsonLucy.class,
	priority = Priority.EXTREMELY_HIGH)
public class UnaryComputerNativeRichardsonLucy<I extends RealType<I>, O extends RealType<O>, K extends RealType<K>, C extends ComplexType<C>>
	extends
	AbstractUnaryComputerOp<RandomAccessibleInterval<I>, RandomAccessibleInterval<O>>
{

	@Parameter
	OpService ops;

	@Parameter
	UIService ui;

	@Parameter
	LogService log;

	@Parameter
	RandomAccessibleInterval<K> psf;

	@Parameter
	int iterations;

	@Parameter(required = false)
	boolean nonCirculant = true;

	@Parameter(required = false)
	NativeRichardsonLucy rl;

	OutOfBoundsFactory<I, RandomAccessibleInterval<I>> obfInput;

	@SuppressWarnings("unchecked")
	@Override
	public void compute(final RandomAccessibleInterval<I> input,
		final RandomAccessibleInterval<O> output)
	{

		// compute extended size of the image based on PSF dimensions
		final long[] extendedSize = new long[output.numDimensions()];

		for (int d = 0; d < output.numDimensions(); d++) {
			extendedSize[d] = output.dimension(d) + psf.dimension(d);
		}

		// if non circulant extend image with zeros
		if (nonCirculant) {
			obfInput = new OutOfBoundsConstantValueFactory<>(Util.getTypeFromInterval(
				input).createVariable());
		}
		// otherwise extend mirror
		else {
			obfInput = new OutOfBoundsMirrorFactory<I, RandomAccessibleInterval<I>>(
				Boundary.SINGLE);
		}

		RandomAccessibleInterval<I> paddedInput = (RandomAccessibleInterval<I>) ops
			.run(DefaultPadInputFFT.class, input, new FinalDimensions(extendedSize),
				false, obfInput);

		RandomAccessibleInterval<K> paddedPSF = (RandomAccessibleInterval<K>) ops
			.run(DefaultPadShiftKernelFFT.class, psf, new FinalDimensions(
				extendedSize), false);

		Img<FloatType> deconv = runFilter(paddedInput, paddedPSF, output);

		Interval interval = Intervals.createMinMax(-paddedInput.min(0), -paddedInput
			.min(1), -paddedInput.min(2), -paddedInput.min(0) + input.dimension(0) -
				1, -paddedInput.min(1) + input.dimension(1) - 1, -paddedInput.min(2) +
					input.dimension(2) - 1);

		// -- copy (crop) padded back to original size
		ops().run(Ops.Copy.RAI.class, output, Views.zeroMin(Views.interval(deconv,
			interval)));

	}

	Img<FloatType> runFilter(RandomAccessibleInterval<I> paddedInput,
		RandomAccessibleInterval<K> paddedPSF, Dimensions originalDimensions)
	{

		rl.loadLibrary();

		FloatPointer fpInput = null;
		FloatPointer fpPSF = null;
		FloatPointer fpOutput = null;

		// convert image to FloatPointer
		fpInput = ConvertersUtility.ii3DToFloatPointer(Views.iterable((Views
			.zeroMin(paddedInput))));

		// convert PSF to FloatPointer
		fpPSF = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(paddedPSF));

		int paddedSize = (int) (paddedInput.dimension(0) * paddedInput.dimension(
			1) * paddedInput.dimension(2));

		if (nonCirculant) {
			// create output
			fpOutput = new FloatPointer(paddedSize);

			// compute sum of image and divide by padded size
			float meanOverPaddedSize = ops.stats().sum(Views.iterable(paddedInput))
				.getRealFloat() / paddedSize;

			// set first guess to flat sheet
			for (int i = 0; i < paddedSize; i++) {
				fpOutput.put(i, meanOverPaddedSize);
			}
		}
		else {
			fpOutput = ConvertersUtility.ii3DToFloatPointer(Views.iterable((Views
				.zeroMin(paddedInput))));
		}

		// create normal
		FloatPointer normalFP = rl.createNormal(paddedInput, originalDimensions,
			fpPSF);

		final long startTime = System.currentTimeMillis();

		// Call the decon
		int error = rl.callRichardsonLucy(iterations, paddedInput, fpInput, fpPSF, fpOutput, normalFP);

		if (error>0) {
			log.error("YacuDecu returned error code "+error);
		}
		
		// copy output to array
		final float[] arrayOutput = new float[paddedSize];
		fpOutput.get(arrayOutput);

		final Img<FloatType> deconv = ArrayImgs.floats(arrayOutput, new long[] {
			paddedInput.dimension(0), paddedInput.dimension(1), paddedInput.dimension(
				2) });

		return deconv;
	}
}
