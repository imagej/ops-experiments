package net.imagej.ops.experiments.filter;

import net.imagej.ops.Ops;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.filter.AbstractFilterF;
import net.imagej.ops.filter.pad.DefaultPadInputFFT;
import net.imagej.ops.filter.pad.DefaultPadShiftKernelFFT;
import net.imagej.ops.special.function.BinaryFunctionOp;
import net.imagej.ops.special.function.Functions;
import net.imglib2.Dimensions;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;

/**
 * Abstract class for Native FFT filters based on the JavaCpp framework
 * @author bnorthan
 *
 * @param <I>
 * @param <O>
 * @param <K>
 * @param <C>
 */
public abstract class AbstractNativeFFTFilterF<I extends RealType<I>, O extends RealType<O> & NativeType<O>, K extends RealType<K>, C extends ComplexType<C> & NativeType<C>>
		extends AbstractFilterF<I, O, K, C> {

	@Override
	@SuppressWarnings({ "unchecked", "rawtypes" })
	public void initialize() {
		super.initialize();

		/**
		 * Op used to pad the input
		 */
		setPadOp((BinaryFunctionOp) Functions.binary(ops(), DefaultPadInputFFT.class, RandomAccessibleInterval.class,
				RandomAccessibleInterval.class, Dimensions.class, true, getOBFInput()));

		/**
		 * Op used to pad the kernel
		 */
		setPadKernelOp((BinaryFunctionOp) Functions.binary(ops(), DefaultPadShiftKernelFFT.class,
				RandomAccessibleInterval.class, RandomAccessibleInterval.class, Dimensions.class, true));
	}

	/**
	 * convert image and psf to FLoatPointers and call Cuda Richardson Lucy
	 * wrapper
	 */
	@Override
	public void computeFilter(final RandomAccessibleInterval<I> input, final RandomAccessibleInterval<K> kernel,
			RandomAccessibleInterval<O> output, long[] paddedSize) {
		
		// load native libraries, this is required before using FloatPointers
		loadNativeLibraries();

		// convert image to FloatPointer
		final FloatPointer fpInput = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(input));

		// convert PSF to FloatPointer
		final FloatPointer fpKernel = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(kernel));

		// get a FloatPointer for the output
		final FloatPointer fpOutput = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(input));

		final long startTime = System.currentTimeMillis();
		
		runNativeFilter(input, fpInput, fpKernel, fpOutput);
		
		final long endTime = System.currentTimeMillis();
		
		// copy output to array
		final float[] arrayOutput = new float[(int) (input.dimension(0) * input.dimension(1) * input.dimension(2))];
		fpOutput.get(arrayOutput);

		Pointer.free(fpInput);
		Pointer.free(fpKernel);
		Pointer.free(fpOutput);

		long[] imgSize = new long[] { input.dimension(0), input.dimension(1), input.dimension(2) };

		Img<FloatType> outPadded = ArrayImgs.floats(arrayOutput, imgSize);

		// compute unpad interval
		final long[] start = new long[outPadded.numDimensions()];
		final long[] end = new long[outPadded.numDimensions()];

		for (int d = 0; d < output.numDimensions(); d++) {
			long offset = (input.dimension(d) - output.dimension(d)) / 2;
			start[d] = offset;
			end[d] = start[d] + output.dimension(d) - 1;
		}

		// -- copy (crop) padded back to original size
		output = (RandomAccessibleInterval<O>) ops().run(Ops.Copy.RAI.class, output,
				Views.zeroMin(Views.interval(outPadded, new FinalInterval(start, end))));

	}
	
	protected abstract void loadNativeLibraries();
	
	protected abstract void runNativeFilter(Interval dimensions, FloatPointer input, FloatPointer kernel, FloatPointer output);

}