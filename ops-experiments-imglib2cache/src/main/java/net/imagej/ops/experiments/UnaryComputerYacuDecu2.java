
package net.imagej.ops.experiments;

import net.imagej.ops.OpService;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.filter.deconvolve.YacuDecuRichardsonLucyOp;
import net.imagej.ops.experiments.filter.deconvolve.YacuDecuRichardsonLucyWrapper;
import net.imagej.ops.filter.fftSize.DefaultComputeFFTSize;
import net.imagej.ops.special.computer.AbstractUnaryComputerOp;
import net.imagej.ops.special.function.BinaryFunctionOp;
import net.imagej.ops.special.function.Functions;
import net.imglib2.Cursor;
import net.imglib2.FinalDimensions;
import net.imglib2.FinalInterval;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.bytedeco.javacpp.FloatPointer;
import org.scijava.Priority;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

/**
 * Calls YacuDecu (GPU) version of Richardson Lucy
 * 
 * @author bnorthan
 */
@Plugin(type = Ops.Deconvolve.RichardsonLucy.class, priority = Priority.LOW)
public class UnaryComputerYacuDecu2 extends
	AbstractUnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>
{
	
	@Parameter
	OpService ops;

	@Parameter
	UIService ui;

	@Parameter
	LogService log;

	@Parameter
	RandomAccessibleInterval<FloatType> psf;

	@Parameter
	int iterations;
	
	@Override
	public void compute(final RandomAccessibleInterval<FloatType> input,
		final RandomAccessibleInterval<FloatType> output)
	{

		log.info("min: " + output.min(0) + " " + output.min(1) + " " + output.min(
			2));
		log.info("max: " + output.max(0) + " " + output.max(1) + " " + output.max(
			2));

		// min of the cell we are generating data for
		final long[] min = new long[] { output.min(0), output.min(1), output.min(
			2) };
		
		// compute extended size of the image based on PSF dimension
		final long[] extendedSize=new long[output.numDimensions()];
		
		for (int d=0;d<output.numDimensions();d++) {
			extendedSize[d]=output.dimension(d)+psf.dimension(d);
		}
		
		// compute fast FFT extended size
		long[][] fastExtendedSize=(long[][])ops.run(DefaultComputeFFTSize.class, new FinalDimensions(extendedSize),false);
		FinalDimensions fastExtendedDimensions=new FinalDimensions(fastExtendedSize[0]);
		
		// compute extended min and max to obtain a region with size fastExtendedSize
		final long[] minExtendedInput = new long[input.numDimensions()];
		final long[] maxExtendedInput = new long[input.numDimensions()];
		
		for (int d = 0; d < psf.numDimensions(); d++) {
			minExtendedInput[d] = min[d] - (fastExtendedDimensions.dimension(d)-output.dimension(d))/2;
			maxExtendedInput[d] = minExtendedInput[d] + fastExtendedDimensions.dimension(d)-1;
		}

		FinalInterval inputInterval = new FinalInterval(minExtendedInput,
			maxExtendedInput);
		
		IterableInterval<FloatType> inputIterable = Views.zeroMin(Views.interval(
			Views.extendMirrorSingle(input), inputInterval));

		RandomAccessibleInterval<FloatType> paddedPSF=Views.zeroMin(ops.filter().padShiftFFTKernel(psf, fastExtendedDimensions));
		
		//ui.show("intputIterable",inputIterable);
		//ui.show("paddedPSF",paddedPSF);

		// copy to GPU memory
		YacuDecuRichardsonLucyWrapper.load();

		FloatPointer fpInput = null;
		FloatPointer fpPSF = null;
		FloatPointer fpOutput = null;

		// convert image to FloatPointer
		fpInput = ConvertersUtility.ii3DToFloatPointer(inputIterable);

		// convert PSF to FloatPointer
		fpPSF = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(paddedPSF));

		// convert image to FloatPointer to use as output buffer
		// (this means first guess of object is the original image)
		fpOutput = ConvertersUtility.ii3DToFloatPointer(inputIterable);

		final long startTime = System.currentTimeMillis();

		// Call the Cuda wrapper
		YacuDecuRichardsonLucyWrapper.deconv_device(iterations, (int) inputIterable
			.dimension(2), (int) inputIterable.dimension(1), (int) inputIterable
				.dimension(0), fpInput, fpPSF, fpOutput, null);

		final long endTime = System.currentTimeMillis();

		int bufferSize=(int) (inputIterable.dimension(0) *
				inputIterable.dimension(1) * inputIterable.dimension(2));
		
		// copy output to array
		final float[] arrayOutput = new float[bufferSize];
		fpOutput.get(arrayOutput);
		
		final Img<FloatType> deconv = ArrayImgs.floats(arrayOutput, new long[] { inputIterable.dimension(0), inputIterable.dimension(1), inputIterable.dimension(2) });

		//ui.show("deconv",deconv);
		
		// copy the extended deconvolution to the original cell
		Cursor<FloatType> c1 = Views.iterable(Views.zeroMin(output)).cursor();

		RandomAccessibleInterval<FloatType> r = Views.zeroMin(Views.interval(deconv,
			new FinalInterval(new long[] { min[0]-minExtendedInput[0], min[0]-minExtendedInput[0], min[0]-minExtendedInput[0] },
				new long[] { output.dimension(0) - 1, output.dimension(
					1)- 1, output.dimension(2)-1 })));

		RandomAccess<FloatType> ra = r.randomAccess();
		
		c1.fwd();

		while (c1.hasNext()) {

			ra.setPosition(c1);

			// set the value of this pixel of the output image, every Type supports
			// T.set( T type )
			c1.get().set(ra.get());
			c1.fwd();

		}

		/*
		ops().copy().rai(Views.zeroMin(output), Views.interval(deconv,
			new FinalInterval(new long[] { 0, 0, 0 }, new long[] { output.dimension(
				0) - 1, output.dimension(2) - 1, output.dimension(2) - 1 })));*/

	}
}
