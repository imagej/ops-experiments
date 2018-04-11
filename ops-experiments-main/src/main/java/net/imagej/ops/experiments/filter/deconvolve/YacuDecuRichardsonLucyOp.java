package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.ops.OpService;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.filter.AbstractNativeFFTFilterF;
import net.imglib2.Interval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;

import org.bytedeco.javacpp.FloatPointer;
import org.scijava.Priority;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

/**
 * Implements Cuda version of Richardson Lucy deconvolution.
 * 
 * @author bnorthan
 *
 * @param <I>
 * @param <O>
 * @param <K>
 * @param <C>
 */
@Plugin(type = Ops.Deconvolve.RichardsonLucy.class, priority = Priority.LOW_PRIORITY)
public class YacuDecuRichardsonLucyOp<I extends RealType<I>, O extends RealType<O> & NativeType<O>, K extends RealType<K>, C extends ComplexType<C> & NativeType<C>>
		extends AbstractNativeFFTFilterF<I, O, K, C> {

	@Parameter
	OpService ops;
	
	@Parameter
	LogService log;

	@Parameter
	int iterations;
	
	@Parameter(required = false)
	boolean nonCirculant = false;

	@Override
	protected void loadNativeLibraries() {
		YacuDecuRichardsonLucyWrapper.load();

	}

	@Override
	protected void runNativeFilter(Interval inputDimensions, Interval outputDimensions, FloatPointer input, FloatPointer kernel, FloatPointer output) {
		
		final long[] fftSize = new long[] { inputDimensions.dimension(0) / 2 + 1, inputDimensions.dimension(1),
				inputDimensions.dimension(2) };
		
		final FloatPointer X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));

		final FloatPointer H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));

		final FloatPointer mask_;

		int arraySize = (int) (inputDimensions.dimension(0) * inputDimensions.dimension(1)
				* inputDimensions.dimension(2));

		// create the normalization factor needed for non-circulant mode
		if (nonCirculant == true) {

			mask_=NativeDeconvolutionUtility.createNormalizationFactor(ops, inputDimensions, outputDimensions,
						kernel,  X_,H_);
		}
		else {
			mask_=null;
		}
		
		final long startTime = System.currentTimeMillis();

		// Call the Cuda wrapper
		YacuDecuRichardsonLucyWrapper.deconv_device(iterations, (int) inputDimensions.dimension(2),
				(int) inputDimensions.dimension(1), (int) inputDimensions.dimension(0), input, kernel, output, null);

		final long endTime = System.currentTimeMillis();

		log.info("Total execution time (Cuda) is: " + (endTime - startTime));

	}

}
