package net.imagej.ops.experiments.filter.deconvolve;

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
 * Implements MKl version of Richardson Lucy deconvolution.
 * 
 * @author bnorthan
 *
 * @param <I>
 * @param <O>
 * @param <K>
 * @param <C>
 */
@Plugin(type = Ops.Deconvolve.RichardsonLucy.class, priority = Priority.LOW_PRIORITY)
public class MKLRichardsonLucyOp<I extends RealType<I>, O extends RealType<O> & NativeType<O>, K extends RealType<K>, C extends ComplexType<C> & NativeType<C>>
		extends AbstractNativeFFTFilterF<I, O, K, C> {

	@Parameter
	LogService log;

	@Parameter
	int iterations;

	@Override
	protected void loadNativeLibraries() {
		MKLRichardsonLucyWrapper.load();
	}
	
	@Override
	protected void runNativeFilter(Interval dimensions, FloatPointer input, FloatPointer kernel, FloatPointer output) {
		
		final long[] fftSize = new long[] { dimensions.dimension(0) / 2 + 1, dimensions.dimension(1), dimensions.dimension(2) };
		
		final FloatPointer X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));

		final FloatPointer H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));

		final long startTime = System.currentTimeMillis();

		// Call the MKL wrapper
		MKLRichardsonLucyWrapper.mklRichardsonLucy3D(iterations, input, kernel, output, X_, H_,
				(int) dimensions.dimension(2), (int) dimensions.dimension(1), (int) dimensions.dimension(0));
		
		//Pointer.free(fpInput);
		//Pointer.free(fpKernel);
		//Pointer.free(fpOutput);

		final long endTime = System.currentTimeMillis();

		//log.info("Total execution time (MKL) is: " + (endTime - startTime));
	}

}
