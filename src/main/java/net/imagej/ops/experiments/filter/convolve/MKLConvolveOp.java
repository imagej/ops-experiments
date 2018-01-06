package net.imagej.ops.experiments.filter.convolve;

import net.imagej.ops.Ops;
import net.imagej.ops.experiments.filter.AbstractNativeFFTFilterF;
import net.imglib2.Interval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;

import org.bytedeco.javacpp.FloatPointer;
import org.scijava.Priority;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

@Plugin(type = Ops.Filter.Convolve.class, priority = Priority.LOW_PRIORITY)
public class MKLConvolveOp<I extends RealType<I>, O extends RealType<O> & NativeType<O>, K extends RealType<K>, C extends ComplexType<C> & NativeType<C>>
		extends AbstractNativeFFTFilterF<I, O, K, C> {

	@Parameter
	boolean correlate = false;

	@Override
	protected void loadNativeLibraries() {
		MKLConvolveWrapper.load();
	}

	@Override
	protected void runNativeFilter(Interval inputDimensions, Interval outputDimensions, FloatPointer input,
			FloatPointer kernel, FloatPointer output) {

		FloatPointer X_ = null;

		FloatPointer H_ = null;

		if (inputDimensions.numDimensions() == 2) {
			final long[] fftSize = new long[] { inputDimensions.dimension(0) / 2 + 1, inputDimensions.dimension(1) };

			X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1]));

			H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1]));

		} else {
			final long[] fftSize = new long[] { inputDimensions.dimension(0) / 2 + 1, inputDimensions.dimension(1),
					inputDimensions.dimension(2) };

			X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));

			H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));
		}

		// Call the MKL wrapper
		MKLConvolveWrapper.mklConvolve(input, kernel, output, X_, H_, (int) inputDimensions.dimension(1),
				(int) inputDimensions.dimension(0), correlate);

		// Pointer.free(fpInput);
		// Pointer.free(fpKernel);
		// Pointer.free(fpOutput);

	}

}
