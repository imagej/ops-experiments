package net.imagej.ops.experiments.filter.convolve;

import net.imagej.ops.Ops;
import net.imagej.ops.experiments.filter.AbstractNativeFFTFilterF;
import net.imglib2.Interval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;

import org.bytedeco.javacpp.FloatPointer;
import org.scijava.Priority;
import org.scijava.plugin.Plugin;

@Plugin(type = Ops.Filter.Convolve.class, priority = Priority.LOW_PRIORITY)
public class MKLConvolveOp<I extends RealType<I>, O extends RealType<O> & NativeType<O>, K extends RealType<K>, C extends ComplexType<C> & NativeType<C>>
		extends AbstractNativeFFTFilterF<I, O, K, C> {

	@Override
	protected void loadNativeLibraries() {
		MKLConvolveWrapper.load();
	}

	@Override
	protected void runNativeFilter(Interval dimensions, FloatPointer input, FloatPointer kernel, FloatPointer output) {

		FloatPointer X_ = null;

		FloatPointer H_ = null;

		if (dimensions.numDimensions() == 2) {
			final long[] fftSize = new long[] { dimensions.dimension(0) / 2 + 1, dimensions.dimension(1) };

			X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1]));

			H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1]));

		} else {
			final long[] fftSize = new long[] { dimensions.dimension(0) / 2 + 1, dimensions.dimension(1),
					dimensions.dimension(2) };

			X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));

			H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));
		}

		// Call the MKL wrapper
		MKLConvolveWrapper.mklConvolve(input, kernel, output, X_, H_, (int) dimensions.dimension(1),
				(int) dimensions.dimension(0));

		// Pointer.free(fpInput);
		// Pointer.free(fpKernel);
		// Pointer.free(fpOutput);

	}

}
