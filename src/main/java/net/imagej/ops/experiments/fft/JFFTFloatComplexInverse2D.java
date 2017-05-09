package net.imagej.ops.experiments.fft;

import net.imagej.ops.Contingent;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.special.function.AbstractUnaryFunctionOp;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.complex.ComplexFloatType;
import net.imglib2.view.Views;

import org.jtransforms.fft.FloatFFT_2D;
import org.scijava.Priority;
import org.scijava.plugin.Plugin;

@Plugin(type = Ops.Filter.IFFT.class, priority = Priority.LOW_PRIORITY)
public class JFFTFloatComplexInverse2D<C extends ComplexType<C>>
		extends AbstractUnaryFunctionOp<RandomAccessibleInterval<C>, Img<ComplexFloatType>>
		implements Ops.Filter.IFFT, Contingent {

	/**
	 * Compute an 2D inverse FFT using jtransform
	 */
	@Override
	public Img<ComplexFloatType> calculate(final RandomAccessibleInterval<C> in) {

		final long[] size = new long[] { in.dimension(0), in.dimension(1) };

		// TODO: the data needs to be a float array -- so we just convert it
		// eventually should check to see if we have
		// ArrayImg<FloatType,FloatArray> and
		// if so we don't need to copy

		// get data as a float array 
		final float[] data = ConvertersUtility.ii2DComplexToFloatArray(Views.zeroMin(in));

		// instantiate jtransform class and perform complex inverse fft
		final FloatFFT_2D jfft = new FloatFFT_2D(size[0], size[1]);
		jfft.complexInverse(data, true);

		return ArrayImgs.complexFloats(data, size);

	}

	@Override
	public boolean conforms() {
		if (this.in().numDimensions() != 2) {
			return false;
		}

		return true;
	}

}
