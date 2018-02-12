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

import org.bytedeco.javacpp.FloatPointer;
import org.scijava.Priority;
import org.scijava.plugin.Plugin;

@Plugin(type = Ops.Filter.IFFT.class, priority = Priority.LOW_PRIORITY)
public class MKLFFTWFloatRealForward2D<C extends ComplexType<C>>
		extends AbstractUnaryFunctionOp<RandomAccessibleInterval<C>, Img<ComplexFloatType>>
		implements Ops.Filter.FFT, Contingent {

	/**
	 * Compute an 2D forward FFT using FFTW
	 */
	@Override
	public Img<ComplexFloatType> calculate(final RandomAccessibleInterval<C> in) {

		try {
			MKLFFTWFloatRealForward2DWrapper.load();

			// convert to FloatPointer
			final FloatPointer p = ConvertersUtility.ii2DToFloatPointer(Views.zeroMin(in));

			// output size of FFT see
			// http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html
			final long[] fftSize = new long[] { in.dimension(0) / 2 + 1, in.dimension(1) };

			final FloatPointer pout = new FloatPointer(2 * (in.dimension(0) / 2 + 1) * in.dimension(1));
			
			MKLFFTWFloatRealForward2DWrapper.testMKLFFTW(p, pout, (int)in.dimension(0), (int)in.dimension(1));
			
			final float[] out = new float[(int) (2 * (fftSize[0]) * fftSize[1])];

			pout.get(out);

			FloatPointer.free(p);
			FloatPointer.free(pout);

			return ArrayImgs.complexFloats(out, fftSize);

		} catch (final Exception e) {
			System.out.println(e);
			return null;
		}

	}

	@Override
	public boolean conforms() {
		if (this.in().numDimensions() != 2) {
			return false;
		}

		return true;
	}

}
