package net.imagej.ops.experiments.fft;

import static org.bytedeco.javacpp.fftw3.FFTW_ESTIMATE;
import static org.bytedeco.javacpp.fftw3.fftwf_destroy_plan;
import static org.bytedeco.javacpp.fftw3.fftwf_execute;
import static org.bytedeco.javacpp.fftw3.fftwf_plan_dft_c2r_2d;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.fftw3;
import org.bytedeco.javacpp.fftw3.fftwf_plan;
import org.scijava.Priority;
import org.scijava.plugin.Plugin;

import net.imagej.ops.Contingent;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.special.function.AbstractUnaryFunctionOp;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

@Plugin(type = Ops.Filter.IFFT.class, priority = Priority.LOW_PRIORITY)
public class FFTWFloatRealInverse2D<C extends ComplexType<C>> extends
		AbstractUnaryFunctionOp<RandomAccessibleInterval<C>, Img<FloatType>>implements Ops.Filter.FFT, Contingent {

	/**
	 * Compute an 2D forward FFT using FFTW
	 */
	@Override
	public Img<FloatType> calculate(final RandomAccessibleInterval<C> in) {

		try {
			Loader.load(fftw3.class);

			// get data as a float array
			final float[] data = ConvertersUtility.ii2DComplexToFloatArray(Views.zeroMin(in));

			// convert complex RAI to FloatPointer
			final FloatPointer p = new FloatPointer(in.dimension(0) * in.dimension(1) * 2);

			p.put(data);

			// size of real signal
			final int[] realSize = new int[] { ((int) in.dimension(0) - 1) * 2, (int) in.dimension(1) };

			final FloatPointer pout = new FloatPointer(realSize[0] * realSize[1]);

			// create FFT plan
			final fftwf_plan plan = fftwf_plan_dft_c2r_2d(realSize[0], realSize[1], p, pout, (int) FFTW_ESTIMATE);

			fftwf_execute(plan);

			fftwf_destroy_plan(plan);

			final float[] out = new float[(int) (realSize[0] * realSize[1])];

			pout.get(out);

			FloatPointer.free(p);
			FloatPointer.free(pout);

			return ArrayImgs.floats(out, new long[] { realSize[0], realSize[1] });

		} catch (final Exception e) {
			System.out.println(e);
			return null;
		}

	}

	@Override
	public boolean conforms() {

		if (this.in() == null) {
			return true;
		}

		if (this.in().numDimensions() != 2) {
			return false;
		}

		return true;
	}

}
