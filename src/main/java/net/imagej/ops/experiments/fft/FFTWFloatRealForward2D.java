package net.imagej.ops.experiments.fft;

import static org.bytedeco.javacpp.fftw3.FFTW_ESTIMATE;
import static org.bytedeco.javacpp.fftw3.fftwf_destroy_plan;
import static org.bytedeco.javacpp.fftw3.fftwf_execute;
import static org.bytedeco.javacpp.fftw3.fftwf_plan_dft_r2c_2d;

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
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.fftw3;
import org.bytedeco.javacpp.fftw3.fftwf_plan;
import org.scijava.Priority;
import org.scijava.plugin.Plugin;

@Plugin(type = Ops.Filter.IFFT.class, priority = Priority.LOW_PRIORITY)
public class FFTWFloatRealForward2D<C extends ComplexType<C>>
		extends AbstractUnaryFunctionOp<RandomAccessibleInterval<C>, Img<ComplexFloatType>>
		implements Ops.Filter.FFT, Contingent {

	/**
	 * Compute an 2D forward FFT using jtransform
	 */
	@Override
	public Img<ComplexFloatType> calculate(final RandomAccessibleInterval<C> in) {

		try {
			Loader.load(fftw3.class);

			// TODO: the data needs to be a float array -- so we just convert it
			// eventually should check to see if we have
			// ArrayImg<FloatType,FloatArray> and
			// if so we don't need to copy

			// get data as a float array
			final float[] data = ConvertersUtility.ii2DToFloatArray(Views.zeroMin(in));

			// convert to FloatPointer
			final FloatPointer p = new FloatPointer(in.dimension(0) * in.dimension(1));
			
			// output size of FFT see http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html
			final long[] fftSize = new long[] { in.dimension(0) / 2 + 1, in.dimension(1) };
			
			final FloatPointer pout = new FloatPointer(2 * (in.dimension(0) / 2 + 1) * in.dimension(1));

			p.put(data);

			// create FFT plan
			final fftwf_plan plan = fftwf_plan_dft_r2c_2d((int) in.dimension(0), (int) in.dimension(1), p, pout,
					(int) FFTW_ESTIMATE);

			fftwf_execute(plan);
					
			fftwf_destroy_plan(plan);

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
