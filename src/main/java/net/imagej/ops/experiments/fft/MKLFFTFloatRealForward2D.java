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

import org.bytedeco.javacpp.CLongPointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.fftw3;
import org.bytedeco.javacpp.fftw3.fftwf_plan;
import org.scijava.Priority;
import org.scijava.plugin.Plugin;

import static org.bytedeco.javacpp.mkl_rt.*;

@Plugin(type = Ops.Filter.IFFT.class, priority = Priority.LOW_PRIORITY)
public class MKLFFTFloatRealForward2D <C extends ComplexType<C>>
extends AbstractUnaryFunctionOp<RandomAccessibleInterval<C>, Img<ComplexFloatType>>
implements Ops.Filter.FFT, Contingent {
	
	/**
	 * Compute an 2D forward FFT using FFTW
	 */
	@Override
	public Img<ComplexFloatType> calculate(final RandomAccessibleInterval<C> in) {

		try {
			Loader.load(fftw3.class);
			
			DFTI_DESCRIPTOR my_desc1_handle=new DFTI_DESCRIPTOR();
			DFTI_DESCRIPTOR my_desc2_handle;
			long status;
			long[] l=new long[]{32,32};
			
			
			// convert to FloatPointer
			final FloatPointer p = ConvertersUtility.ii2DToFloatPointer(Views.zeroMin(in));

			// output size of FFT see
			// http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html
			final long[] fftSize = new long[] { in.dimension(0) / 2 + 1, in.dimension(1) };

			final FloatPointer pout = new FloatPointer(2 * (in.dimension(0) / 2 + 1) * in.dimension(1));
			
			CLongPointer size=new CLongPointer(2);
			size.position(0).put(in.dimension(0));
			size.position(1).put(in.dimension(1));
			
			// looks like the JavaCpp MKL FFT Wrapper did not work completely right, so I think I am 
			// going to have to wrap it myself
			
			/*
			// create MKL FFT plan
			status = DftiCreateDescriptor( my_desc1_handle, DFTI_SINGLE,
			          DFTI_REAL, 2, size);
			
			status =DftiSetValue( my_desc1_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
			
			
			
			// result is the complex value x[j][k], 0<=j<=31, 0<=k<=99 
			status = DftiCreateDescriptor( &my_desc2_handle, DFTI_SINGLE,
			          DFTI_REAL, 2, l);
			status = DftiCommitDescriptor( my_desc2_handle);
			status = DftiComputeForward( my_desc2_handle, y);
			status = DftiFreeDescriptor(&my_desc2_handle);
			// result is the complex value z(j,k) 0<=j<=31; 0<=k<=99
			// and is stored in CCS format
			  
			 */
			
			

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
