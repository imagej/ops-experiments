package net.imagej.ops.experiments.fft;

import net.imagej.ops.Contingent;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.experiments.ConvertersUtilityTest;
import net.imagej.ops.special.function.AbstractUnaryFunctionOp;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.complex.ComplexFloatType;
import net.imglib2.view.Views;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.FFT;
import org.scijava.Priority;
import org.scijava.plugin.Plugin;

@Plugin(type = Ops.Filter.IFFT.class, priority = Priority.LOW_PRIORITY)
public class ND4JFloatRealForward2D<C extends ComplexType<C>>
		extends AbstractUnaryFunctionOp<RandomAccessibleInterval<C>, Img<ComplexFloatType>>
		implements Ops.Filter.FFT, Contingent {

	/**
	 * Compute an 2D forward FFT using jtransform
	 */
	@Override
	public Img<ComplexFloatType> calculate(final RandomAccessibleInterval<C> in) {

		in.dimension(0);
		in.dimension(1);

		// get data as an INDArray
		DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
		final INDArray innd = Nd4j.zeros((int) in.dimension(0), (int) in.dimension(1));

		ConvertersUtilityTest.IIToINDArrayFloat2D(Views.iterable(in), innd);

		// perform fft
		final IComplexNDArray fftresult = FFT.fftn(innd);

		final long[] fftSize = new long[] { fftresult.rows(), fftresult.columns() };

		return ArrayImgs.complexFloats(fftresult.data().asFloat(), fftSize);
	}

	@Override
	public boolean conforms() {
		if (this.in().numDimensions() != 2) {
			return false;
		}

		return true;
	}

}
