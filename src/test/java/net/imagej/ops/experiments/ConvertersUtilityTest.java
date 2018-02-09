package net.imagej.ops.experiments;

import net.imagej.ops.slice.SlicesII;
import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.view.Views;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ConvertersUtilityTest {

	/**
	 * Convert ND4J Matrix to an II
	 */
	static public <T extends ComplexType<T>> void INDArrayToIIFloat2D(final INDArray in,
			final IterableInterval<T> out) {

		final Cursor<T> cursor = out.cursor();

		// loop through ever row of the matrix
		for (int r = 0; r < in.rows(); r++) {
			final float[] data = in.data().asFloat();

			for (int c = 0; c < in.columns(); c++) {
				cursor.fwd();
				cursor.get().setReal(data[c]);
			}
		}
	}

	/**
	 * Convert II to ND4J INDArray
	 */
	static public <T extends ComplexType<T>> void IIToINDArrayFloat2D(final IterableInterval<T> in,
			final INDArray out) {

		final Cursor<T> cursor = in.cursor();

		// loop through ever row of the matrix
		for (int r = 0; r < in.dimension(0); r++) {
			final float[] data = new float[(int) in.dimension(1)];

			for (int c = 0; c < in.dimension(1); c++) {
				cursor.fwd();
				data[c] = cursor.get().getRealFloat();
			}

			Nd4j.copy(Nd4j.create(data), out.getRow(r));
		}
	}

	/**
	 * Convert a ND4J Matrix to an RAI
	 * 
	 * @param in
	 * @param out
	 */
	static public <T extends ComplexType<T>> void MatrixToRAIRCFloat(final INDArray in,
			final RandomAccessibleInterval<T> out) {

		// the slicer and cursor are used to loop through each slice of data
		final SlicesII<T> outSlicer = new SlicesII<T>(out, new int[] { 0, 1 });
		final Cursor<RandomAccessibleInterval<T>> outCursor = outSlicer.cursor();

		for (int r = 0; r < in.rows(); r++) {
			outCursor.fwd();

			ConvertersUtility.VecToIIRCFloat(in.getRow(r).data().asFloat(), Views.iterable(outCursor.get()));
		}

	}

}
