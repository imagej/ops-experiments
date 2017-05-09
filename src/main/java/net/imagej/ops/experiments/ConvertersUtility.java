package net.imagej.ops.experiments;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.type.numeric.ComplexType;

public class ConvertersUtility {

	/**
	 * Converts from a complex II to an interleaved complex contiguous float[]
	 * 
	 * @param ii
	 * @return contiguous float[] array containing the image data
	 */
	static public <T extends ComplexType<T>> float[] ii2DComplexToFloatArray(final IterableInterval<T> ii) {
		final Cursor<T> c = ii.cursor();

		final int xd = (int) ii.dimension(0);
		final int yd = (int) ii.dimension(1);

		final float[] data = new float[xd * yd * 2];
		final long[] pos = new long[2];

		while (c.hasNext()) {
			c.fwd();
			c.localize(pos);

			final int index = 2 * (int) (pos[0] + pos[1] * xd);

			data[index] = c.get().getRealFloat();
			data[index + 1] = c.get().getImaginaryFloat();

		}

		return data;

	}

	/**
	 * Converts from an II to a contiguous float[]
	 * 
	 * @param ii
	 * @return contiguous float[] array containing the image data
	 */
	static public <T extends ComplexType<T>> float[] ii2DToFloatArray(final IterableInterval<T> ii) {
		final Cursor<T> c = ii.cursor();

		final int xd = (int) ii.dimension(0);
		final int yd = (int) ii.dimension(1);

		final float[] data = new float[xd * yd * 2];
		final long[] pos = new long[2];

		while (c.hasNext()) {
			c.fwd();
			c.localize(pos);

			final int index = (int) (pos[0] + pos[1] * xd);

			data[index] = c.get().getRealFloat();

		}

		return data;

	}
}
