package net.imagej.ops.experiments;

import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.type.numeric.ComplexType;

import org.bytedeco.javacpp.FloatPointer;

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

		final float[] data = new float[xd * yd];
		final long[] pos = new long[2];

		while (c.hasNext()) {
			c.fwd();
			c.localize(pos);

			final int index = (int) (pos[0] + pos[1] * xd);

			data[index] = c.get().getRealFloat();

		}

		return data;

	}

	/**
	 * Copy an array to an II
	 * 
	 * @param in
	 * @param out
	 */
	static public <T extends ComplexType<T>> void VecToIIRCFloat(final float[] in, final IterableInterval<T> out) {

		final Cursor<T> c = out.cursor();

		final long[] pos = new long[c.numDimensions()];

		while (c.hasNext()) {
			c.fwd();

			c.localize(pos);

			final int index = (int) (pos[0] * out.dimension(1) + pos[1]);

			c.get().setReal(in[index]);
		}
	}

	/**
	 * Converts from an II to a FloatPointer
	 * 
	 * @param ii
	 * @return FloatPointer containing the image data
	 */
	static public <T extends ComplexType<T>> FloatPointer ii2DToFloatPointer(final IterableInterval<T> ii) {
		final Cursor<T> c = ii.cursor();

		final int xd = (int) ii.dimension(0);
		final int yd = (int) ii.dimension(1);

		final long[] pos = new long[2];

		FloatPointer imgfp = new FloatPointer(ii.dimension(0) * ii.dimension(1));

		while (c.hasNext()) {
			c.fwd();
			c.localize(pos);

			final int index = (int) (pos[0] + pos[1] * xd);

			imgfp.put(index, c.get().getRealFloat());

		}

		return imgfp;

	}

	/**
	 * Converts from an II to a FloatPointer
	 * 
	 * @param ii
	 * @return FloatPointer containing the image data
	 */
	static public <T extends ComplexType<T>> FloatPointer ii3DToFloatPointer(final IterableInterval<T> ii) {
		final Cursor<T> c = ii.cursor();

		final int xd = (int) ii.dimension(0);
		final int yd = (int) ii.dimension(1);
		final int zd = (int) ii.dimension(2);

		final long[] pos = new long[3];

		FloatPointer imgfp = new FloatPointer(ii.dimension(0) * ii.dimension(1) * ii.dimension(2));

		while (c.hasNext()) {
			c.fwd();
			c.localize(pos);

			final int index = (int) (pos[0] + pos[1] * xd + pos[2] * xd * yd);

			imgfp.put(index, c.get().getRealFloat());

		}

		return imgfp;

	}

	/**
	 * Converts from a complex II to a FloatPointer
	 * 
	 * @param ii
	 * @return FloatPointer containing the image data
	 */
	static public <T extends ComplexType<T>> FloatPointer ii2DComplexToFloatPointer(final IterableInterval<T> ii) {
		final Cursor<T> c = ii.cursor();

		final int xd = (int) ii.dimension(0);
		final int yd = (int) ii.dimension(1);

		final FloatPointer imgfp = new FloatPointer(xd * yd * 2);
		final long[] pos = new long[2];

		while (c.hasNext()) {
			c.fwd();
			c.localize(pos);

			final int index = 2 * (int) (pos[0] + pos[1] * xd);

			imgfp.put(index, c.get().getRealFloat());
			imgfp.put(index + 1, c.get().getImaginaryFloat());

		}

		return imgfp;

	}

}
