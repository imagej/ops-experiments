
package net.imagej.ops.experiments;

import net.imglib2.Cursor;
import net.imglib2.Dimensions;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.bytedeco.javacpp.FloatPointer;

public class ConvertersUtility {

	/**
	 * Converts from a complex II to an interleaved complex contiguous float[]
	 * 
	 * @param ii
	 * @return contiguous float[] array containing the image data
	 */
	static public <T extends ComplexType<T>> float[] ii2DComplexToFloatArray(
		final IterableInterval<T> ii)
	{
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
	static public <T extends ComplexType<T>> float[] ii2DToFloatArray(
		final IterableInterval<T> ii)
	{
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
	 * Converts from an II to a contiguous float[]
	 * 
	 * @param ii
	 * @return contiguous float[] array containing the image data
	 */
	static public <T extends ComplexType<T>> float[] ii3DToFloatArray(
		final IterableInterval<T> ii)
	{
		final Cursor<T> c = ii.cursor();

		final int xd = (int) ii.dimension(0);
		final int yd = (int) ii.dimension(1);
		final int zd = (int) ii.dimension(2);

		final float[] data = new float[xd * yd * zd];
		final long[] pos = new long[3];

		while (c.hasNext()) {
			c.fwd();
			c.localize(pos);

			final int index = (int) (pos[0] + pos[1] * xd + pos[2] * xd * yd);

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
	static public <T extends ComplexType<T>> void VecToIIRCFloat(final float[] in,
		final IterableInterval<T> out)
	{

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
	static public <T extends ComplexType<T>> FloatPointer ii2DToFloatPointer(
		final IterableInterval<T> ii)
	{
		final Cursor<T> c = ii.cursor();

		final int xd = (int) ii.dimension(0);

		final long[] pos = new long[2];

		final FloatPointer imgfp = new FloatPointer(ii.dimension(0) * ii.dimension(
			1));

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
	static public <T extends ComplexType<T>> FloatPointer ii3DToFloatPointer_(
		final IterableInterval<T> ii)
	{
		final Cursor<T> c = ii.cursor();

		final int xd = (int) ii.dimension(0);
		final int yd = (int) ii.dimension(1);

		final long[] pos = new long[3];

		final FloatPointer imgfp = new FloatPointer(ii.dimension(0) * ii.dimension(
			1) * ii.dimension(2));

		while (c.hasNext()) {
			c.fwd();
			c.localize(pos);

			final int index = (int) (pos[0] + pos[1] * xd + pos[2] * xd * yd);

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
	static public <T extends ComplexType<T>> FloatPointer ii3DToFloatPointer__(
		final IterableInterval<T> ii)
	{
		final Cursor<T> c = ii.cursor();

		float[] temp = new float[(int) (ii.dimension(0) * ii.dimension(1) * ii
			.dimension(2))];

		int index = 0;

		while (c.hasNext()) {
			c.fwd();
			temp[index++] = c.get().getRealFloat();

		}

		FloatPointer imgfp = new FloatPointer(temp);

		return imgfp;

	}

	/**
	 * Converts from an II to a FloatPointer
	 * 
	 * @param ii
	 * @return FloatPointer containing the image data
	 */
	static public <T extends ComplexType<T>> FloatPointer ii3DToFloatPointer(
		final RandomAccessibleInterval<T> rai)
	{

		final IterableInterval<T> ii = Views.iterable(rai);

		final Cursor<T> c = ii.cursor();

		long totalSize = ii.dimension(0) * ii.dimension(1) * ii.dimension(2);

		int chunkSize = (int) Math.min(Integer.MAX_VALUE - 128, totalSize);

		float[] temp = new float[chunkSize];

		FloatPointer imgfp = new FloatPointer(totalSize);

		int numElementsInChunk = 0;
		int chunkNum = 0;

		while (c.hasNext()) {

			c.fwd();
			temp[numElementsInChunk++] = c.get().getRealFloat();

			// if the chunk is full put it in the FloatPointer
			if (numElementsInChunk == chunkSize) {
				imgfp.put(temp, chunkNum * chunkSize, numElementsInChunk);
				numElementsInChunk = 0;
				chunkNum++;
			}
		}

		// if there is a partial chunk left put it in the FloatPointer
		if (numElementsInChunk > 0) {
			imgfp.put(temp, chunkNum * chunkSize, numElementsInChunk);
		}

		return imgfp;

	}

	/**
	 * Converts from an II to a FloatPointer
	 * 
	 * @param ii
	 * @return FloatPointer containing the image data
	 */
	static public <T extends ComplexType<T>> FloatPointer ii3DToFloatPointerArray(
		final RandomAccessibleInterval<T> rai)
	{

		float[] temp = new float[(int) (rai.dimension(0) * rai.dimension(1) * rai
			.dimension(2))];

		Img<FloatType> imgTemp = ArrayImgs.floats(temp, new long[] { rai.dimension(
			0), rai.dimension(1), rai.dimension(2) });

		LoopBuilder.setImages(rai, imgTemp).multiThreaded().forEachPixel((a, b) -> b
			.set(a.getRealFloat()));

		FloatPointer imgfp = new FloatPointer(temp);

		return imgfp;

	}

	/**
	 * Converts from a complex II to a FloatPointer
	 * 
	 * @param ii
	 * @return FloatPointer containing the image data
	 */
	static public <T extends ComplexType<T>> FloatPointer
		ii2DComplexToFloatPointer(final IterableInterval<T> ii)
	{
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

	/**
	 * Converts from a FloatPointer to an Img
	 */
	static public Img<FloatType> floatPointerToImg3D(final FloatPointer fp,
		final Dimensions d)
	{

		final float[] temp = new float[(int) (d.dimension(0) * d.dimension(1) * d
			.dimension(2))];
		fp.get(temp);

		return ArrayImgs.floats(temp, new long[] { d.dimension(0), d.dimension(1), d
			.dimension(2) });
	}

	/**
	 * Compute mean of a FloatPointer
	 */
	static public float mean(final FloatPointer fp, final int length) {
		float mean = 0;
		for (int i = 0; i < length; i++) {
			mean += fp.get(i);
		}
		return mean;
	}

}
