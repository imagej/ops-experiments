package net.imagej.ops.experiments;

import static org.bytedeco.javacpp.cuda.cudaMalloc;
import static org.bytedeco.javacpp.cuda.cudaMemcpy;
import static org.bytedeco.javacpp.cuda.cudaMemcpyHostToDevice;
import static org.bytedeco.javacpp.cuda.cudaMemcpyDeviceToHost;
import static org.bytedeco.javacpp.cuda.cudaDeviceSynchronize;

import org.bytedeco.javacpp.FloatPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import net.imagej.ops.slice.SlicesII;
import net.imglib2.Cursor;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.view.Views;

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

			VecToIIRCFloat(in.getRow(r).data().asFloat(), Views.iterable(outCursor.get()));
		}

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

		FloatPointer imgfp = new FloatPointer(ii.dimension(0) * ii.dimension(1)*ii.dimension(2));

		while (c.hasNext()) {
			c.fwd();
			c.localize(pos);

			final int index = (int) (pos[0] + pos[1] * xd+pos[2]*xd*yd);

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

		final FloatPointer imgfp=new FloatPointer(xd*yd*2);
		final long[] pos = new long[2];

		while (c.hasNext()) {
			c.fwd();
			c.localize(pos);

			final int index = 2 * (int) (pos[0] + pos[1] * xd);

			imgfp.put(index, c.get().getRealFloat());
			imgfp.put(index+1, c.get().getImaginaryFloat());


		}

		return imgfp;

	}

	/**
	 * Converts from FloatPointer to FloatPointer on Device
	 * 
	 * @param ii
	 * @return FloatPointer containing the image data
	 */
	static public <T extends ComplexType<T>> FloatPointer floatPointerHostToDevice(final FloatPointer in,
			int size) {
		FloatPointer out = new FloatPointer();

		CudaUtility.checkCudaErrors(cudaMalloc(out, size * Float.BYTES));
		CudaUtility.checkCudaErrors(cudaMemcpy(out, in, size * Float.BYTES, cudaMemcpyHostToDevice));

		return out;

	}
	
	/**
	 * Converts from FloatPointer Device to FloatPointer on Host
	 * 
	 * @param ii
	 * @return FloatPointer containing the image data
	 */
	static public <T extends ComplexType<T>> FloatPointer floatPointerDeviceToHost(final FloatPointer device,
			int size) {
		FloatPointer host = new FloatPointer(size);
        cudaDeviceSynchronize();
        cudaMemcpy(host, device, size*Float.BYTES, cudaMemcpyDeviceToHost);

		return host;

	}


	static public <T extends ComplexType<T>> FloatPointer ii2DToDeviceFloatPointer(final IterableInterval<T> ii) {
		FloatPointer hostfp = ii2DToFloatPointer(ii);
		FloatPointer devicefp = floatPointerHostToDevice(hostfp, (int) (ii.dimension(0) * ii.dimension(1)));

		FloatPointer.free(hostfp);

		return devicefp;
	}

}
