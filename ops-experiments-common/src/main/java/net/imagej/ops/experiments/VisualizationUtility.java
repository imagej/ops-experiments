
package net.imagej.ops.experiments;

import net.imagej.ImageJ;
import net.imagej.ops.Ops;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;

public class VisualizationUtility {

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static <T> Img<T> project(final ImageJ ij,
		final RandomAccessibleInterval<T> img, final int dim)
	{
		int d;
		final int[] projected_dimensions = new int[img.numDimensions() - 1];
		int i = 0;
		for (d = 0; d < img.numDimensions(); d++) {
			if (d != dim) {
				projected_dimensions[i] = (int) img.dimension(d);
				i += 1;
			}
		}

		final Img<T> proj = (Img<T>) ij.op().create().img(projected_dimensions);

		final UnaryComputerOp op = (UnaryComputerOp) ij.op().op(Ops.Stats.Max.NAME,
			img);

		final Img<T> projection = (Img<T>) ij.op().transform().project(proj, img,
			op, dim);
		return projection;
	}

}
