
package net.imagej.ops.experiments;

import net.imagej.ImageJ;
import net.imglib2.Cursor;
import net.imglib2.FinalDimensions;
import net.imglib2.Point;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.region.hypersphere.HyperSphere;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.planar.PlanarImgs;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

public class ConvertersTest {

	static ImageJ ij = new ImageJ();

	public static void main(String[] args) throws Exception {

		System.out.println("starting test...");

		Img<FloatType> arrayTest = ArrayImgs.floats(new long[] { 512, 512, 300 });
		Img<FloatType> planarTest = PlanarImgs.floats(new long[] { 512, 512, 300 });
		Img<FloatType> opsCreateTest = ij.op().create().img(new FinalDimensions(
			new long[] { 512, 512, 300 }), new FloatType());
		RandomAccessibleInterval<FloatType> inBoundsInterval = Views.interval(
			arrayTest, Intervals.createMinMax(10, 10, 10, 500, 500, 290));
		RandomAccessibleInterval<FloatType> outOfBoundsInterval = Views.interval(
			Views.extendZero(arrayTest), Intervals.createMinMax(-10, -10, -10, 490,
				490, 290));
		
		drawSphereInCenter(arrayTest);
		
		ij.ui().show(arrayTest);

		float[] arrayFloats = convert(arrayTest, "array ");
		float[] planarFloats = convert(planarTest, "planar ");
		float[] opsCreateFloats = convert(opsCreateTest, "ops ");
		float[] inBoundsFloats = convert(inBoundsInterval, "in bounds rai ");
		float[] outOfBoundsFloats = convert(outOfBoundsInterval,
			"out of bounds rai ");

		Img<FloatType> test = ArrayImgs.floats(outOfBoundsFloats, new long[] {
			outOfBoundsInterval.dimension(0), outOfBoundsInterval.dimension(1),
			outOfBoundsInterval.dimension(2) });
		
		ij.ui().show(test);
	}

	public static float[] convert(RandomAccessibleInterval<FloatType> rai,
		String testName)
	{

		long start, end;

		start = System.currentTimeMillis();

		float[] floats = new float[(int) (rai.dimension(0) * rai.dimension(1) * rai
			.dimension(2))];

		Cursor<FloatType> cursor = Views.iterable(rai).cursor();

		cursor.fwd();

		int i = 0;

		while (cursor.hasNext()) {
			floats[i++] = cursor.get().getRealFloat();
			cursor.fwd();
		}

		end = System.currentTimeMillis();

		System.out.println(testName + " time " + (end - start));

		return floats;

	}

	public static <T extends RealType<T>> void drawSphereInCenter(Img<T> img) {
		final Point center = new Point(img.numDimensions());

		for (int d = 0; d < img.numDimensions(); d++)
			center.setPosition(img.dimension(d) / 2, d);

		long radius = Math.min(img.dimension(0), Math.min(img.dimension(1), img
			.dimension(2)));

		T intensity = img.firstElement().copy();
		intensity.setReal(255.);;

		HyperSphere<T> hyperSphere = new HyperSphere<>(Views.extendZero(img), center, radius);

		for (final T value : hyperSphere) {
			value.setReal(intensity.getRealFloat());
		}

	}
}
