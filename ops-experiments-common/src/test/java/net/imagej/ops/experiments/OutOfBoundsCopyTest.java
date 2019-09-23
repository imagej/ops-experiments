
package net.imagej.ops.experiments;

import java.util.ArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;

import net.imagej.ImageJ;
import net.imglib2.Cursor;
import net.imglib2.Point;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.algorithm.region.hypersphere.HyperSphere;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

import org.python.modules.math;

public class OutOfBoundsCopyTest {

	static ImageJ ij = new ImageJ();

	public static void main(String[] args) throws Exception {

		System.out.println("starting test...");

		Img<FloatType> arrayIn = ArrayImgs.floats(new long[] { 512, 512, 300 });

		drawSphereInCenter(arrayIn);

		// System.out.println(MultiThreaded);

		RandomAccessibleInterval<FloatType> outOfBoundsInterval = Views.interval(
			Views.extendZero(arrayIn), Intervals.createMinMax(-10, -10, -10, 519, 519,
				309));

		// convertNaive(outOfBoundsInterval, "Out of bounds naive");
		// convertLoopBuilder(outOfBoundsInterval, "Out of bounds LoopBuilder");

		convertNaive(outOfBoundsInterval, "Out of bounds naive");
		
		for (int i = 0; i < 1; i++) {
			int numberOfThreads = (int) math.pow(2, i);
			
			ForkJoinPool executor = new ForkJoinPool(numberOfThreads);
			
			executor.submit(() -> {

				convertLoopBuilder(outOfBoundsInterval, "Out of bounds LoopBuilder " +
					numberOfThreads);
				
				// LoopBuilder.setImages(rai, temp).multiThreaded().forEachPixel(
				// (a, b) -> b.setReal(a.getRealFloat())
				// );
			}).get();
			
			convertChunked(outOfBoundsInterval, numberOfThreads,
					"Out of bounds threaded "+numberOfThreads);
	
				System.out.println();
		}
		
		

	}

	public static float[] convertNaive(RandomAccessibleInterval<FloatType> rai,
		String testName)
	{

		long start, end;

		start = System.currentTimeMillis();

		float[] floats = new float[(int) (rai.dimension(0) * rai.dimension(1) * rai
			.dimension(2))];
		
		System.out.println(rai.dimension(0)*rai.dimension(1)*rai.dimension(2));

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

	static class convertChunked implements Runnable {

		int id;
		int numThreads;
		RandomAccessibleInterval<FloatType> rai;
		float[] floats;

		public convertChunked(int numThreads, int id,
			RandomAccessibleInterval<FloatType> rai, float[] floats)
		{
			this.numThreads = numThreads;
			this.id = id;
			this.rai = rai;
			this.floats = floats;
		}

		@Override
		public void run() {
			try {
				System.out.println(" numthreads "+numThreads+" id "+id);

				Cursor<FloatType> cursor = Views.iterable(rai).cursor();

			//	int i = id;
				cursor.jumpFwd(id);

				long start, end;

				start = System.currentTimeMillis();
				
				for (int i=0;i<floats.length;i+=numThreads) {
					floats[i] = 5;//cursor.get().getRealFloat();
					i = i + numThreads;
					//cursor.jumpFwd(numThreads);
				}

			/*	while (cursor.hasNext()) {
					floats[i] = cursor.get().getRealFloat();
					i = i + numThreads;
					cursor.jumpFwd(numThreads);
				}
*/				
			//	System.out.println("i is: "+i);
				
				end = System.currentTimeMillis();

				System.out.println(  "Chumker time " + (end - start));



			}
			catch (Exception err) {
				err.printStackTrace();
			}
		}
	}

	public static float[] convertChunked(RandomAccessibleInterval<FloatType> rai,
		int numThreads, String testName)
	{

		long start, end;

		start = System.currentTimeMillis();

		float[] floats = new float[(int) (rai.dimension(0) * rai.dimension(1) * rai
			.dimension(2))];

		ArrayList<Future> tasks=new ArrayList<Future>();
		
		try {
			for (int i = 0; i < numThreads; i++) {
				tasks.add(ij.thread().run(new convertChunked(numThreads, i, rai, floats)));
			}
		}
		catch (Exception err) {
			err.printStackTrace();
		}
		
		for (final Future f:tasks) {
			try {
				f.get();
			}
			catch (InterruptedException | ExecutionException exc) {
				// TODO Auto-generated catch block
				exc.printStackTrace();
			}
		}
		
		end = System.currentTimeMillis();

		System.out.println(testName + " time " + (end - start));

		return floats;

	}

	public static float[] convertLoopBuilder(
		RandomAccessibleInterval<FloatType> rai, String testName)
	{

		long start, end;

		start = System.currentTimeMillis();

		float[] floats = new float[(int) (rai.dimension(0) * rai.dimension(1) * rai
			.dimension(2))];

		Img<FloatType> temp = ArrayImgs.floats(floats, new long[] { rai.dimension(
			0), rai.dimension(1), rai.dimension(2) });

		LoopBuilder.setImages(rai, temp).multiThreaded().flatIterationOrder(true)
			.forEachPixel((a, b) -> b.setReal(a.getRealFloat()));

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

		HyperSphere<T> hyperSphere = new HyperSphere<>(Views.extendZero(img),
			center, radius);

		for (final T value : hyperSphere) {
			value.setReal(intensity.getRealFloat());
		}

	}
}
