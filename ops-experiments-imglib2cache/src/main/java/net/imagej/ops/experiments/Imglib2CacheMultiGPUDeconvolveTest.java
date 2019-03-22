
package net.imagej.ops.experiments;

import static net.imglib2.cache.img.DiskCachedCellImgOptions.options;

import java.io.IOException;
import java.util.ArrayList;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.FutureTask;

import net.imagej.ImageJ;
import net.imagej.ops.experiments.filter.deconvolve.YacuDecuRichardsonLucyWrapper;
import net.imagej.ops.experiments.testImages.Bars;
import net.imagej.ops.experiments.testImages.DeconvolutionTestData;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.cache.img.CellLoader;
import net.imglib2.cache.img.DiskCachedCellImg;
import net.imglib2.cache.img.DiskCachedCellImgFactory;
import net.imglib2.cache.img.DiskCachedCellImgOptions;
import net.imglib2.cache.img.DiskCachedCellImgOptions.CacheType;
import net.imglib2.cache.img.SingleCellArrayImg;
import net.imglib2.img.Img;
import net.imglib2.img.cell.Cell;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;

public class Imglib2CacheMultiGPUDeconvolveTest<T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	/**
	 * This examples demonstrates calling GPU deconvolution cell by cell on an image
	 * using DiskCachedCellFactory.
	 * 
	 * @throws InterruptedException
	 */
	public static <T extends RealType<T> & NativeType<T>> void main(final String[] args)
			throws IOException, InterruptedException {

		ij.launch(args);

		DeconvolutionTestData testData = new Bars("../images/");
		// DeconvolutionTestData testData = new CElegans();
		// DeconvolutionTestData testData = new HalfBead();

		testData.LoadImages(ij);
		RandomAccessibleInterval<FloatType> imgF = testData.getImg();
		RandomAccessibleInterval<FloatType> psfF = testData.getPSF();

		ImageJFunctions.show(imgF);
		ImageJFunctions.show(psfF);

		final int iterations = 100;
		final int cellBorderXY = (int) psfF.dimension(0);
		final int cellBorderZ = 0;

		final int[] cellDimensions = new int[] { (int) Math.ceil(imgF.dimension(0) / 2),
				(int) Math.ceil(imgF.dimension(1) / 2), (int) imgF.dimension(2) };

		@SuppressWarnings("unchecked")
		final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver = (UnaryComputerOp) Computers
				.unary(ij.op(), UnaryComputerYacuDecuWrapper.class, RandomAccessibleInterval.class, imgF, psfF,
						new long[] { cellBorderXY, cellBorderXY, cellBorderZ }, iterations);

		final DiskCachedCellImgOptions writeOnlyDirtyOptions = options().cellDimensions(cellDimensions)
				.cacheType(CacheType.BOUNDED).maxCacheSize(100);
		final DiskCachedCellImgFactory<FloatType> factory = new DiskCachedCellImgFactory<>(writeOnlyDirtyOptions);

		int numGPUs = 4;

		// create a queue to hold the GPU jobs, max size of queue is the number of GPUS
		BlockingQueue<FutureTask> queue = new ArrayBlockingQueue<FutureTask>(numGPUs);

		// create a list of GPU monitors.  This class will loop and wait for GPU jobs
		ArrayList<GPUQueueMonitor> monitors = new ArrayList<GPUQueueMonitor>();

		// for each GPU create and add a GPU monitor
		for (int i = 0; i < numGPUs; i++) {
			GPUQueueMonitor looper = new GPUQueueMonitor(queue, i + 1);
			monitors.add(looper);
			looper.start();
		}

		// create a loader, the loader doesn't do the work, but defines a lambda
		// and puts the lambda in the queue
		CellLoader loader = (SingleCellArrayImg img) -> {
			FutureTask future = new FutureTask(() -> {
				deconvolver.compute(imgF, img);
			}, "finished");
			queue.put(future); // blocks until space is free in queue
			future.get(); // blocks until GPU thread has processed the job
		};

		// create a new image using the disk cache factory. Pass it the loader defined
		// above
		DiskCachedCellImg<FloatType, RandomAccessibleInterval<FloatType>> out = (DiskCachedCellImg) factory.create(
				new long[] { imgF.dimension(0), imgF.dimension(1), imgF.dimension(2) }, new FloatType(), loader,
				options().initializeCellsAsDirty(true));

		long start = System.currentTimeMillis();
		// out.getCells().forEach(Cell::getData);

		// trigger
		//out.getCells().forEach(a -> new Thread(new Runnable() {
//
//			@Override
//			public void run() {
//				a.getData();
//			}
//		}).start());

		Thread t1 = trigger(out, cellDimensions, 0, 0);
		Thread t2 = trigger(out, cellDimensions, 1, 0);
		Thread t3 = trigger(out, cellDimensions, 0, 1);
		Thread t4 = trigger(out, cellDimensions, 1, 1);

		t1.join();
		t2.join();
		t3.join();
		t4.join();

		long finish = System.currentTimeMillis();
		System.out.println("processing time: " + (finish - start));

		// now show the image this should trigger the loader for each cell
		ij.ui().show(out);

	}

	static <T extends RealType<T>> Thread trigger(RandomAccessibleInterval<T> out, int[] cellDimensions, int xcell,
			int ycell) {
		Thread thread = new Thread(new Runnable() {
			@Override
			public void run() {
				RandomAccess<T> ra = out.randomAccess();
				ra.setPosition(new int[] { xcell * cellDimensions[0], ycell * cellDimensions[1], 0 });
				ra.get().getRealDouble();
			}
		});

		thread.start();
		return thread;
	}

}
