
package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.ops.OpService;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.filter.fftSize.NextSmoothNumber;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.Dimensions;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.loops.LoopBuilder;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.bytedeco.javacpp.FloatPointer;
import org.scijava.log.LogService;

public class CudaDeconvolutionUtility {

	/**
	 * create normalization factor described here
	 * http://bigwww.epfl.ch/deconvolution/challenge2013/index.html?p=doc_math_rl
	 * 
	 * @param ops
	 * @param log
	 * @param paddedDimensions
	 * @param outputDimensions
	 * @param kernel
	 * @return
	 */
	public static FloatPointer createNormalizationFactor(final OpService ops,
		final LogService log, final Dimensions paddedDimensions,
		final Dimensions outputDimensions, final FloatPointer kernel)
	{
		System.out.println("CreateNormalizationFactor for dimensions " +
			paddedDimensions.dimension(0) + " " + paddedDimensions.dimension(1) +
			" " + paddedDimensions.dimension(2));
		long starttime, endtime;

		// compute convolution interval
		final long[] start = new long[paddedDimensions.numDimensions()];
		final long[] end = new long[paddedDimensions.numDimensions()];

		for (int d = 0; d < outputDimensions.numDimensions(); d++) {
			final long offset = (paddedDimensions.dimension(d) - outputDimensions
				.dimension(d)) / 2;
			start[d] = offset;
			end[d] = start[d] + outputDimensions.dimension(d) - 1;
		}

		final Interval convolutionInterval = new FinalInterval(start, end);

		starttime = System.currentTimeMillis();

		final Img<FloatType> normal = ops.create().img(paddedDimensions,
			new FloatType());

		endtime = System.currentTimeMillis();
		//System.out.println("create " + (endtime - starttime));
		starttime = System.currentTimeMillis();

		final RandomAccessibleInterval<FloatType> temp = Views.interval(Views
			.zeroMin(normal), convolutionInterval);

		LoopBuilder.setImages(temp).multiThreaded().forEachPixel(a -> a.setOne());

		endtime = System.currentTimeMillis();
		//System.out.println("set ones " + (endtime - starttime));
		starttime = System.currentTimeMillis();

		final FloatPointer normalFP = ConvertersUtility.ii3DToFloatPointer(normal);
		endtime = System.currentTimeMillis();
		System.out.println("Convert " + (endtime - starttime));
		starttime = System.currentTimeMillis();

		// Call the cuda wrapper to make normal
		int error = YacuDecuRichardsonLucyWrapper.conv_device((int) paddedDimensions
			.dimension(2), (int) paddedDimensions.dimension(1), (int) paddedDimensions
				.dimension(0), normalFP, kernel, normalFP, 1);
		endtime = System.currentTimeMillis();
		System.out.println("Convolve GPU" + (endtime - starttime));
		starttime = System.currentTimeMillis();

		if (error != 0) {
			log.error("YacuDecu returned error code %d " + error);
		}

		YacuDecuRichardsonLucyWrapper.removeSmallValues(normalFP, normal.size());
		
		endtime = System.currentTimeMillis();
		System.out.println("Remove small values " + (endtime - starttime));

		return normalFP;
	}

	public static void printYacuDecuMemory(long[] imgSize, long[] psfSize) {
		AlgorithmMemory3D mem = getYacuDecuMemoryInfo(imgSize, psfSize);
		System.out.println("Extended width " + mem.extendedWidth);
		System.out.println("Extended height " + mem.extendedHeight);
		System.out.println("Extended num slices " + mem.extendedNumSlices);
		System.out.println("Image buffer size (GB) " + mem.imageBufferSize);
		System.out.println("Num buffers " + mem.numBuffers);
		System.out.println("Work size (GB) " + mem.workSpaceSize);
		System.out.println("Total memory needed (GB) " + mem.memoryNeeded);
		System.out.println();
	}

	public static AlgorithmMemory3D getYacuDecuMemoryInfo(long[] imgSize,
		long[] psfSize)
	{
		AlgorithmMemory3D mem = new AlgorithmMemory3D();

		mem.extendedWidth = NextSmoothNumber.nextSmooth((int) imgSize[0] +
			(int) psfSize[0]);
		mem.extendedHeight = NextSmoothNumber.nextSmooth((int) imgSize[1] +
			(int) psfSize[1]);
		mem.extendedNumSlices = NextSmoothNumber.nextSmooth((int) imgSize[2] +
			(int) psfSize[2]);
		mem.bytesPerPixel = 4;
		mem.imageBufferSize = (float) (mem.extendedWidth * mem.extendedHeight *
			mem.extendedNumSlices * mem.bytesPerPixel) /
			(float) AlgorithmMemory3D.KB_GB_DIVISOR;

		long workSize = YacuDecuRichardsonLucyWrapper.getWorkSize(
			(int) mem.extendedNumSlices, (int) mem.extendedHeight,
			(int) mem.extendedWidth);

		mem.workSpaceSize = (float) workSize /
			(float) AlgorithmMemory3D.KB_GB_DIVISOR;

		mem.numBuffers = 7;

		mem.memoryNeeded = mem.imageBufferSize * mem.numBuffers + mem.workSpaceSize;

		return mem;
	}

	public static long getYacuDecuMemory(long[] imgSize, long[] psfSize) {
		// compute extended size of the image based on PSF dimensions
		final long[] extendedSize = new long[imgSize.length];

		long numberVoxels = 1;

		for (int d = 0; d < imgSize.length; d++) {
			extendedSize[d] = imgSize[d] + psfSize[d];
			extendedSize[d] = (long) NextSmoothNumber.nextSmooth(
				(int) extendedSize[d]);
			numberVoxels *= extendedSize[d];
		}

		long workSize = YacuDecuRichardsonLucyWrapper.getWorkSize(
			(int) extendedSize[2], (int) extendedSize[1], (int) extendedSize[0]);

		// right now we always use floats
		int voxelSize = 4;

		return voxelSize * numberVoxels * 7 + workSize;
	}

	public static Img<FloatType> callYacuDecu(OpService ops, Img<FloatType> img,
		Img<FloatType> psf, int numIterations)
	{

		Img<FloatType> deconvolved = ops.create().img(img);

		callYacuDecu(ops, img, psf, deconvolved, numIterations);
		return deconvolved;
	}

	public static void callYacuDecu(OpService ops,
		RandomAccessibleInterval<FloatType> img,
		RandomAccessibleInterval<FloatType> psf,
		RandomAccessibleInterval<FloatType> deconvolved, int numIterations)
	{
		@SuppressWarnings("unchecked")
		final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver =
			(UnaryComputerOp) Computers.unary(ops, UnaryComputerYacuDecu.class,
				RandomAccessibleInterval.class, img, psf, numIterations);

		deconvolver.compute(img, deconvolved);

	}

}
