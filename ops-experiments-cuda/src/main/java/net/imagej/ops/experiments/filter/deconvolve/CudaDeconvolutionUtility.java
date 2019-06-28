
package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.ops.OpService;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.Dimensions;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.bytedeco.javacpp.FloatPointer;

public class CudaDeconvolutionUtility {

	public static FloatPointer createNormalizationFactor(final OpService ops,
		final Dimensions paddedDimensions, final Dimensions outputDimensions,
		final FloatPointer kernel, final FloatPointer X_, final FloatPointer H_)
	{
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

		final Img<FloatType> normal = ops.create().img(paddedDimensions,
			new FloatType());
		final RandomAccessibleInterval<FloatType> temp = Views.interval(Views
			.zeroMin(normal), convolutionInterval);

		for (final FloatType f : Views.iterable(temp)) {
			f.setOne();
		}
		

		// ui.show(Views.zeroMin(normal));

		final FloatPointer normalFP = ConvertersUtility.ii3DToFloatPointer(Views
			.zeroMin(normal));

		// Call the cuda wrapper to make normal
		YacuDecuRichardsonLucyWrapper.conv_device((int) paddedDimensions.dimension(
			2), (int) paddedDimensions.dimension(1), (int) paddedDimensions.dimension(
				0), normalFP, kernel, normalFP, 1);

		// remove small values from the mask
		for (int i = 0; i < normal.size(); i++) {
			if (normalFP.get(i) < 0.00001) {
				normalFP.put(i, 1.f);
			}
		}

		return normalFP;
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
			(UnaryComputerOp) Computers.unary(ops, UnaryComputerYacuDecuNC.class,
				RandomAccessibleInterval.class, img, psf, numIterations);

		deconvolver.compute(img, deconvolved);

	}
}
