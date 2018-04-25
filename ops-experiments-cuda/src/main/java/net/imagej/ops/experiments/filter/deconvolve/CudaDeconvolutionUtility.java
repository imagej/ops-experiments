
package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.ops.OpService;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.bytedeco.javacpp.FloatPointer;

public class CudaDeconvolutionUtility {

	static FloatPointer createNormalizationFactor(final OpService ops,
		final Interval inputDimensions, final Interval outputDimensions,
		final FloatPointer kernel, final FloatPointer X_, final FloatPointer H_)
	{
		// compute convolution interval
		final long[] start = new long[inputDimensions.numDimensions()];
		final long[] end = new long[inputDimensions.numDimensions()];

		for (int d = 0; d < outputDimensions.numDimensions(); d++) {
			final long offset = (inputDimensions.dimension(d) - outputDimensions
				.dimension(d)) / 2;
			start[d] = offset;
			end[d] = start[d] + outputDimensions.dimension(d) - 1;
		}

		final Interval convolutionInterval = new FinalInterval(start, end);

		final Img<FloatType> normal = ops.create().img(inputDimensions,
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
		YacuDecuRichardsonLucyWrapper.conv_device((int) inputDimensions.dimension(
			2), (int) inputDimensions.dimension(1), (int) inputDimensions.dimension(
				0), normalFP, kernel, normalFP, 1);

		// remove small values from the mask
		for (int i = 0; i < normal.size(); i++) {
			if (normalFP.get(i) < 0.00001) {
				normalFP.put(i, 1.f);
			}
		}

		return normalFP;
	}
}
