package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.ops.OpService;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.experiments.filter.convolve.MKLConvolve3DWrapper;
import net.imglib2.FinalInterval;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.bytedeco.javacpp.FloatPointer;

public class NativeDeconvolutionUtility {
	static FloatPointer createNormalizationFactor(OpService ops, Interval inputDimensions, Interval outputDimensions,
			FloatPointer kernel, FloatPointer X_, FloatPointer H_) {
		// compute convolution interval
		final long[] start = new long[inputDimensions.numDimensions()];
		final long[] end = new long[inputDimensions.numDimensions()];

		for (int d = 0; d < outputDimensions.numDimensions(); d++) {
			long offset = (inputDimensions.dimension(d) - outputDimensions.dimension(d)) / 2;
			start[d] = offset;
			end[d] = start[d] + outputDimensions.dimension(d) - 1;
		}

		Interval convolutionInterval = new FinalInterval(start, end);

		Img<FloatType> mask = (Img<FloatType>) ops.create().img(inputDimensions, new FloatType());
		RandomAccessibleInterval<FloatType> temp = Views.interval(Views.zeroMin(mask), convolutionInterval);

		for (FloatType f : Views.iterable(temp)) {
			f.setOne();
		}

		// ui.show(Views.zeroMin(mask));

		final FloatPointer mask_ = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(mask));

		// Call the MKL wrapper to make normal
		MKLConvolve3DWrapper.mklConvolve3D(mask_, kernel, mask_, X_, H_, (int) inputDimensions.dimension(2),
				(int) inputDimensions.dimension(1), (int) inputDimensions.dimension(0), true);
		
		return mask_;
	}
}
