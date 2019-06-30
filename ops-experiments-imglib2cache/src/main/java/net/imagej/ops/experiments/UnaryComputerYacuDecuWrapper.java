
package net.imagej.ops.experiments;

import net.imagej.ops.Ops;
import net.imagej.ops.experiments.filter.deconvolve.YacuDecuRichardsonLucyOp;
import net.imagej.ops.special.computer.AbstractUnaryComputerOp;
import net.imagej.ops.special.function.BinaryFunctionOp;
import net.imagej.ops.special.function.Functions;
import net.imglib2.Cursor;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.scijava.Priority;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

/**
 * Wrap Richardson Lucy Cuda deconvolution in a UnaryComputerOp so we can run it
 * within the imglib2 cache framework.
 * 
 * @author bnorthan
 */
@Plugin(type = Ops.Deconvolve.RichardsonLucy.class, priority = Priority.LOW)
public class UnaryComputerYacuDecuWrapper extends
	AbstractUnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>>
{

	@Parameter
	UIService ui;

	@Parameter
	LogService log;

	@Parameter
	RandomAccessibleInterval<FloatType> psf;

	@Parameter
	int[] overlap;

	@Parameter
	int iterations;

	@Parameter(required = false)
	boolean nonCirculant = false;

	BinaryFunctionOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver;

	@Override
	public void initialize() {

		deconvolver = (BinaryFunctionOp) Functions.binary(ops(),
			YacuDecuRichardsonLucyOp.class, RandomAccessibleInterval.class, in(), psf,
			null, null, null, null, false, iterations, nonCirculant);

	}

	@Override
	public void compute(final RandomAccessibleInterval<FloatType> input,
		final RandomAccessibleInterval<FloatType> output)
	{

		log.info("min: " + output.min(0) + " " + output.min(1) + " " + output.min(
			2));
		log.info("max: " + output.max(0) + " " + output.max(1) + " " + output.max(
			2));

		// min and max of the cell
		final long[] min = new long[] { output.min(0), output.min(1), output.min(
			2) };
		final long[] max = new long[] { output.max(0), output.max(1), output.max(
			2) };

		// extended min and max
		final long[] mine = new long[input.numDimensions()];
		final long[] maxe = new long[input.numDimensions()];

		// overlap size
		int[] overlapmin = new int[input.numDimensions()];
		int[] overlapmax = new int[input.numDimensions()];

		// if at start or end of the image set overlap size to 0
		for (int d = 0; d < input.numDimensions(); d++) {
			overlapmin[d] = overlap[d];
			overlapmax[d] = overlap[d];

			if (min[d] == 0) {
				overlapmin[d] = 0;
			}
			if (max[d] == input.dimension(d) - 1) {
				overlapmax[d] = 0;
			}
		}

		// calculated extended min and max
		mine[0] = min[0] - overlapmin[0];
		mine[1] = min[1] - overlapmin[1];
		mine[2] = min[2];

		maxe[0] = max[0] + overlapmax[0];
		maxe[1] = max[1] + overlapmax[1];
		maxe[2] = max[2];

		RandomAccessibleInterval<FloatType> inputcopy = Views.interval(input, mine,
			maxe);

		// call deconvolver
		final RandomAccessibleInterval<FloatType> deconv = deconvolver.calculate(
			Views.zeroMin(inputcopy), psf);

		// get the valid part of the extended deconvolution
		RandomAccessibleInterval<FloatType> valid = Views.zeroMin(Views.interval(
			deconv, new FinalInterval(new long[] { overlapmin[0], overlapmin[1], 0 },
				new long[] { deconv.dimension(0) - overlapmax[0] - 1, deconv.dimension(
					1) - overlapmax[1] - 1, output.dimension(2) - 1 })));

		// copy the extended deconvolution to the original cell
		ops().copy().rai(Views.zeroMin(output), valid);
	}
}
