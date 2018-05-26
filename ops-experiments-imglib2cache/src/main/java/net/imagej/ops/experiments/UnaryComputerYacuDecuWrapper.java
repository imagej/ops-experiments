
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
	int[] border;

	@Parameter
	int iterations;
	
	@Parameter(required=false)
	boolean nonCirculant=false;

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

		// border size
		int[] bordermin = new int[input.numDimensions()];
		int[] bordermax = new int[input.numDimensions()];

		// if at start or end of the image set border size to 0
		for (int d = 0; d < input.numDimensions(); d++) {
			bordermin[d] = border[d];
			bordermax[d] = border[d];

			if (min[d] == 0) {
				bordermin[d] = 0;
			}
			if (max[d] == input.dimension(d) - 1) {
				bordermax[d] = 0;
			}
		}

		// calculated extended min and max
		mine[0] = min[0] - bordermin[0];
		mine[1] = min[1] - bordermin[1];
		mine[2] = min[2];

		maxe[0] = max[0] + bordermax[0];
		maxe[1] = max[1] + bordermax[1];
		maxe[2] = max[2];

		 RandomAccessibleInterval<FloatType> inputcopy = ops().copy().rai(Views
		 .interval(input, mine, maxe));

		// call deconvolver
		final RandomAccessibleInterval<FloatType> deconv = deconvolver.calculate(
			Views.zeroMin(inputcopy), psf);

		// copy the extended deconvolution to the original cell
		Cursor<FloatType> c1 = Views.iterable(Views.zeroMin(output)).cursor();

		RandomAccessibleInterval<FloatType> r = Views.zeroMin(Views.interval(deconv,
			new FinalInterval(new long[] { bordermin[0], bordermin[1], 0 },
				new long[] { deconv.dimension(0) - bordermax[0] - 1, deconv.dimension(
					1) - bordermax[1] - 1, output.dimension(2) - 1 })));

		RandomAccess<FloatType> ra = r.randomAccess();

		c1.fwd();

		while (c1.hasNext()) {

			ra.setPosition(c1);

			// set the value of this pixel of the output image, every Type supports
			// T.set( T type )
			c1.get().set(ra.get());
			c1.fwd();

		}

		// ui.show(Views.zeroMin(output));

		/*
		ops().copy().rai(Views.zeroMin(output), Views.interval(deconv,
			new FinalInterval(new long[] { 0, 0, 0 }, new long[] { output.dimension(
				0) - 1, output.dimension(2) - 1, output.dimension(2) - 1 })));*/

	}
}
