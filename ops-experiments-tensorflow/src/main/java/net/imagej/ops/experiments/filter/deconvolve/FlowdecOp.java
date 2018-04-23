package net.imagej.ops.experiments.filter.deconvolve;

import java.util.Optional;

import org.scijava.Priority;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.tensorflow.Tensor;

import net.imagej.ops.OpService;
import net.imagej.ops.Ops;
import net.imagej.ops.special.function.AbstractBinaryFunctionOp;
import net.imagej.tensorflow.Tensors;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.RealType;

@SuppressWarnings("deprecation")
@Plugin(type = Ops.Deconvolve.RichardsonLucy.class, priority = Priority.LOW_PRIORITY)
public class FlowdecOp<I extends RealType<I>, O extends RealType<O>, K extends RealType<K>> 
	extends 
	AbstractBinaryFunctionOp<RandomAccessibleInterval<I>, RandomAccessibleInterval<K>, RandomAccessibleInterval<O>> {

	@Parameter
	OpService ops;

	@Parameter
	LogService log;

	@Parameter(description="Number of deconvolution iterations")
	int iterations;

	@Parameter(required = false, description = "Minimum border/padding to add around edges of "
			+ "volume to deconvolve as an integer number of pixels to add along each dimension")
	private int[] padMin = null;

	@Parameter(required = false, description="Padding 'mode' determines whether or not "
			+ "dimensions are automatically scaled out to the next highest power of 2 "
			+ "('log2') or are left as is ('none')")
	private String padMode = null;

	@Override
	public void initialize() {
		super.initialize();
	}

	@SuppressWarnings("unchecked")
	@Override
	public RandomAccessibleInterval<O> calculate(RandomAccessibleInterval<I> input,
			RandomAccessibleInterval<K> kernel) {

		/** Configure an executable deconvolution task **/
		// Task builders aren't really necessary here yet but the point of separating the
		// definition from the execution was to potentially be able to then specify
		// something like .setDevice("/gpu:0") or .setDevice("/gpu:1") as a way of 
		// dispatching multiple images to deconvolve (from different channels perhaps?)
		// to different GPUs.
		FlowdecTask.Builder task = Flowdec.richardsonLucy().task(
			Tensors.tensor(input), 
			Tensors.tensor(kernel),
			this.iterations, 
			Optional.ofNullable(this.padMode), 
			Optional.ofNullable(this.padMin));

		// Do nothing about setting devices for execution and let TensorFlow decide
		// the best execution plan
		Tensor<Float> result = (Tensor<Float>)task.build().call().getTensor();

		return (RandomAccessibleInterval<O>) Tensors.imgFloat(result);

	}

}
