package net.imagej.ops.experiments.filter.deconvolve;

import java.util.Arrays;
import java.util.Map;
import java.util.Optional;

import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

public class Flowdec {

	// Default name of primary result tensor from all algorithms
	static final String DEFAULT_RESULT_TENSOR_KEY = "result";

	// Default serving signature key in meta graph signature 
	// See: https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/python/saved_model/
	// signature_constants.py
	static final String DEFAULT_SERVING_SIGNATURE_DEF_KEY = "serving_default";

	// Tag associated with "serving" graphs used in original graph export
	// See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/
	// tag_constants.py
	static final String DEFAULT_SERVING_KEY = "serve";

	public static RichardsonLucy richardsonLucy() {
		return new RichardsonLucy();
	}

	public static enum ArgType {
		INPUT, OUTPUT
	}

	public static enum Arg {

		DATA("data", ArgType.INPUT), 
		KERNEL("kernel", ArgType.INPUT), 
		NITER("niter", ArgType.INPUT), 
		PAD_MIN("pad_min", ArgType.INPUT),
		PAD_MODE("pad_mode", ArgType.INPUT),
		RESULT("result", ArgType.OUTPUT);

		public final String name;
		public final ArgType type;

		Arg(String name, ArgType type) {
			this.name = name;
			this.type = type;
		}
	}

	@SuppressWarnings("rawtypes")
	public static abstract class Algo<T extends Algo> {
	}

	public static class RichardsonLucy extends Algo<RichardsonLucy> {

		private static final String ALGOORITHM_NAME = "richardsonlucy";
		private static final String DOMAIN_TYPE = "complex";

		String getGraphName(int ndims) {
			return ALGOORITHM_NAME + "-" + DOMAIN_TYPE + "-" + ndims + "d";
		}

		public FlowdecTask.Builder task(
				Tensor<?> data, 
				Tensor<?> kernel, 
				int niter, 
				Optional<String> padMode, 
				Optional<int[]> padMin) {

			if (data.shape().length != kernel.shape().length) {
				throw new IllegalArgumentException(String.format(
						"Data and kernel must have same number of " + "dimensions (data shape = %s, kernel shape = %s)",
						Arrays.toString(data.shape()), Arrays.toString(kernel.shape())));
			}
			int ndims = data.shape().length;

			FlowdecTask.Builder builder = FlowdecTask.newBuilder(getGraphName(ndims))
					.addInput(Arg.DATA.name, data)
					.addInput(Arg.KERNEL.name, kernel)
					.addInput(Arg.NITER.name, Tensors.create(niter))
					.addOutput(Arg.RESULT.name);
			
			if (padMin.isPresent()) {
				builder.addInput(Arg.PAD_MIN.name, Tensors.create(padMin.get()));
			}
			
			if (padMode.isPresent()) {
				builder.addInput(Arg.PAD_MODE.name, Tensors.create(padMode.get()));
			}
			
			return builder;
		}

	}

	public static class TensorResult {

		private Map<String, Tensor<?>> data;

		public TensorResult(Map<String, Tensor<?>> data) {
			this.data = data;
		}

		public Tensor<?> getTensor() {
			return this.getTensor(DEFAULT_RESULT_TENSOR_KEY);
		}
		
		public Tensor<?> getTensor(String name) {
			if (this.data.isEmpty()) {
				throw new IllegalStateException("No data found in TF graph results");
			}
			if (!this.data.containsKey(name)) {
				throw new IllegalStateException("Failed to find result '" + name + "' in TF Graph results");
			}
			return this.data.get(name);
		}

		public TensorData data() {
			return this.data(DEFAULT_RESULT_TENSOR_KEY);
		}

		public TensorData data(String tensor) {
			return new TensorData(this.getTensor(tensor));
		}

	}

	public static class TensorData {
		private final Tensor<?> data;

		public TensorData(Tensor<?> data) {
			super();
			this.data = data;
		}

		protected float[] float1d() {
			Tensor<Float> res = this.data.expect(Float.class);
			if (res.shape().length != 1) {
				throw new IllegalStateException(
						"Tensor result has " + res.shape().length + " dimensions but exactly 1 was expected");
			}
			int x = (int) res.shape()[0];
			float[] arr = new float[x];
			res.copyTo(arr);
			return arr;
		}

		protected float[][] float2d() {
			Tensor<Float> res = this.data.expect(Float.class);
			if (res.shape().length != 2) {
				throw new IllegalStateException(
						"Tensor result has " + res.shape().length + " dimensions but exactly 2 were expected");
			}
			int x = (int) res.shape()[0];
			int y = (int) res.shape()[1];
			float[][] arr = new float[x][y];
			res.copyTo(arr);
			return arr;
		}

		public float[][][] float3d() {
			Tensor<Float> res = this.data.expect(Float.class);
			if (res.shape().length != 3) {
				throw new IllegalStateException(
						"Tensor result has " + res.shape().length + " dimensions but exactly 3 were expected");
			}
			int z = (int) res.shape()[0];
			int x = (int) res.shape()[1];
			int y = (int) res.shape()[2];
			float[][][] arr = new float[z][x][y];
			res.copyTo(arr);
			return arr;
		}

	}

}
