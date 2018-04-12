package net.imagej.ops.experiments.filter.deconvolve;

import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.Callable;

import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;

import net.imagej.ops.experiments.filter.deconvolve.Flowdec.TensorResult;

public class FlowdecTask implements Callable<TensorResult>, Runnable {

	private Builder builder;
	private TensorResult result;
	
	FlowdecTask(Builder builder){
		this.builder = builder;
	}
	
	public static class Builder {
		String graphName;
		Map<String, Tensor<?>> inputs;
		List<String> outputs;
		Optional<String> device;
		Optional<ConfigProto> sessionConfig;
		FlowdecGraph graph;
		
		public Builder(String graphName) {
			this.inputs = new LinkedHashMap<>();
			this.outputs = new LinkedList<>();
			this.device = Optional.empty();
			this.sessionConfig = Optional.empty();
			this.graphName = graphName;
		}
		
		Builder addInput(String name, Tensor<?> value) {
			this.inputs.put(name, value);
			return this;
		}
		
		Builder addOutput(String name) {
			this.outputs.add(name);
			return this;
		}
		
		public Builder setDevice(String device) {
			this.device = Optional.ofNullable(device);
			return this;
		}
		
		public Builder setSessionConfig(ConfigProto config) {
			this.sessionConfig = Optional.ofNullable(config);
			return this;
		}
		
		public FlowdecTask build() {
			if (this.inputs.isEmpty()) {
				throw new IllegalStateException("At least one input should be specified for task");
			}
			if (this.outputs.isEmpty()) {
				throw new IllegalStateException("At least one output should be specified for task");
			}
			this.graph = FlowdecGraph.load(this.graphName);
			return new FlowdecTask(this);
		}
		
	}
	
	public static Builder newBuilder(String graphName) {
		return new Builder(graphName);
	}
	
	@Override
	public synchronized void run() {

		if (this.result != null) {
			return;
		}
		
		FlowdecGraph g = this.builder.graph;
		if (this.builder.device.isPresent()) {
			g = g.setDevice(this.builder.device.get());	
		}
		
		MetaGraphDef mg = g.getMetaGraph();
		SignatureDef sd = mg.getSignatureDefMap().get(Flowdec.DEFAULT_SERVING_SIGNATURE_DEF_KEY);
		
		Optional<ConfigProto> conf = this.builder.sessionConfig;
		try (Session sess = conf.isPresent() ? 
				new Session(g.getComputationGraph(), conf.get().toByteArray()) : 
					new Session(g.getComputationGraph())){
			
			Runner runner = sess.runner();
			
			Set<String> inputNames = sd.getInputsMap().keySet();
			Set<String> outputNames = sd.getOutputsMap().keySet();
			for (String name : this.builder.inputs.keySet()) {
				// Get name of tensor corresponding to input name; this is typically
				// similar to the name of the input itself but does need to be (eg
				// the input name may be "data" while the tensor placeholder is "data:0")
				if (!inputNames.contains(name)) {
					throw new IllegalArgumentException("Graph input '" + 
							name + "' not valid (valid inputs = '" + inputNames + ")");
				}
				String inputTensorName = sd.getInputsMap().get(name).getName();
				runner.feed(inputTensorName, this.builder.inputs.get(name));
			}
			for (String name : this.builder.outputs) {
				if (!outputNames.contains(name)) {
					throw new IllegalArgumentException("Graph output '" + 
							name + "' not valid (valid outputs = '" + outputNames + ")");
				}
				String outputTensorName = sd.getOutputsMap().get(name).getName();
				runner.fetch(outputTensorName);
			}

			Map<String, Tensor<?>> map = new LinkedHashMap<>();
			List<Tensor<?>> tensors = runner.run();
			for (int i = 0; i < this.builder.outputs.size(); i++) {
				map.put(this.builder.outputs.get(i), tensors.get(i));
			}
			this.result = new TensorResult(map);
		}
	}

	public synchronized TensorResult getResult() {
		return this.result;
	}
	
	@Override
	public synchronized TensorResult call() {
		this.run();
		return this.getResult();
	}
	
}
