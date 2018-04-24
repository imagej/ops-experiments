package net.imagej.ops.experiments.filter.deconvolve;

import java.nio.file.Path;
import java.nio.file.Paths;

import org.tensorflow.Graph;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.MetaGraphDef;

import com.google.protobuf.InvalidProtocolBufferException;

public class FlowdecGraph {

	private final MetaGraphDef metaGraph;
	private Graph computationGraph;
	
	FlowdecGraph(MetaGraphDef metaGraph, Graph computationGraph) {
		super();
		this.metaGraph = metaGraph;
		this.computationGraph = computationGraph;
	}

	private static Path getTensorFlowGraphDir() {
		return Paths.get("src", "main", "resources", "tensorflow", "graphs");
	}
	
	public static FlowdecGraph load(String graphDir) {
		Path path = getTensorFlowGraphDir()
				.resolve(graphDir)
				.normalize()
				.toAbsolutePath();
		
		SavedModelBundle model;
		MetaGraphDef metaGraph;
		Graph computationGraph;
		try {
			model = SavedModelBundle.load(path.toString(), 
					Flowdec.DEFAULT_SERVING_KEY);
			metaGraph = MetaGraphDef.parseFrom(model.metaGraphDef());
			computationGraph = model.graph();
		} catch (InvalidProtocolBufferException e) {
			throw new RuntimeException("Failed to retrieve meta graph from "
					+ "saved model at '" + path + "'", e);
		}
		
		return new FlowdecGraph(metaGraph, computationGraph);
	}
	
	public MetaGraphDef getMetaGraph() {
		return metaGraph;
	}

	public Graph getComputationGraph() {
		return computationGraph;
	}
	
	/**
	 * Set device associated with TensorFlow graph execution
	 * 
	 * @param device Name of device to place execution on; e.g. "/cpu:0", "/gpu:1", etc.
	 * See <a href="https://www.tensorflow.org/programmers_guide/faq">TF FAQ</a> for more details.
	 * @return this
	 */
	public FlowdecGraph setDevice(String device) {
		this.computationGraph = setDevice(this.computationGraph, device);
		return this;
	}
	
	static Graph setDevice(Graph g, String device) {
		
		// This functionality is based entirely on:
		// https://stackoverflow.com/questions/47799972/tensorflow-java-multi-gpu-inference

		GraphDef.Builder builder;
		try {
			builder = GraphDef.parseFrom(g.toGraphDef()).toBuilder();
		} catch (InvalidProtocolBufferException e) {
			throw new RuntimeException(e);
		}
		
		for (int i = 0; i < builder.getNodeCount(); ++i) {
			builder.getNodeBuilder(i).setDevice(device);
		}
		
		Graph gd = new Graph();
		gd.importGraphDef(builder.build().toByteArray());
		return gd;
	}
}
