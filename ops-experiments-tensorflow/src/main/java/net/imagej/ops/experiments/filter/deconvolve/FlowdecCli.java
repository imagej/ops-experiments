package net.imagej.ops.experiments.filter.deconvolve;

import java.util.Optional;
import java.util.logging.Logger;

import org.tensorflow.Tensors;

import ij.IJ;
import ij.ImagePlus;
import ij.plugin.Concatenator;
import ij.process.FloatProcessor;
import net.imagej.ops.experiments.filter.deconvolve.Flowdec.TensorResult;

/**
 * This is a throw-away testing class for running 3D deconvolution, primarily
 * used as a way to run on gpu-enabled machines with no graphics drivers 
 * (i.e. non-desktop linux servers).
 * 
 * @author eczech
 */
public class FlowdecCli {
	
	static final Logger log = Logger.getLogger(FlowdecCli.class.getSimpleName());
	
	public static void main(String[] args) {
		// I don't know preferred logging config in scijava/imagej so hack this in for now
	    System.setProperty("java.util.logging.SimpleFormatter.format",
	            "%1$tY-%1$tm-%1$td %1$tH:%1$tM:%1$tS %4$-6s %2$s %5$s%6$s%n");
	    
		if (args.length != 4) {
			throw new IllegalArgumentException("Expecting 4 arguments "
					+ "[image_path, psf_path, n_iter, output_path], given " + args.length);
		}
		String imgPath = args[0];
		String psfPath = args[1];
		int niter = Integer.valueOf(args[2]);
		String outPath = args[3];

		log.info("Loading image and PSF");
		FlowdecTask.Builder task = Flowdec.richardsonLucy().task(
				Tensors.create(toFloatArray(IJ.openImage(imgPath))), 
				Tensors.create(toFloatArray(IJ.openImage(psfPath))),
				niter,
				Optional.ofNullable(null), 
				Optional.ofNullable(null));

		log.info("Running deconvolution");
		long start = System.currentTimeMillis();
		TensorResult res = task.build().call();
		long end = System.currentTimeMillis();
		log.info(String.format("Deconvolution complete in %.3f seconds", (end - start) / 1000.));
		
		log.info("Saving result to '" + outPath + "'");
		ImagePlus img = toImage(res.data().float3d());
		IJ.save(img, outPath);
		
		log.info("Done");
		System.exit(0);
	}
	
	static float[][][] toFloatArray(ImagePlus img) {
		int nz = img.getStackSize();
		float[][][] d = new float[nz][][];
		for (int z = 1; z <= nz; z++) {
			d[z-1] = img.getImageStack().getProcessor(z)
						.convertToFloat().getFloatArray();
		}
		return d;	
	}
	
	static ImagePlus toImage(float[][][] data) {
		if (data.length == 0) {
			throw new IllegalArgumentException("Cannot create image from empty array");
		}
		ImagePlus[] stack = new ImagePlus[data.length];
		for (int z = 0; z < data.length; z++) {
			stack[z] = new ImagePlus("Slice " + z, new FloatProcessor(data[z]));
		}
		return new Concatenator().concatenate(stack, true);
	}

	
}
