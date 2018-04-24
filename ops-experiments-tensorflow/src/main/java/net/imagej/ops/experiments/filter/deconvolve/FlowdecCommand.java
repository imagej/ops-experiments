package net.imagej.ops.experiments.filter.deconvolve;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import net.imagej.ImgPlus;
import net.imagej.ops.OpService;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;

@Plugin(type = Command.class, headless = true, menuPath = "Plugins>OpsExperiments>Flowdec Deconvolution")
public class FlowdecCommand implements Command {
	@Parameter
	OpService ops;

	@SuppressWarnings("rawtypes")
	@Parameter
	ImgPlus img;

	@Parameter(type = ItemIO.INPUT)
	Integer iterations = 100;

	@Parameter(type = ItemIO.INPUT)
	Float numericalAperture = 1.4f;

	@Parameter(type = ItemIO.INPUT)
	Float wavelength = 550f;

	@Parameter(type = ItemIO.INPUT)
	Float riImmersion = 1.5f;

	@Parameter(type = ItemIO.INPUT)
	Float riSample = 1.4f;

	@Parameter(type = ItemIO.INPUT)
	Float xySpacing = 62.9f;

	@Parameter(type = ItemIO.INPUT)
	Float zSpacing = 160f;

	@Parameter(type = ItemIO.INPUT)
	Float depth = 0f;

	@SuppressWarnings("rawtypes")
	@Parameter(type = ItemIO.OUTPUT)
	Img psf;

	@SuppressWarnings("rawtypes")
	@Parameter(type = ItemIO.OUTPUT)
	Img deconvolved;

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public void run() {

		Img<FloatType> imgF = ops.convert().float32(img);

		wavelength = wavelength * 1E-9f;
		xySpacing = xySpacing * 1E-9f;
		zSpacing = zSpacing * 1E-9F;

		if (wavelength < 545E-9) {
			wavelength = 545E-9f;
		}
		// create the diffraction based psf
		psf = (Img<FloatType>) ops.create().kernelDiffraction(img, numericalAperture, wavelength, riSample, riImmersion, xySpacing,
				zSpacing, depth, new FloatType());

		deconvolved = (Img) ops.run(FlowdecOp.class, imgF, psf, iterations);

	}

}
