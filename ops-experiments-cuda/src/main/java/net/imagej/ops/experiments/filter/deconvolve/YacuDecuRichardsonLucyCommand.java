package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.ImgPlus;
import net.imagej.ops.OpService;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

@Plugin(type = Command.class, headless = true, menuPath = "Plugins>OpsExperiments>YacuDecu Deconvolution")
public class YacuDecuRichardsonLucyCommand implements Command {
	@Parameter
	OpService ops;

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

	@Parameter(type = ItemIO.OUTPUT)
	Img psf;

	@Parameter(type = ItemIO.OUTPUT)
	Img deconvolved;

	public void run() {

		Img<FloatType> imgF = ops.convert().float32(img);

		wavelength = wavelength * 1E-9f;
		xySpacing = xySpacing * 1E-9f;
		zSpacing = zSpacing * 1E-9F;

		if (wavelength < 545E-9) {
			wavelength = 545E-9f;
		}
		// create the diffraction based psf
		psf = (Img) ops.create().kernelDiffraction(img, numericalAperture, wavelength, riSample, riImmersion, xySpacing,
				zSpacing, depth, new FloatType());

		// normalize PSF energy to 1
		float sumPSF = ops.stats().sum(psf).getRealFloat();
		FloatType val = new FloatType();
		val.set(sumPSF);
		psf = (Img<FloatType>) ops.math().divide(psf, val);

		deconvolved = (Img) ops.run(YacuDecuRichardsonLucyOp.class, imgF, psf, new long[] { 0, 0, 0 }, iterations);

	}

}
