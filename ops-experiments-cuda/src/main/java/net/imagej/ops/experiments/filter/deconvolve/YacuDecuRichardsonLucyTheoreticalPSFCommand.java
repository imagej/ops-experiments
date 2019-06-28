package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.ImgPlus;
import net.imagej.ops.OpService;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

@Plugin(type = Command.class, headless = true, menuPath = "Plugins>OpsExperiments>YacuDecu Theoretical PSF")
public class YacuDecuRichardsonLucyTheoreticalPSFCommand<T extends RealType<T> & NativeType<T>> implements Command {
	@Parameter
	OpService ops;

	@Parameter
	ImgPlus<T> img;

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
	Img<FloatType> psf;

	@Parameter(type = ItemIO.OUTPUT)
	Img<FloatType> deconvolved;

	@Override
	public void run() {

		Img<FloatType> imgF = ops.convert().float32(img);

		wavelength = wavelength * 1E-9f;
		xySpacing = xySpacing * 1E-9f;
		zSpacing = zSpacing * 1E-9F;

		if (wavelength < 545E-9) {
			wavelength = 545E-9f;
		}
		
		FinalDimensions psfDims=new FinalDimensions(64,64,50);
		// create the diffraction based psf
		psf =  ops.create().kernelDiffraction(psfDims, numericalAperture, wavelength, riSample, riImmersion, xySpacing,
				zSpacing, depth, new FloatType());

		// normalize PSF energy to 1
		float sumPSF = ops.stats().sum(psf).getRealFloat();
		FloatType val = new FloatType();
		val.set(sumPSF);
		psf = (Img<FloatType>) ops.math().divide(psf, val);

		@SuppressWarnings("unchecked")
		final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver =
			(UnaryComputerOp) Computers.unary(ops, UnaryComputerYacuDecuNC.class,
				RandomAccessibleInterval.class, imgF, psf, iterations);

		deconvolved = ops.create().img(imgF);

		deconvolver.compute(imgF, deconvolved);

	}

}
