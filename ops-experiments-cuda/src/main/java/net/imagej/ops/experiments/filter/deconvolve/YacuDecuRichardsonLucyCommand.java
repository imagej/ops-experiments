
package net.imagej.ops.experiments.filter.deconvolve;

import io.scif.img.ImgSaver;
import io.scif.services.DatasetIOService;

import java.io.File;

import net.imagej.Dataset;
import net.imagej.ops.OpService;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

@Plugin(type = Command.class, headless = true,
	menuPath = "Plugins>OpsExperiments>YacuDecu Deconvolution")
public class YacuDecuRichardsonLucyCommand<T extends RealType<T> & NativeType<T>>
	implements Command
{

	@Parameter
	LogService log;

	@Parameter
	OpService ops;

	@Parameter
	DatasetIOService dio;

	@Parameter
	Dataset img;

	@Parameter
	Dataset psf;

	@Parameter
	Integer iterations = 100;

	@Parameter(required = false, style = "directory")
	File outputDir = null;

	@Parameter(type = ItemIO.OUTPUT)
	Img<FloatType> deconvolved;

	@Override
	public void run() {

		// log.setLevel(2);
		log.error("show log");
		// System.setProperty("scijava.log.level", "debug");

		// convert PSF and Image to Float Type
		@SuppressWarnings("unchecked")
		Img<FloatType> imgF = ops.convert().float32((Img<T>) img.getImgPlus()
			.getImg());
		@SuppressWarnings("unchecked")
		Img<FloatType> psfF = ops.convert().float32((Img<T>) psf.getImgPlus()
			.getImg());

		// normalize PSF energy to 1
		float sumPSF = ops.stats().sum(psfF).getRealFloat();
		FloatType val = new FloatType();
		val.set(sumPSF);
		psfF = (Img<FloatType>) ops.math().divide(psfF, val);

		@SuppressWarnings("unchecked")
		final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver =
			(UnaryComputerOp) Computers.unary(ops, UnaryComputerYacuDecu.class,
				RandomAccessibleInterval.class, imgF, psfF, iterations);

		log.info("Processing " + img.getImgPlus().getName());

		deconvolved = ops.create().img(imgF);

		deconvolver.compute(imgF, deconvolved);

		if (outputDir != null) {
			String outName = outputDir.getAbsolutePath() + "/deconvolved_" + img
				.getName();
			log.info("saving to" + outName);
			new ImgSaver().saveImg(outName, deconvolved);
		}
	}
}
