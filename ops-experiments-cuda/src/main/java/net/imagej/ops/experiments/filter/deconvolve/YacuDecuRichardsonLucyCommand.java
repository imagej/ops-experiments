package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.Dataset;
import net.imagej.ImgPlus;
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
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

@Plugin(type = Command.class, headless = true, menuPath = "Plugins>OpsExperiments>YacuDecu Deconvolution")
public class YacuDecuRichardsonLucyCommand<T extends RealType<T> & NativeType<T>> implements Command {
	@Parameter
	OpService ops;

	@Parameter
	Dataset img;

	@Parameter
	Dataset psf;

	@Parameter
	Integer iterations = 100;

	@Parameter(type = ItemIO.OUTPUT)
	Img<FloatType> deconvolved;

	@Override
	public void run() {

		// convert PSF and Image to Float Type
		@SuppressWarnings("unchecked")
		Img<FloatType> imgF = ops.convert().float32((Img<T>)img.getImgPlus().getImg());
		@SuppressWarnings("unchecked")
		Img<FloatType> psfF = ops.convert().float32((Img<T>)psf.getImgPlus().getImg());

		// normalize PSF energy to 1
		float sumPSF = ops.stats().sum(psfF).getRealFloat();
		FloatType val = new FloatType();
		val.set(sumPSF);
		psfF = (Img<FloatType>) ops.math().divide(psfF, val);

		@SuppressWarnings("unchecked")
		final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver =
			(UnaryComputerOp) Computers.unary(ops, UnaryComputerYacuDecuNC.class,
				RandomAccessibleInterval.class, imgF, psfF, iterations);

		deconvolved = ops.create().img(imgF);

		deconvolver.compute(imgF, deconvolved);

	}

}
