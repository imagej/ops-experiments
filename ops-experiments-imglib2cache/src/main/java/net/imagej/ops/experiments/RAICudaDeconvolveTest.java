
package net.imagej.ops.experiments;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.testImages.Bars;
import net.imagej.ops.experiments.testImages.CElegans;
import net.imagej.ops.experiments.testImages.DeconvolutionTestData;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

public class RAICudaDeconvolveTest<T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	/**
	 * This examples demonstrates calling GPU deconvolution cell by cell on an
	 * image using DiskCachedCellFactory.
	 */
	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{

		ij.launch(args);

		//DeconvolutionTestData testData = new Bars();
		DeconvolutionTestData testData = new CElegans();
		// DeconvolutionTestData testData = new HalfBead();

		testData.LoadImages(ij);
		RandomAccessibleInterval<FloatType> img = testData.getImg();
		RandomAccessibleInterval<FloatType> psf = testData.getPSF();

		ImageJFunctions.show(img);
		ImageJFunctions.show(psf);

		final int iterations = 100;

		@SuppressWarnings("unchecked")
		final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver =
			(UnaryComputerOp) Computers.unary(ij.op(), UnaryComputerYacuDecuNC.class,
				RandomAccessibleInterval.class, img, psf, iterations);

		/*RandomAccessibleInterval<FloatType> out = Views.interval(img,
			new FinalInterval(new long[] { 0, 0, 0 }, new long[] { img.dimension(0),
				img.dimension(1) / 2, img.dimension(2) }));*/
		
		RandomAccessibleInterval<FloatType> out=ij.op().create().img(img);

		deconvolver.compute(img, out);

		// show the output (this will invoke deconvolution on each cell lazily).
		ImageJFunctions.show(out, "Output");

	}

}
