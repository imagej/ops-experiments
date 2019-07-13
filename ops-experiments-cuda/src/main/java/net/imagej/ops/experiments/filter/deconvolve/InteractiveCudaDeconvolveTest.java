
package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.experiments.VisualizationUtility;
import net.imagej.ops.experiments.testImages.CElegans;
import net.imagej.ops.experiments.testImages.DeconvolutionTestData;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class InteractiveCudaDeconvolveTest<T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{

		System.out.println("CWD: " + System.getProperty("user.dir"));
		final String libPathProperty = System.getProperty("java.library.path");
		System.out.println("Lib path:" + libPathProperty);

		ij.launch(args);

		ij.log().setLevel(2);

		final int iterations = 100;
		final int borderXY = 32;
		final int borderZ = 50;

		// DeconvolutionTestData testData = new Bars();
		DeconvolutionTestData testData = new CElegans("../images/");
		// DeconvolutionTestData testData = new HalfBead();

		testData.LoadImages(ij);
		RandomAccessibleInterval<FloatType> imgF = testData.getImg();
		RandomAccessibleInterval<FloatType> psfF = testData.getPSF();

		ij.ui().show("img ", imgF);
		ij.ui().show("psf ", psfF);

		long startTime, endTime;

		// run Cuda Richardson Lucy op

		@SuppressWarnings("unchecked")
		final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver =
			(UnaryComputerOp) Computers.unary(ij.op(), UnaryComputerYacuDecu.class,
				RandomAccessibleInterval.class, imgF, psfF, iterations);

		startTime = System.currentTimeMillis();

		/*final RandomAccessibleInterval<FloatType> deconvolved =
			(RandomAccessibleInterval<FloatType>) ij.op().run(
				YacuDecuRichardsonLucyOp.class, imgF, psfF, new long[] { borderXY,
					borderXY, borderZ }, null, null, null, false, iterations, true);*/

		RandomAccessibleInterval<FloatType> deconvolved = ij.op().create().img(
			imgF);

		deconvolver.compute(imgF, deconvolved);

		endTime = System.currentTimeMillis();

		ij.log().info("Total execution time cuda (decon+overhead) is: " + (endTime -
			startTime));

		ij.ui().show("cuda op deconvolved", deconvolved);

		// Render projections along X and Z axes
		ij.ui().show("Original (YZ)", VisualizationUtility.project(ij, imgF, 0));
		ij.ui().show("Deconvolved (YZ)", VisualizationUtility.project(ij,
			deconvolved, 0));
		ij.ui().show("Original (XY)", VisualizationUtility.project(ij, imgF, 2));
		ij.ui().show("Deconvolved (XY)", VisualizationUtility.project(ij,
			deconvolved, 2));
	}

}
