
package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.experiments.testImages.Bars;
import net.imagej.ops.experiments.testImages.DeconvolutionTestData;
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

		final int iterations = 100;
		final int borderXY = 32;
		final int borderZ = 50;

		DeconvolutionTestData testData = new Bars();
		// DeconvolutionTestData testData = new CElegans();
		// DeconvolutionTestData testData = new HalfBead();

		testData.LoadImages(ij);
		RandomAccessibleInterval<FloatType> imgF = testData.getImg();
		RandomAccessibleInterval<FloatType> psfF = testData.getPSF();

		ij.ui().show("img ", imgF);
		ij.ui().show("psf ", psfF);

		long startTime, endTime;

		// run Cuda Richardson Lucy op

		startTime = System.currentTimeMillis();

		final RandomAccessibleInterval<FloatType> outputCuda =
			(RandomAccessibleInterval<FloatType>) ij.op().run(
				YacuDecuRichardsonLucyOp.class, imgF, psfF, new long[] { borderXY,
					borderXY, borderZ }, null, null, null, false, iterations, true);

		endTime = System.currentTimeMillis();

		ij.log().info("Total execution time cuda (decon+overhead) is: " + (endTime -
			startTime));

		ij.ui().show("cuda op deconvolved", outputCuda);
	}

}
