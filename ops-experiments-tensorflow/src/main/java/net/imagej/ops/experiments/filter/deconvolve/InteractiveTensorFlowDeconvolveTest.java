
package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.VisualizationUtility;
import net.imagej.ops.experiments.testImages.Bars;
import net.imagej.ops.experiments.testImages.CElegans;
import net.imagej.ops.experiments.testImages.DeconvolutionTestData;
import net.imagej.ops.experiments.testImages.HalfBead;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class InteractiveTensorFlowDeconvolveTest {

	@SuppressWarnings({ "unchecked", "deprecation" })
	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws InterruptedException, IOException
	{

		// create an instance of imagej
		final ImageJ ij = new ImageJ();

		// launch it
		ij.launch(args);
		
		// TODO: Modify tensoflow RL so it can take float images as input
		DeconvolutionTestData testData = new Bars();
		//DeconvolutionTestData testData = new CElegans();
		//DeconvolutionTestData testData = new HalfBead();

		testData.LoadImages(ij);
		Img<FloatType> imgF = (Img<FloatType>)testData.getImg();
		Img<FloatType> psfF = (Img<FloatType>)testData.getPSF();


		final int iterations = 100;
		final int[] pad = new int[] { 20, 20, 20 };

		final long startTime = System.currentTimeMillis();

		final Img<FloatType> res = (Img<FloatType>) ij.op().run(FlowdecOp.class,
			imgF, psfF, iterations, pad);

		final long endTime = System.currentTimeMillis();

		ij.log().info("Total execution time tensorflow (decon+overhead) is: " +
			(endTime - startTime));

		// Render projections along X and Z axes
		ij.ui().show("Original (YZ)", VisualizationUtility.project(ij, imgF, 0));
		ij.ui().show("Deconvolved (YZ)", VisualizationUtility.project(ij, res, 0));
		ij.ui().show("Original (XY)", VisualizationUtility.project(ij, imgF, 2));
		ij.ui().show("Deconvolved (XY)", VisualizationUtility.project(ij, res, 2));

	}

}
