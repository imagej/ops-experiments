
package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.Ops;
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

	@SuppressWarnings({ "unchecked", "rawtypes" })
	static <T> Img<T> project(final ImageJ ij, final Img<T> img, final int dim) {
		int d;
		final int[] projected_dimensions = new int[img.numDimensions() - 1];
		int i = 0;
		for (d = 0; d < img.numDimensions(); d++) {
			if (d != dim) {
				projected_dimensions[i] = (int) img.dimension(d);
				i += 1;
			}
		}

		final Img<T> proj = (Img<T>) ij.op().create().img(projected_dimensions);

		final UnaryComputerOp op = (UnaryComputerOp) ij.op().op(Ops.Stats.Max.NAME,
			img);

		final Img<T> projection = (Img<T>) ij.op().transform().project(proj, img,
			op, dim);
		return projection;
	}

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
		ij.ui().show("Original (YZ)", project(ij, imgF, 0));
		ij.ui().show("Deconvolved (YZ)", project(ij, res, 0));
		ij.ui().show("Original (XY)", project(ij, imgF, 2));
		ij.ui().show("Deconvolved (XY)", project(ij, res, 2));

	}

}
