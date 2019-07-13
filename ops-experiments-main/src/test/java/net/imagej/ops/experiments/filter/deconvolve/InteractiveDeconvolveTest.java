
package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.experiments.testImages.Bars;
import net.imagej.ops.experiments.testImages.DeconvolutionTestData;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class InteractiveDeconvolveTest<T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{

		final String libPathProperty = System.getProperty("java.library.path");
		System.out.println("Lib path:" + libPathProperty);

		ij.launch(args);

		DeconvolutionTestData testData = new Bars("../images/");
		// DeconvolutionTestData testData = new CElegans();
		// DeconvolutionTestData testData = new HalfBead();

		testData.LoadImages(ij);
		RandomAccessibleInterval<FloatType> imgF = testData.getImg();
		RandomAccessibleInterval<FloatType> psfF = testData.getPSF();

		ij.ui().show("bars ", imgF);
		ij.ui().show("psf ", psfF);

		final int iterations = 100;
		final int pad = 20;

		// run Ops Richardson Lucy

		final long startTime = System.currentTimeMillis();

		//Img<FloatType> out = (Img<FloatType>) ij.op().create().img(imgF);
		
		final Img<FloatType> deconvolved = (Img<FloatType>) ij.op().deconvolve()
			.richardsonLucy(imgF, psfF, new long[] { pad, pad, pad }, null, null,
				null, null, iterations, false, false);

		final long endTime = System.currentTimeMillis();

		ij.log().info("Total execution time (Ops) is: " + (endTime - startTime));

		ij.ui().show("Richardson Lucy deconvolved", deconvolved);

		/*
		
		// run Cuda Richardson Lucy op
		
		startTime = System.currentTimeMillis();
		
		final RandomAccessibleInterval<FloatType> outputCuda = (RandomAccessibleInterval<FloatType>) ij.op()
				.run(YacuDecuRichardsonLucyOp.class, imgF, psfF, new long[] { pad, pad, pad }, iterations);
		
		endTime = System.currentTimeMillis();
		
		ij.log().info("Total execution time (Cuda) is: " + (endTime - startTime));
		
		
		ij.ui().show("cuda op deconvolved", outputCuda);
		
		// run MKL Richardson Lucy
		
		startTime = System.currentTimeMillis();
		
		// run MKL Richardson Lucy op
		final RandomAccessibleInterval<FloatType> outputMKL = (RandomAccessibleInterval<FloatType>) ij.op()
				.run(MKLRichardsonLucyOp.class, imgF, psfF, new long[]{pad,pad,pad}, null, null, null, iterations, true);
		
		endTime = System.currentTimeMillis();
		
		
		ij.log().info("Total execution time (MKL) is: " + (endTime - startTime));
		
		
		ij.ui().show("mkl op deconvolved", outputMKL);
		*/

	}

}
