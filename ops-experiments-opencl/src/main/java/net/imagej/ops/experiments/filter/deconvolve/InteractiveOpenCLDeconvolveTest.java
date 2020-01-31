
package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.imagej.ImageJ;
import net.imagej.ops.experiments.testImages.Bars;
import net.imagej.ops.experiments.testImages.DeconvolutionTestData;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

public class InteractiveOpenCLDeconvolveTest<T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{
		// check the library path, can be useful for debugging
		System.out.println(System.getProperty("java.library.path"));

		ij.launch(args);

		// get the test data
		DeconvolutionTestData testData = new Bars("../images/");

		testData.LoadImages(ij);
		RandomAccessibleInterval<FloatType> imgF = testData.getImg();
		RandomAccessibleInterval<FloatType> psfF = testData.getPSF();

		// take a look at it
		ij.ui().show("img ", imgF);
		ij.ui().show("psf ", psfF);

		// get CLIJ
		CLIJ clij = CLIJ.getInstance();
		
		System.out.println(CLIJ.getAvailableDeviceNames());

		// transfer image to the GPU
		ClearCLBuffer gpuImg = clij.push(imgF);

		ClearCLBuffer gpuEstimate = OpenCLFFTUtility.runDecon(gpuImg, psfF, ij
			.op());
		
		// show the result
		clij.show(gpuEstimate, "GPU Decon Result");

	}
}
