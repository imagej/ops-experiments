
package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.experiments.testImages.Bars;
import net.imagej.ops.experiments.testImages.DeconvolutionTestData;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.bytedeco.javacpp.FloatPointer;

public class InteractiveArrayFireTests<T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{

		ArrayFireWrapper.load();

		System.out.println("CWD: " + System.getProperty("user.dir"));
		final String libPathProperty = System.getProperty("java.library.path");
		System.out.println("Java Library path:" + libPathProperty);
		System.out.println();
		System.out.println("System Library path " + System.getenv(
			"LD_LIBRARY_PATH"));
		System.out.println();
		System.out.println("System path " + System.getenv("PATH"));
		ij.launch(args);

		ij.log().setLevel(2);

		final int iterations = 100;
		final int borderXY = 32;
		final int borderZ = 50;

		DeconvolutionTestData testData = new Bars("../images/");
		// DeconvolutionTestData testData = new CElegans("../images/");
		// DeconvolutionTestData testData = new Bead("../images/");

		testData.LoadImages(ij);
		RandomAccessibleInterval<FloatType> imgF = testData.getImg();
		RandomAccessibleInterval<FloatType> psfF = testData.getPSF();

		ij.ui().show("img ", imgF);
		ij.ui().show("psf ", psfF);

		long startTime, endTime;

		RandomAccessibleInterval<FloatType> extendedPSF = Views.zeroMin(ij.op()
			.filter().padShiftFFTKernel(psfF, imgF));

		ij.ui().show("padded shifted PSF ", extendedPSF);

		long start = System.currentTimeMillis();
		FloatPointer fpInput = null;
		FloatPointer fpPSF = null;
		FloatPointer fpOutput = null;

		// convert image to FloatPointer
		fpInput = ConvertersUtility.ii3DToFloatPointer(imgF);

		// convert PSF to FloatPointer
		fpPSF = ConvertersUtility.ii3DToFloatPointer(extendedPSF);
		long finish = System.currentTimeMillis();

		System.out.println("Conversion Time: " + (finish - start));

		start = System.currentTimeMillis();

		int paddedSize = (int) (imgF.dimension(0) * imgF.dimension(1) * imgF
			.dimension(2));

		fpOutput = ConvertersUtility.ii3DToFloatPointer(imgF);

		finish = System.currentTimeMillis();

		ArrayFireWrapper.conv2(imgF.dimension(0), imgF.dimension(1), imgF.dimension(
			2), fpInput, fpPSF, fpOutput);

		// copy output to array
		final float[] arrayOutput = new float[paddedSize];
		fpOutput.get(arrayOutput);

		final Img<FloatType> conv = ArrayImgs.floats(arrayOutput, new long[] { imgF
			.dimension(0), imgF.dimension(1), imgF.dimension(2) });

		ij.ui().show(conv);

	}

}
