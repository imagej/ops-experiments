package net.imagej.ops.experiments.filter.convolve;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.experiments.fft.MKLFFTWFloatRealForward2DWrapper;
import net.imagej.ops.experiments.filter.deconvolve.MKLRichardsonLucyOp;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;

public class InteractiveConvolveTest {

	final static String inputName = "./images/bridge.tif";

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(final String[] args) throws IOException {

		// Loader.load();
		MKLConvolveWrapper.load();

		System.out.println();
		System.out.println(System.getProperty("java.library.path"));

		ij.launch(args);

		@SuppressWarnings("unchecked")
		final Img<T> img = (Img<T>) ij.dataset().open(inputName).getImgPlus().getImg();

		final Img<DoubleType> kernel = (Img<DoubleType>) ij.op().create().kernelGauss(10, 10);

		// run MKL convolve op
		RandomAccessibleInterval<FloatType> outputMKL = (RandomAccessibleInterval<FloatType>) ij.op()
				.run(MKLConvolveOp.class, img, kernel, new long[] { 0, 0, 0 }, false);

		ij.ui().show("MKL convolved", outputMKL);

		// run MKL convolve op
		RandomAccessibleInterval<FloatType> correlatedMKL = (RandomAccessibleInterval<FloatType>) ij.op()
				.run(MKLConvolveOp.class, img, kernel, new long[] { 0, 0, 0 }, true);

		ij.ui().show("MKL correlated", correlatedMKL);

		// run Java CPP convolve op
		RandomAccessibleInterval<FloatType> outputJavaCPP = (RandomAccessibleInterval<FloatType>) ij.op()
				.run(JavaCPPConvolveOp.class, img, kernel, new long[] { 0, 0, 0 });

		ij.ui().show("JavaCPP convolved", outputJavaCPP);

	}

}
