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

	final static String inputName = "./bridge.tif";

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
				.run(MKLConvolveOp.class, img, kernel, new long[] { 0, 0, 0 });

		ij.ui().show("MKL convolved", outputMKL);

		// run Java CPP convolve op
		RandomAccessibleInterval<FloatType> outputJavaCPP = (RandomAccessibleInterval<FloatType>) ij.op()
				.run(JavaCPPConvolveOp.class, img, kernel, new long[] { 0, 0, 0 });

		ij.ui().show("JavaCPP convolved", outputMKL);
		/*
		 * RandomAccessibleInterval<T> extendedImg =
		 * ij.op().filter().padFFTInput(img, new
		 * FinalDimensions(img.dimension(0), img.dimension(1)));
		 * 
		 * ij.ui().show("original", Views.zeroMin(extendedImg));
		 * 
		 * 
		 * 
		 * ij.ui().show(kernel);
		 * 
		 * RandomAccessibleInterval<DoubleType> extendedKernel =
		 * ij.op().filter().padShiftFFTKernel(kernel, new
		 * FinalDimensions(img.dimension(0), img.dimension(1)));
		 * 
		 * //ij.ui().show(extended);
		 * 
		 * ij.ui().show(Views.zeroMin(extendedKernel));
		 * 
		 * 
		 * // Convolve
		 * 
		 * // convert to FloatPointer final FloatPointer x =
		 * ConvertersUtility.ii2DToFloatPointer(Views.zeroMin(extendedImg));
		 * 
		 * // convert to FloatPointer final FloatPointer h =
		 * ConvertersUtility.ii2DToFloatPointer(Views.zeroMin(extendedKernel));
		 * 
		 * 
		 * // output size of FFT see //
		 * http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data
		 * .html final long[] fftSize = new long[] { extendedImg.dimension(0) /
		 * 2 + 1, extendedImg.dimension(1) };
		 * 
		 * final FloatPointer X_ = new FloatPointer(2 *
		 * (extendedImg.dimension(0) / 2 + 1) * extendedImg.dimension(1));
		 * 
		 * final FloatPointer H_ = new FloatPointer(2 *
		 * (extendedImg.dimension(0) / 2 + 1) * extendedImg.dimension(1));
		 * 
		 * 
		 * MKLConvolveWrapper.mklConvolve(x, h, X_, H_,
		 * (int)extendedImg.dimension(0), (int)extendedImg.dimension(1));
		 * 
		 * final float[] convolved = new
		 * float[(int)(extendedImg.dimension(0)*extendedImg.dimension(1)) ];
		 * 
		 * x.get(convolved);
		 * 
		 * FloatPointer.free(x); FloatPointer.free(h); FloatPointer.free(X_);
		 * FloatPointer.free(H_);
		 * 
		 * long[] test=new long[]{extendedImg.dimension(0),
		 * extendedImg.dimension(1)};
		 * 
		 * Img out=ArrayImgs.floats(convolved, test);
		 * 
		 * ij.ui().show(out);
		 */

		//

	}

}
