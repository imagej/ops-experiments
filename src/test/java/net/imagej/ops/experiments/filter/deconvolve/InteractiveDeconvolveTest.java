package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.Ops.Filter.PadShiftFFTKernel;
import net.imagej.ops.experiments.ConvertersUtility;
import net.imagej.ops.experiments.filter.convolve.MKLConvolveWrapper;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.bytedeco.javacpp.FloatPointer;

public class InteractiveDeconvolveTest<T extends RealType<T> & NativeType<T>> {

	final static String inputName = "../ops-images/deconvolvolution/Bars-G10-P15-stack.tif";
	final static String psfName = "../ops-images/deconvolvolution/PSF-Bars-stack.tif";

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(final String[] args) throws IOException {
		String libPathProperty = System.getProperty("java.library.path");
		System.out.println("Lib path:"+libPathProperty);

		ij.launch(args);
		
		@SuppressWarnings("unchecked")
		Img<T> img = (Img<T>) ij.dataset().open(inputName).getImgPlus().getImg();
		Img<FloatType> imgF = ij.op().convert().float32(img);

		@SuppressWarnings("unchecked")
		final Img<T> psf = (Img<T>) ij.dataset().open(psfName).getImgPlus().getImg();

		// convert PSF to float
		Img<FloatType> psfF = ij.op().convert().float32(psf);
		
		// normalize PSF
		FloatType sum = new FloatType(ij.op().stats().sum(psfF).getRealFloat());
		psfF = (Img<FloatType>) ij.op().math().divide(psfF, sum);

		// shift PSF so that the center is at 0,0,0
		RandomAccessibleInterval<FloatType> shiftedPSF = (RandomAccessibleInterval<FloatType>) ij.op().run(
				PadShiftFFTKernel.class, psfF,
				new FinalDimensions(img.dimension(0), img.dimension(1), img.dimension(2)));

		ij.ui().show("shifted PSF", Views.zeroMin(shiftedPSF));

		ij.ui().show("bars", img);
		ij.ui().show("psfF", psf);
		
		int iterations=100;

		testOpsRL(imgF, psfF, iterations);
		testCudaRL(imgF, shiftedPSF, iterations);
		//testMKLRL(imgF, shiftedPSF, iterations);
		//testMKLRL(imgF, shiftedPSF, iterations);
		
	}

	public static <T extends RealType<T> & NativeType<T>> void testCudaRL(Img<FloatType> img,
			RandomAccessibleInterval<FloatType> shiftedPSFF, int iterations) throws IOException {

		ij.log().info("Cuda Richardson Lucy");

		CudaRichardsonLucyWrapper.load();

		// convert image to FloatPointer
		final FloatPointer x = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(img));

		// convert PSF to FloatPointer
		final FloatPointer h = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(shiftedPSFF));

		final FloatPointer y = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(img));

		ij.log().info("starting Cuda decon\n");

		final long startTime = System.currentTimeMillis();

		CudaRichardsonLucyWrapper.deconv_device(iterations, (int) img.dimension(2), (int) img.dimension(1),
				(int) img.dimension(0), x, h, y);

		final long endTime = System.currentTimeMillis();

		ij.log().info("Total execution time (Cuda) is: " + (endTime - startTime));

		final float[] deconvolved = new float[(int) (img.dimension(0) * img.dimension(1) * img.dimension(2))];

		y.get(deconvolved);

		FloatPointer.free(x);
		FloatPointer.free(h);
		FloatPointer.free(y);

		long[] imgSize = new long[] { img.dimension(0), img.dimension(1), img.dimension(2) };

		Img out = ArrayImgs.floats(deconvolved, imgSize);

		ij.ui().show("Cuda Deconvolved", out);

	}

	public static <T extends RealType<T> & NativeType<T>> void testMKLRL(Img<FloatType> img,
			RandomAccessibleInterval<FloatType> psfShiftF, int iterations) throws IOException {

		MKLRichardsonLucyWrapper.load();

		// convert image to FloatPointer
		final FloatPointer x = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(img));

		// convert PSF to FloatPointer
		final FloatPointer h = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(psfShiftF));

		final FloatPointer y = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(img));

		final long[] fftSize = new long[] { img.dimension(0) / 2 + 1, img.dimension(1), img.dimension(2) };

		final FloatPointer X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));

		final FloatPointer H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));

		ij.log().info("Starting MKL decon");

		final long startTime = System.currentTimeMillis();

		MKLRichardsonLucyWrapper.mklRichardsonLucy3D(iterations, x, h, y, X_, H_, (int) img.dimension(2), (int) img.dimension(1),
				(int) img.dimension(0));

		final long endTime = System.currentTimeMillis();

		ij.log().info("Total execution time (MKL) is: " + (endTime - startTime));

		final float[] deconvolved = new float[(int) (img.dimension(0) * img.dimension(1) * img.dimension(2))];

		ij.log().info("start");
		
		y.get(deconvolved);

		ij.log().info("1");
		
		FloatPointer.free(X_);
		ij.log().info("2");
		
		FloatPointer.free(H_);
		ij.log().info("3");
		
		FloatPointer.free(x);
		ij.log().info("4");
		
		FloatPointer.free(h);
		ij.log().info("5");
		
		FloatPointer.free(y);

		long[] imgSize = new long[] { img.dimension(0), img.dimension(1), img.dimension(2) };

		Img out = ArrayImgs.floats(deconvolved, imgSize);

		ij.ui().show("deconvolved", out);

	}

	public static <T extends RealType<T> & NativeType<T>> void testOpsRL(Img<FloatType> imgF, Img<FloatType> psfF, int iterations)
			throws IOException {

		ij.log().info("Starting Ops Richardson Lucy");

		final long startTime = System.currentTimeMillis();

		Img<FloatType> deconvolved = (Img<FloatType>) ij.op().deconvolve().richardsonLucy(imgF, psfF,
				new long[] { 0, 0, 0 }, iterations);

		final long endTime = System.currentTimeMillis();

		ij.log().info("Total execution time (Ops) is: " + (endTime - startTime));

		ij.ui().show("Richardson Lucy deconvolved", deconvolved);

	}

}
