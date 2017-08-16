package net.imagej.ops.experiments.fft;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.special.function.Functions;
import net.imagej.ops.special.function.UnaryFunctionOp;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.complex.ComplexFloatType;
import net.imglib2.util.Util;

import org.bytedeco.javacpp.Loader;

public class InteractiveFFTTest {

	final static String inputName = "./bridge.tif";

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(final String[] args) throws IOException {

		System.out.println();
		System.out.println(System.getProperty("java.library.path"));

		ij.launch(args);

		System.out.println(System.getProperty("user.dir"));

		@SuppressWarnings("unchecked")
		final Img<T> img = (Img<T>) ij.dataset().open(inputName).getImgPlus().getImg();

		ij.ui().show("original", img);
		
		
		
		UnaryFunctionOp fftwMKL = (UnaryFunctionOp) Functions.unary(ij.op(), MKLFFTWFloatRealForward2D.class, Img.class,
				img);
		
		
		
		runFFTTest(fftwMKL,null, img);
/*
		UnaryFunctionOp fftwRF = (UnaryFunctionOp) Functions.unary(ij.op(), FFTWFloatRealForward2D.class, Img.class,
				img);

		// Make FFTW forward and inverse ops and call the test
		UnaryFunctionOp fftwRF = (UnaryFunctionOp) Functions.unary(ij.op(), FFTWFloatRealForward2D.class, Img.class,
				img);

		UnaryFunctionOp fftwRI = (UnaryFunctionOp) Functions.unary(ij.op(), FFTWFloatRealInverse2D.class,
				RandomAccessibleInterval.class, Img.class);

		runFFTTest(fftwRF, fftwRI, img);

		UnaryFunctionOp jfftRF = (UnaryFunctionOp) Functions.unary(ij.op(), JFFTFloatRealForward2D.class, Img.class,
				img);

		UnaryFunctionOp jfftRI = (UnaryFunctionOp) Functions.unary(ij.op(), JFFTFloatRealInverse2D.class,
				RandomAccessibleInterval.class, Img.class);

		runFFTTest(jfftRF, jfftRI, img);

		UnaryFunctionOp cufftRF = (UnaryFunctionOp) Functions.unary(ij.op(), CUFFTFloatRealForward2D.class, Img.class,
				img);

		UnaryFunctionOp cufftRI = (UnaryFunctionOp) Functions.unary(ij.op(), CUFFTFloatRealInverse2D.class,
				RandomAccessibleInterval.class, Img.class);

		runFFTTest(cufftRF, cufftRI, img);*/

	}

	private static <T extends RealType<T> & NativeType<T>> void runFFTTest(UnaryFunctionOp forward,
			UnaryFunctionOp inverse, Img<T> img) {
		final Img<ComplexFloatType> fftImg = (Img<ComplexFloatType>) forward.calculate(img);

		ImageJFunctions.show(fftImg).setTitle("fft power spectrum: " + forward.getClass().toString());

		if (inverse != null) {
			applyLowPass(fftImg, 40);
	
			final Img<T> filtered = (Img<T>) inverse.calculate(fftImg);
	
			ij.ui().show("low pass: " + inverse.getClass().toString(), filtered);
		}
	}

	private static <C extends ComplexType<C> & NativeType<C>> void applyLowPass(final Img<C> fft, final int radius) {
		final Cursor<C> fftCursor = fft.cursor();

		final long[] topLeft = new long[] { 0, 0 };
		final long[] bottomLeft = new long[] { 0, fft.dimension(1) };
		final long[] pos = new long[2];

		while (fftCursor.hasNext()) {
			fftCursor.fwd();
			fftCursor.localize(pos);

			final double distFromTopLeft = Util.distance(topLeft, pos);
			final double distFromBottomLeft = Util.distance(bottomLeft, pos);

			if ((distFromTopLeft > radius) && (distFromBottomLeft > radius)) {
				fftCursor.get().setZero();
			}
		}
	}
}
