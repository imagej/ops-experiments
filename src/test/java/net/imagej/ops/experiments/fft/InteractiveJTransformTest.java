package net.imagej.ops.experiments.fft;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.ComplexType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.complex.ComplexFloatType;
import net.imglib2.util.Util;

public class InteractiveJTransformTest {

	final static String inputName = "./bridge.tif";

	public static <T extends RealType<T> & NativeType<T>> void main(final String[] args) throws IOException {
		final ImageJ ij = new ImageJ();
		ij.launch(args);

		System.out.println(System.getProperty("user.dir"));

		@SuppressWarnings("unchecked")
		final Img<T> input = (Img<T>) ij.dataset().open(inputName).getImgPlus().getImg();

		ij.ui().show("original", input);

		final Img<ComplexFloatType> fft = (Img<ComplexFloatType>) ij.op().run(JFFTFloatRealForward2D.class, input);

		ImageJFunctions.show(fft).setTitle("fft power spectrum");

		applyLowPass(fft, 40);

		final Img<T> output = (Img<T>) ij.op().run(JFFTFloatRealInverse2D.class, fft);

		ij.ui().show(output);

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
