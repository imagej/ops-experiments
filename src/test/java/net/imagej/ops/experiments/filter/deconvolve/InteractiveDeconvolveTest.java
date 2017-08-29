package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.imagej.ImageJ;
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

public class InteractiveDeconvolveTest {

	final static String inputName = "../ops-images/deconvolvolution/Bars-G10-P15-stack.tif";
	final static String psfName = "../ops-images/deconvolvolution/PSF-Bars-stack.tif";

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(final String[] args) throws IOException {

		ij.launch(args);
		
		ij.log().error("Start Richardson Lucy");
		
		MKLRichardsonLucyWrapper.load();

		@SuppressWarnings("unchecked")
		final Img<T> img = (Img<T>) ij.dataset().open(inputName).getImgPlus().getImg();

		@SuppressWarnings("unchecked")
		final Img<T> psf = (Img<T>) ij.dataset().open(psfName).getImgPlus().getImg();
		Img<FloatType> psfF=ij.op().convert().float32(psf);
		
		FloatType sum=new FloatType(ij.op().stats().sum(psfF).getRealFloat());
		psfF=(Img<FloatType>)ij.op().math().divide(psfF, sum);

		ij.ui().show("bars", img);
		ij.ui().show("psf", psfF);

		RandomAccessibleInterval<T> extendedImg = ij.op().filter().padFFTInput(img,
				new FinalDimensions(img.dimension(0), img.dimension(1), img.dimension(2)));

		ij.ui().show("extended image", Views.zeroMin(extendedImg));

		RandomAccessibleInterval<FloatType> extendedPSF = ij.op().filter().padShiftFFTKernel(psfF,
				new FinalDimensions(img.dimension(0), img.dimension(1), img.dimension(2)));

		// ij.ui().show(extended);

		ij.ui().show("extended PSF",Views.zeroMin(extendedPSF));

		// convert image to FloatPointer
		final FloatPointer x = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(extendedImg));

		// convert PSF to FloatPointer
		final FloatPointer h = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(extendedPSF));

		final FloatPointer y = ConvertersUtility.ii3DToFloatPointer(Views.zeroMin(extendedImg));

		// output size of FFT see
		// http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html
		final long[] fftSize = new long[] { extendedImg.dimension(0)/2+1, extendedImg.dimension(1),
				extendedImg.dimension(2) };

		final FloatPointer X_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));

		final FloatPointer H_ = new FloatPointer(2 * (fftSize[0] * fftSize[1] * fftSize[2]));

		
		MKLRichardsonLucyWrapper.mklRichardsonLucy3D(x, h, y, X_, H_, (int) extendedImg.dimension(2),
				(int) extendedImg.dimension(1), (int) extendedImg.dimension(0));
		
		final float[] deconvolved = new float[(int)(extendedImg.dimension(0)*extendedImg.dimension(1)*extendedImg.dimension(2)) ];
		
		y.get(deconvolved);

		FloatPointer.free(x);
		FloatPointer.free(h);
		FloatPointer.free(y);
		FloatPointer.free(X_);
		FloatPointer.free(H_);
		
		long[] imgSize=new long[]{extendedImg.dimension(0), extendedImg.dimension(1), extendedImg.dimension(2)};

		Img out=ArrayImgs.floats(deconvolved, imgSize);
		
		ij.ui().show("deconvolved",out);

	}

}
