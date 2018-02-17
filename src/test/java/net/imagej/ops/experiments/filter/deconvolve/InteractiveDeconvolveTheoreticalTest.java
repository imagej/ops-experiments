package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class InteractiveDeconvolveTheoreticalTest<T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(final String[] args) throws IOException {

		String libPathProperty = System.getProperty("java.library.path");
		System.out.println("Lib path:" + libPathProperty);

		ij.launch(args);

		// String inputName =
		// "C:/Users/bnorthan/Dropbox/Deconvolution_Test_Set/FromJeanYves/Slide_17015-02_512_2.tif";
		// Dimensions psfDimensions=new FinalDimensions(64,64,50);

		// double numericalAperture=1.4;
		// double wavelength=550E-09;
		// double riImmersion=1.5f;
		// double riSample=1.3f;
		// double xySpacing=162.5E-9;
		// double zSpacing=280E-9;
		// double depth=0;

		// String inputName =
		// "../ops-images/deconvolution/CElegans-CY3-crop.tif";
		// Dimensions psfDimensions = new FinalDimensions(65, 65, 128);

		// double numericalAperture = 1.4;
		// double wavelength = 654E-09;
		// double riImmersion = 1.5f;
		// double riSample = 1.4f;
		// double xySpacing = 64.5E-9;
		// double zSpacing = 160E-9;
		// double depth = 0;

		String inputName = "C:/Users/bnorthan/Dropbox/Deconvolution_Test_Set/McNamara/GM 20131101Fri_StellarisFISH_1_w61 = DAPI ROI.tif";

		// Dimensions psfDimensions = new FinalDimensions(65, 65, 128);
		Dimensions psfDimensions = new FinalDimensions(256, 256, 128);

		double numericalAperture = 1.4;
		double wavelength = 550E-09;
		double riImmersion = 1.5f;
		double riSample = 1.4f;
		double xySpacing = 62.9E-9;
		double zSpacing = 160E-9;
		double depth = 0;
		// double depth=6200

		// open image and convert to 32 bit
		@SuppressWarnings("unchecked")
		Img<T> img = (Img<T>) ij.dataset().open(inputName).getImgPlus().getImg();
		Img<FloatType> imgF = ij.op().convert().float32(img);

		// create the diffraction based psf
		Img<FloatType> psf = (Img) ij.op().create().kernelDiffraction(psfDimensions, numericalAperture, wavelength,
				riSample, riImmersion, xySpacing, zSpacing, depth, new FloatType());

		// normalize PSF energy to 1
		float sumPSF = ij.op().stats().sum(psf).getRealFloat();
		FloatType val = new FloatType();
		val.set(sumPSF);
		psf = (Img<FloatType>) ij.op().math().divide(psf, val);

		ij.ui().show("bars ", img);
		ij.ui().show("psf", psf);

		int iterations = 100;
		int pad = 0;

		long startTime, endTime;

		// run Ops Richardson Lucy

		boolean opsRL = false;
		boolean mklRL = false;
		boolean cudaRL = true;

		if (opsRL) {

			startTime = System.currentTimeMillis();

			Img<FloatType> deconvolved = (Img<FloatType>) ij.op().deconvolve().richardsonLucy(imgF, psf, null, null,
					null, null, null, 30, true, true);

			endTime = System.currentTimeMillis();

			ij.log().info("Total execution time (Cuda) is: " + (endTime - startTime));

			ij.ui().show("java op deconvolved", deconvolved);

		}

		// run Cuda Richardson Lucy op

		if (cudaRL) {
			startTime = System.currentTimeMillis();

			RandomAccessibleInterval<FloatType> outputCuda = (RandomAccessibleInterval<FloatType>) ij.op()
					.run(YacuDecuRichardsonLucyOp.class, imgF, psf, new long[] { pad, pad, pad }, iterations);

			ij.ui().show("cuda op deconvolved", outputCuda);
		}

		// run MKL Richardson Lucy op

		if (mklRL) {
			RandomAccessibleInterval<FloatType> deconvolvedMKL = (RandomAccessibleInterval<FloatType>) ij.op().run(
					MKLRichardsonLucyOp.class, imgF, psf, new long[] { 30, 30, 20 }, null, null, null, iterations,
					true);

			ij.ui().show("mkl op deconvolved", deconvolvedMKL);
		}

	}

}
