
package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imagej.axis.Axes;
import net.imagej.axis.AxisType;
import net.imagej.ops.experiments.ImageUtility;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import ij.IJ;

public class BIGLabCElegansCudaDeconvolveTest<T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{

		// print user directory and library path for debugging purposes
		System.out.println("CWD: " + System.getProperty("user.dir"));
		final String libPathProperty = System.getProperty("java.library.path");
		System.out.println("Lib path:" + libPathProperty);

		ij.launch(args);
		ij.log().setLevel(2);

		final int iterations = 100;

		// load all three channels of the CElegans dataset from here 
		// http://bigwww.epfl.ch/deconvolution/index.html#data
		Img<T> img1 = (Img<T>) ((Dataset) ij.io().open(
			"../../../images/CElegans/CElegans-CY3.tif")).getImgPlus().getImg();
		Img<T> img2 = (Img<T>) ((Dataset) ij.io().open(
			"../../../images/CElegans/CElegans-DAPI.tif")).getImgPlus().getImg();
		Img<T> img3 = (Img<T>) ((Dataset) ij.io().open(
			"../../../images/CElegans/CElegans-FITC.tif")).getImgPlus().getImg();

		// load al three channels of the CElegans PSF
		Img<T> psf1 = (Img<T>) ((Dataset) ij.io().open(
			"../../../images/CElegans/PSF-CElegans-CY3.tif")).getImgPlus().getImg();
		Img<T> psf2 = (Img<T>) ((Dataset) ij.io().open(
			"../../../images/CElegans/PSF-CElegans-DAPI.tif")).getImgPlus().getImg();
		Img<T> psf3 = (Img<T>) ((Dataset) ij.io().open(
			"../../../images/CElegans/PSF-CElegans-FITC.tif")).getImgPlus().getImg();

		// combine Imgs and PSFs into a multi channel ImgPlus
		ImgPlus imgPlus = ImageUtility.createMultiChannelImgPlus(ij.dataset(), img1,
			img2, img3);
		ImgPlus imgPlusPSFs = ImageUtility.createMultiChannelImgPlus(ij.dataset(),
			psf1, psf2, psf3);

		ij.ui().show(imgPlus);
		IJ.run("Make Composite");

		ij.ui().show(imgPlusPSFs);
		IJ.run("Make Composite");

		// convert img and PSF to 32 bit
		Img<FloatType> imgF = ij.op().convert().float32((Img<T>) imgPlus.getImg());

		RandomAccessibleInterval<FloatType> psfF = ij.op().convert().float32(
			(Img<T>) imgPlusPSFs.getImg());

		long startTime, endTime;

		// run Cuda Richardson Lucy op

		// create Img for output
		RandomAccessibleInterval<FloatType> deconvolved = ij.op().create().img(
			imgF);

		// loop through all the channels
		for (int c = 0; c < 3; c++) {
			// crop PSF
			Img psfTemp = ImageUtility.cropSymmetric(Views.hyperSlice(psfF, 3, c),
				new long[] { 64, 64, 41 }, ij.op());

			// subtract min from PSF
		  RandomAccessibleInterval<FloatType> psfSubtract = Views.zeroMin(ImageUtility.subtractMin(psfTemp, ij.op()));

			// normalize PSF
		  RandomAccessibleInterval<FloatType> psfNormalized = Views.zeroMin(ImageUtility.normalize(psfTemp, ij.op()));

		  // create the deconvolution op 
			@SuppressWarnings("unchecked")
			final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver =
				(UnaryComputerOp) Computers.unary(ij.op(), UnaryComputerYacuDecu.class,
					RandomAccessibleInterval.class, RandomAccessibleInterval.class, psfNormalized,
					iterations);

			startTime = System.currentTimeMillis();
			
			System.out.println(Views.hyperSlice(imgF, 3, c).numDimensions());

			// deconvolve
			deconvolver.compute(Views.hyperSlice(imgF, 3, c), Views.hyperSlice(
				deconvolved, 3, c));

			endTime = System.currentTimeMillis();
			System.out.println("Total execution time cuda (decon+overhead) is: " +
				(endTime - startTime));

		}

		AxisType[] axisTypes = new AxisType[] { Axes.X, Axes.Y, Axes.Z,
			Axes.CHANNEL };

		ImgPlus imgPlusDeconvolved= new ImgPlus(ij.dataset().create(deconvolved), "deconvolved", axisTypes);

		ij.ui().show("cuda op deconvolved", imgPlusDeconvolved);
		IJ.run("Make Composite");

		// TODO Multi-channel projections
		/*
				// Render projections along X and Z axes
				ij.ui().show("Original (YZ)", VisualizationUtility.project(ij, imgF, 0));
				ij.ui().show("Deconvolved (YZ)", VisualizationUtility.project(ij,
					deconvolved, 0));
				ij.ui().show("Original (XY)", VisualizationUtility.project(ij, imgF, 2));
				ij.ui().show("Deconvolved (XY)", VisualizationUtility.project(ij,
					deconvolved, 2));
					*/
	}

}
