package net.imagej.ops.experiments.filter.deconvolve;

import com.sun.jna.Pointer;

import java.io.IOException;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ops.Ops;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.complex.ComplexFloatType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

import org.jocl.NativePointerObject;
import org.jocl.cl_mem;

import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.clearcl.ClearCLContext;
import net.haesleinhuepf.clij.clearcl.ClearCLPeerPointer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;

public class InteractiveOpenCLFFTTest <T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{
		ij.launch(args);
		
		System.out.println(System.getProperty("java.library.path"));

		// load the dataset
		Dataset dataset = (Dataset) ij.io().open("../images/bridge-odd.tif");

		Img<FloatType> img = ij.op().convert().float32((Img)dataset.getImgPlus().getImg());
		
		// show the image
		ij.ui().show(img);
	
		RandomAccessibleInterval<ComplexFloatType> resultComplex = OpenCLFFTUtility.runFFT(img, true, ij.op());
		ImageJFunctions.show(resultComplex, "FFT OpenCL");
		
		RandomAccessibleInterval<ComplexFloatType> fftOps = ij.op().filter().fft(img);
		ImageJFunctions.show(fftOps, "FFT Ops");

	}

}
