package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ops.Ops;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

import org.jocl.NativePointerObject;
import org.jocl.cl_mem;

import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.clearcl.ClearCLPeerPointer;

public class InteractiveOpenCLTests <T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{
		ij.launch(args);
		
		System.out.println(System.getProperty("java.library.path"));
				// load the dataset
		Dataset dataset = (Dataset) ij.io().open("../images/bridge.tif");

		Img<FloatType> ok = ij.op().convert().float32((Img)dataset.getImgPlus().getImg());
		// show the image
		ij.ui().show(ok);


		CLIJ clij = CLIJ.getInstance();

		ClearCLBuffer gpuInput = clij.push(ok);
		ClearCLBuffer gpuFFT = clij.create(gpuInput);
		
		cl_mem test2;

		ClearCLPeerPointer test=gpuInput.getPeerPointer();
		
		System.out.println(test.getClass());
		System.out.println("test 2 is ");
		NativePointerObject inPointer=(NativePointerObject)(gpuInput.getPeerPointer().getPointer());
		NativePointerObject outPointer=(NativePointerObject)(gpuFFT.getPeerPointer().getPointer());
	
		String hack=inPointer.toString().substring(7, 21);
		
		long crazy=Long.decode(hack);
		System.out.println(hack);
		System.out.println(crazy);
		
		hack=outPointer.toString().substring(7, 21);
		long reallyCrazy=Long.decode(hack);
		
	
		System.out.println(inPointer);
		System.out.println(inPointer.getClass());
		
		System.out.println(inPointer.toString());
	
		OpenCLWrapper.fft2d_long((long)(gpuInput.getWidth()),gpuInput.getHeight(), crazy, reallyCrazy);
	//	test.mPointer.nativePointer;
		
		clij.show(gpuFFT, "GPU FFT");

		gpuInput.close();
		gpuFFT.close();


	}


}
