
package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.imagej.ImageJ;
import net.imagej.ops.experiments.testImages.Bars;
import net.imagej.ops.experiments.testImages.DeconvolutionTestData;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.view.Views;

public class InteractiveOpenCLDeconvolveTest<T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{
		// check the library path, can be useful for debugging
		System.out.println(System.getProperty("java.library.path"));

		ij.launch(args);

		// get the test data
		DeconvolutionTestData testData = new Bars("../images/");

		testData.LoadImages(ij);
		RandomAccessibleInterval<FloatType> imgF = testData.getImg();
		RandomAccessibleInterval<FloatType> psfF = testData.getPSF();

		// take a look at it
		ij.ui().show("img ", imgF);
		ij.ui().show("psf ", psfF);

		// extend and shift the PSF
		RandomAccessibleInterval<FloatType> extendedPSF = Views.zeroMin(ij.op()
			.filter().padShiftFFTKernel(psfF, imgF));

		ij.ui().show("padded shifted PSF ", extendedPSF);

		// get CLIJ
		CLIJ clij = CLIJ.getInstance();

		long start = System.currentTimeMillis();

		// transfer image and PSF to the GPU
		ClearCLBuffer gpuImg = clij.push(imgF);

		ClearCLBuffer gpuEstimate = OpenCLFFTUtility.runDecon(gpuImg, psfF, ij
			.op());

		/*
		ClearCLBuffer gpuPSF= clij.push(extendedPSF);
		
		// transfer another copy of the image to the GPU to use as the initial value of the estimate
		ClearCLBuffer gpuEstimate = clij.push(imgF);
		
		// Use a hack to get long pointers to the CL Buffers, context, queue and device
		// (TODO: Use a more sensible approach once Robert H's pull request is released)
		long longPointerImg=OpenCLDeconvolveUtility.hackPointer((NativePointerObject)(gpuImg.getPeerPointer().getPointer()));
		long longPointerPSF=OpenCLDeconvolveUtility.hackPointer((NativePointerObject)(gpuPSF.getPeerPointer().getPointer()));
		long longPointerEstimate=OpenCLDeconvolveUtility.hackPointer((NativePointerObject)(gpuEstimate.getPeerPointer().getPointer()));
		long l_context= OpenCLDeconvolveUtility.hackPointer((NativePointerObject)(clij.getClearCLContext().getPeerPointer().getPointer()));
		long l_queue= OpenCLDeconvolveUtility.hackPointer((NativePointerObject)(clij.getClearCLContext().getDefaultQueue().getPeerPointer().getPointer()));
		long l_device = OpenCLDeconvolveUtility.hackPointer((NativePointerObject)clij.getClearCLContext().getDevice().getPeerPointer().getPointer());
		
		// call the decon wrapper, the estimate will be updated with 100 iterations of RL 
		OpenCLWrapper.deconv_long(100, imgF.dimension(0), imgF.dimension(1), imgF.dimension(2), longPointerImg, longPointerPSF, longPointerEstimate, longPointerImg, l_context, l_queue, l_device); 
		
		long finish= System.currentTimeMillis();
		
		System.out.println("OpenCL Decon time "+(finish-start));
		 */

		// show the result
		clij.show(gpuEstimate, "GPU Decon Result");

	}
}
