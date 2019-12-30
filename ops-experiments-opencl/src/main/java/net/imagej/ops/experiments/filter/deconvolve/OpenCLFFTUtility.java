
package net.imagej.ops.experiments.filter.deconvolve;

import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.imagej.ops.OpService;
import net.imagej.ops.filter.pad.DefaultPadInputFFT;
import net.imglib2.FinalDimensions;
import net.imglib2.Interval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory;
import net.imglib2.outofbounds.OutOfBoundsMirrorFactory.Boundary;
import net.imglib2.type.numeric.complex.ComplexFloatType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Intervals;
import net.imglib2.view.Views;

import org.jocl.NativePointerObject;

public class OpenCLFFTUtility {

	public static RandomAccessibleInterval<ComplexFloatType> runFFT(
		RandomAccessibleInterval<FloatType> img, boolean reflectAndCenter,
		OpService ops)
	{
		// extend the image to a smooth number as clFFT does not support
		// all FFT sizes
		img = (RandomAccessibleInterval<FloatType>) ops.run(
			DefaultPadInputFFT.class, img, img, false);

		// get CLIJ and push to GPU
		CLIJ clij = CLIJ.getInstance();
		ClearCLBuffer gpuInput = clij.push(img);

		// run FFT
		ClearCLBuffer fft = runFFT(gpuInput);

		// pull result from GPU
		RandomAccessibleInterval<FloatType> result =
			(RandomAccessibleInterval<FloatType>) clij.pullRAI(fft);

		// convert to Complex
		// TODO: do this without a copy (CLIJ needs complex types?)
		RandomAccessibleInterval<ComplexFloatType> resultComplex = copyAsComplex(
			result);

		if (reflectAndCenter) {
			// compute the interval of a full sized centered FFT
			Interval interval = Intervals.createMinMax(-img.dimension(0) / 2, -img
				.dimension(1) / 2, img.dimension(0) / 2, img.dimension(1) / 2);

			// reflect and center
			resultComplex = (RandomAccessibleInterval<ComplexFloatType>) Views
				.interval(Views.extend(resultComplex,
					new OutOfBoundsMirrorFactory<ComplexFloatType, RandomAccessibleInterval<ComplexFloatType>>(
						Boundary.SINGLE)), interval);
		}

		return resultComplex;
	}

	public static ClearCLBuffer runFFT(ClearCLBuffer gpuImg) {
		CLIJ clij = CLIJ.getInstance();

		// compute complex FFT dimension assuming Hermitian interleaved
		long[] fftDim = new long[] { (gpuImg.getWidth() / 2 + 1) * 2, gpuImg
			.getHeight() };

		// create GPU memory for FFT
		ClearCLBuffer gpuFFT = clij.create(fftDim, NativeTypeEnum.Float);

		// use a hack to get the long pointers to in, out, context and queue.
		long l_in = hackPointer((NativePointerObject) (gpuImg.getPeerPointer()
			.getPointer()));
		long l_out = hackPointer((NativePointerObject) (gpuFFT.getPeerPointer()
			.getPointer()));
		long l_context = hackPointer((NativePointerObject) (clij.getClearCLContext()
			.getPeerPointer().getPointer()));
		long l_queue = hackPointer((NativePointerObject) (clij.getClearCLContext()
			.getDefaultQueue().getPeerPointer().getPointer()));

		// call the native code that runs the FFT
		OpenCLWrapper.fft2d_long((long) (gpuImg.getWidth()), gpuImg.getHeight(),
			l_in, l_out, l_context, l_queue);

		return gpuFFT;
	}

	public static ClearCLBuffer runDecon(ClearCLBuffer gpuImg,
		RandomAccessibleInterval<FloatType> psf, OpService ops)
	{

		// extend and shift the PSF
		RandomAccessibleInterval<FloatType> extendedPSF = Views.zeroMin(ops.filter()
			.padShiftFFTKernel(psf, new FinalDimensions(gpuImg.getDimensions())));

		// get CLIJ
		CLIJ clij = CLIJ.getInstance();

		long start = System.currentTimeMillis();

		// transfer PSF to the GPU
		ClearCLBuffer gpuPSF = clij.push(extendedPSF);

		// create another copy of the image to use as the initial value
		ClearCLBuffer gpuEstimate = clij.create(gpuImg);
		clij.op().copy(gpuImg, gpuEstimate);

		// Use a hack to get long pointers to the CL Buffers, context, queue and
		// device
		// (TODO: Use a more sensible approach once Robert H's pull request is
		// released)
		long longPointerImg = hackPointer((NativePointerObject) (gpuImg
			.getPeerPointer().getPointer()));
		long longPointerPSF = hackPointer((NativePointerObject) (gpuPSF
			.getPeerPointer().getPointer()));
		long longPointerEstimate = hackPointer((NativePointerObject) (gpuEstimate
			.getPeerPointer().getPointer()));
		long l_context = hackPointer((NativePointerObject) (clij.getClearCLContext()
			.getPeerPointer().getPointer()));
		long l_queue = hackPointer((NativePointerObject) (clij.getClearCLContext()
			.getDefaultQueue().getPeerPointer().getPointer()));
		long l_device = hackPointer((NativePointerObject) clij.getClearCLContext()
			.getDevice().getPeerPointer().getPointer());

		// call the decon wrapper (100 iterations of RL)
		OpenCLWrapper.deconv_long(100, gpuImg.getDimensions()[0], gpuImg
			.getDimensions()[1], gpuImg.getDimensions()[2], longPointerImg,
			longPointerPSF, longPointerEstimate, longPointerImg, l_context, l_queue,
			l_device);

		long finish = System.currentTimeMillis();

		System.out.println("OpenCL Decon time " + (finish - start));

		return gpuEstimate;
	}

	static long hackPointer(NativePointerObject pointer) {

		String splitString = pointer.toString().split("\\[")[1];
		String hack = splitString.substring(0, splitString.length() - 1);

		return Long.decode(hack);
	}

	static Img<ComplexFloatType> copyAsComplex(
		RandomAccessibleInterval<FloatType> in)
	{
		float[] temp = new float[(int) (in.dimension(0) * in.dimension(1))];
		int i = 0;
		for (FloatType f : Views.iterable(in)) {
			temp[i++] = f.getRealFloat();
		}

		return ArrayImgs.complexFloats(temp, new long[] { in.dimension(0) / 2, in
			.dimension(1) });
	}
}
