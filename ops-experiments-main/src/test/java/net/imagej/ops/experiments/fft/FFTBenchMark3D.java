package net.imagej.ops.experiments.fft;

//import static org.bytedeco.javacpp.cufftw;
//import static org.bytedeco.javacpp.cufftw.FFTW_ESTIMATE;
//import static org.bytedeco.javacpp.cufftw.fftwf_destroy_plan;
//import static org.bytedeco.javacpp.cufftw.fftwf_execute_dft_r2c;
//import static org.bytedeco.javacpp.cufftw.fftwf_plan_dft_r2c_2d;

import static org.bytedeco.javacpp.fftw3.FFTW_ESTIMATE;
import static org.bytedeco.javacpp.fftw3.fftwf_destroy_plan;
import static org.bytedeco.javacpp.fftw3.fftwf_execute;
import static org.bytedeco.javacpp.fftw3.fftwf_init_threads;
import static org.bytedeco.javacpp.fftw3.fftwf_plan_dft_r2c_3d;
import static org.bytedeco.javacpp.fftw3.fftwf_plan_with_nthreads;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import net.imagej.ImageJ;
import net.imagej.ops.filter.fft.CreateOutputFFTMethods;
import net.imagej.ops.filter.fft.FFTMethodsOpC;
import net.imagej.ops.filter.pad.PadInputFFTMethods;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imagej.ops.special.function.BinaryFunctionOp;
import net.imagej.ops.special.function.Functions;
import net.imagej.ops.special.function.UnaryFunctionOp;
import net.imglib2.Cursor;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.FloatAccess;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.complex.ComplexFloatType;
import net.imglib2.type.numeric.real.FloatType;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.fftw3;
import org.bytedeco.javacpp.fftw3.fftwf_plan;
import org.jtransforms.utils.CommonUtils;

import edu.emory.mathcs.jtransforms.fft.FloatFFT_3D;
import pl.edu.icm.jlargearrays.ConcurrencyUtils;

public class FFTBenchMark3D {

	final static ImageJ ij = new ImageJ();

	final static Random random = new Random();

	public static <T extends RealType<T> & NativeType<T>> void main(final String[] args) throws IOException {

		ij.launch(args);

		Loader.load(fftw3.class);

		final int numCores = Runtime.getRuntime().availableProcessors();

		final long testSize = 256;

		final FinalDimensions dims = new FinalDimensions(testSize, testSize, testSize);

		final ArrayImg<FloatType, FloatAccess> image = getRandomImage(dims, new FloatType());

		final FloatArray f = (FloatArray) image.update(null);
		final float[] aImage = f.getCurrentStorageArray();

		// make a copy of the array to use for jtransform (jtransform is inplace
		// so it will change the data)
		final float[] aImage2 = new float[(int) (dims.dimension(0) * dims.dimension(1) * dims.dimension(2))];

		int i = 0;
		for (float val : aImage) {
			aImage2[i++] = val;
		}

		final FloatFFT_3D jfft = new FloatFFT_3D((int) dims.dimension(0), (int) dims.dimension(1),
				(int) dims.dimension(2));

		int nthread = numCores;
		int threadsBegin2D = 65536;
		int threadsBegin3D = 65536;

		ConcurrencyUtils.setNumberOfThreads(nthread);
		CommonUtils.setThreadsBeginN_2D(threadsBegin2D);
		CommonUtils.setThreadsBeginN_3D(threadsBegin3D);

		// jtransform warm up
		// jfft.realForward(aImage);

		long startTime = System.nanoTime();

		jfft.realForward(aImage2);

		System.out.println("JTransform execution time: " + ((float) (System.nanoTime() - startTime)) / 1000000);

		final long[] fftSize = new long[] { dims.dimension(0) / 2, dims.dimension(1), dims.dimension(2) };

		Img<ComplexFloatType> jfftImage = ArrayImgs.complexFloats(aImage2, fftSize);

		ImageJFunctions.show(jfftImage).setTitle("jtransform power spectrum");
		;

		// Start FFTW Benchmark
		try {

			// convert to FloatPointer
			final FloatPointer p = new FloatPointer(dims.dimension(0) * dims.dimension(1) * dims.dimension(2));

			final FloatPointer pout = new FloatPointer(
					2 * (dims.dimension(0) / 2 + 1) * dims.dimension(1) * dims.dimension(2));

			fftwf_init_threads();

			fftwf_plan_with_nthreads(numCores);

			// create FFT plan
			final fftwf_plan plan = fftwf_plan_dft_r2c_3d((int) dims.dimension(0), (int) dims.dimension(1),
					(int) dims.dimension(2), p, pout, (int) FFTW_ESTIMATE);

			// fftw warm up

			p.put(aImage);
			// fftwf_execute(plan);

			startTime = System.nanoTime();

			fftwf_execute(plan);

			System.out.println("FFTW execution time: " + ((float) (System.nanoTime() - startTime)) / 1000000);

			fftwf_destroy_plan(plan);

			float[] aOut = new float[(int) (2 * (dims.dimension(0) / 2 + 1) * dims.dimension(1) * dims.dimension(2))];
			pout.get(aOut);

			FloatPointer.free(p);

			Img<ComplexFloatType> fftwImage = ArrayImgs.complexFloats(aOut, fftSize);

			ImageJFunctions.show(fftwImage).setTitle("fftw power spectrum");
		} catch (Exception e) {
			System.out.println("FFTW bench mark failed with: " + e);
		}

		// end FFTW benchmark
		/*
		 * // Start CUFFT benchmark // convert to device (cuda) FloatPointer
		 * 
		 * // output size of FFT see //
		 * http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data
		 * .html final long[] fftSize = new long[] { dims.dimension(0) / 2 + 1,
		 * dims.dimension(1) };
		 * 
		 * final FloatPointer poutDevice = new FloatPointer();
		 * 
		 * // malloc memory on device checkCudaErrors(cudaMalloc(poutDevice, 2 *
		 * fftSize[0] * fftSize[1] * Float.BYTES));
		 * 
		 * // float pointer for device FloatPointer pDevice = new
		 * FloatPointer();
		 * 
		 * long size = dims.dimension(0) * dims.dimension(1);
		 * 
		 * CudaUtility.checkCudaErrors(cudaMalloc(pDevice, size * Float.BYTES));
		 * 
		 * // create FFT plan final org.bytedeco.javacpp.cufftw.fftwf_plan
		 * plancufft = org.bytedeco.javacpp.cufftw.fftwf_plan_dft_r2c_2d( (int)
		 * dims.dimension(0), (int) dims.dimension(1), pDevice, poutDevice,
		 * (int) org.bytedeco.javacpp.cufftw.FFTW_ESTIMATE);
		 * 
		 * // warm up for (final FloatPointer pHost : floatPointers) {
		 * 
		 * CudaUtility.checkCudaErrors(cudaMemcpy(pDevice, pHost, size *
		 * Float.BYTES, cudaMemcpyHostToDevice));
		 * 
		 * org.bytedeco.javacpp.cufftw.fftwf_execute_dft_r2c(plancufft, pDevice,
		 * poutDevice); }
		 * 
		 * startTime = System.nanoTime();
		 * 
		 * for (final FloatPointer pHost : floatPointers) {
		 * 
		 * CudaUtility.checkCudaErrors(cudaMemcpy(pDevice, pHost, size *
		 * Float.BYTES, cudaMemcpyHostToDevice));
		 * 
		 * org.bytedeco.javacpp.cufftw.fftwf_execute_dft_r2c(plancufft, pDevice,
		 * poutDevice); }
		 * 
		 * System.out.println("CUFFTW execution time: " + ((float)
		 * (System.nanoTime() - startTime)) / 1000000);
		 * 
		 * org.bytedeco.javacpp.cufftw.fftwf_destroy_plan(plancufft);
		 * 
		 * // End CUFFT benchmark
		 * 
		 * benchMarkOpsFFT(images);
		 */
	}

	private static <T extends RealType<T> & NativeType<T>, A> ArrayImg<T, A> getRandomImage(final Dimensions d,
			final T t) {
		final ArrayImgFactory<T> fac = new ArrayImgFactory<T>();

		final ArrayImg<T, A> out = (ArrayImg<T, A>) ij.op().create().img(d, t, fac);

		final Cursor<T> c = out.cursor();

		while (c.hasNext()) {
			c.fwd();
			c.get().setReal(random.nextFloat());
		}

		return out;
	}

	private static <T extends RealType<T> & NativeType<T>, A> void benchMarkOpsFFT(
			final ArrayList<ArrayImg<T, A>> images) {

		final Type fftType = new ComplexFloatType();

		final BinaryFunctionOp<RandomAccessibleInterval<T>, Dimensions, RandomAccessibleInterval<T>> padOp = (BinaryFunctionOp) Functions
				.binary(ij.op(), PadInputFFTMethods.class, RandomAccessibleInterval.class,
						RandomAccessibleInterval.class, Dimensions.class, true);

		final UnaryFunctionOp<Dimensions, RandomAccessibleInterval<ComplexFloatType>> createOp = (UnaryFunctionOp) Functions
				.unary(ij.op(), CreateOutputFFTMethods.class, RandomAccessibleInterval.class, Dimensions.class, fftType,
						true);

		final UnaryComputerOp<RandomAccessibleInterval<T>, RandomAccessibleInterval<ComplexFloatType>> fft = (UnaryComputerOp) Computers
				.unary(ij.op(), FFTMethodsOpC.class, images.get(0), RandomAccessibleInterval.class);

		// create the complex output
		final RandomAccessibleInterval<ComplexFloatType> output = createOp.calculate(images.get(0));

		final long startTime = System.nanoTime();

		for (final Img image : images) {

			fft.compute(padOp.calculate(image, image), output);
		}

		System.out.println(
				"FFT Methods Computer Op execution time: " + ((float) (System.nanoTime() - startTime)) / 1000000);
	}

}
