
package net.haesleinhuepf.clij.customconvolutionplugin;

import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import edu.mines.jtk.dsp.Conv;
import ij.IJ;
import ij.ImagePlus;
import ij.gui.NewImage;
import ij.process.ImageProcessor;
import net.haesleinhuepf.clij.CLIJ;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.real.FloatType;
import org.junit.Test;
import org.scijava.Context;

import static org.junit.Assert.*;

public class DeconvolveBenchmarkGauss {

	private Context context;

	@Test
	public void benchmarkCLIJOPS() {
		ImagePlus input = NewImage.createFloatImage("test", 100, 100, 100,
			NewImage.FILL_BLACK);
		input.setZ(50);
		ImageProcessor ip = input.getProcessor();
		ip.setf(10, 10, 100);
		ip.setf(40, 10, 100);
		ip.setf(10, 40, 100);

		context = new Context(OpService.class);

		OpService ops = context.getService(OpService.class);

		int numIterations = 16;

		double[] sigmas = new double[] { 3.0, 5.0, 7.0, 9.0 };
		
		System.out.println(CLIJ.clinfo());

		for (double sigma : sigmas) {
			RandomAccessibleInterval psf = ops.create().kernelGauss(sigma, 3,
				new FloatType());
			System.out.println();
			System.out.println("PSF size is " + psf.dimension(0) + " " + psf
				.dimension(1) + " " + psf.dimension(2));

			runConvolveDeconvolveOps(input, psf, numIterations);
			runConvolveDeconvolveCLIJ(input, psf, numIterations);
			
		}

	}

	private void runConvolveDeconvolveCLIJ(ImagePlus input,
		RandomAccessibleInterval psf, int numIterations)
	{
		// init CLIJ
		CLIJ clij = CLIJ.getInstance();
		
		ClearCLBuffer inputCL = clij.convert(input, ClearCLBuffer.class);
		ClearCLBuffer psfCL = clij.convert(psf, ClearCLBuffer.class);
		ClearCLBuffer convolvedCL = clij.createCLBuffer(inputCL);
		ClearCLBuffer deconvolvedCL = clij.createCLBuffer(inputCL);
		
		Convolve.convolveWithCustomKernel(clij, inputCL, psfCL, convolvedCL);
		
		long start=System.currentTimeMillis();
		Deconvolve.deconvolveWithCustomKernel(clij, convolvedCL, psfCL,
			deconvolvedCL, numIterations);
		long end = System.currentTimeMillis();
		System.out.println("deconvolve with clij "+(end-start));
	}

	private void runConvolveDeconvolveOps(ImagePlus input,
		RandomAccessibleInterval psfRai, int numIterations)
	{
		// init ops
		OpService ops = context.getService(OpService.class);

		RandomAccessibleInterval<FloatType> inputRai = ImageJFunctions.convertFloat(
			input);

		RandomAccessibleInterval<FloatType> convolvedRai = ops.filter().convolve(
			inputRai, psfRai);

		long start = System.currentTimeMillis();
		RandomAccessibleInterval<FloatType> deconvolvedRai = ops.deconvolve()
			.richardsonLucy(convolvedRai, psfRai, numIterations);
		long end = System.currentTimeMillis();
		System.out.println("deconvolve with ops "+(end-start));
	}

	private long timeStamp;

	private void stamp(String text) {
		if (text.length() > 0) {
			System.out.println(text + " took " + (System.currentTimeMillis() -
				timeStamp) + " msec");
		}

		timeStamp = System.currentTimeMillis();
	}
}
