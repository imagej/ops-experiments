
package net.imagej.ops.experiments.filter.deconvolve;

import java.io.IOException;

import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.customconvolutionplugin.Deconvolve;
import net.imagej.ImageJ;
import net.imagej.ops.experiments.testImages.Bars;
import net.imagej.ops.experiments.testImages.DeconvolutionTestData;
import net.imagej.ops.special.computer.Computers;
import net.imagej.ops.special.computer.UnaryComputerOp;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class InteractiveDeconvolveTest<T extends RealType<T> & NativeType<T>> {

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(
		final String[] args) throws IOException
	{

		final String libPathProperty = System.getProperty("java.library.path");
		System.out.println("Lib path:" + libPathProperty);

		ij.launch(args);
		
		stamp("Starting tests");

		DeconvolutionTestData testData = new Bars("../images/");
		// DeconvolutionTestData testData = new CElegans();
		// DeconvolutionTestData testData = new HalfBead();

		testData.LoadImages(ij);
		RandomAccessibleInterval<FloatType> imgF = testData.getImg();
		RandomAccessibleInterval<FloatType> psfF = testData.getPSF();

		ij.ui().show("bars ", imgF);
		ij.ui().show("psf ", psfF);

		final int iterations = 100;
		final int pad = 20;
		
		RandomAccessibleInterval<FloatType> deconvolvedMKL = deconvolveMKL(imgF, psfF, iterations);
		stamp("Deconvolve with MKL ");
				
		RandomAccessibleInterval<FloatType>deconvolvedOps = deconvolveOps(imgF, psfF, iterations);
		stamp("Deconvolve with Ops ");
		
		RandomAccessibleInterval<FloatType> deconvolvedCuda = deconvolveCuda(imgF, psfF, iterations);
		stamp("Deconvolve with Cuda ");
	
		//RandomAccessibleInterval<FloatType> deconvolvedCLIJ = deconvolveCLIJ(imgF, psfF, iterations);
		//stamp("Deconvolve with CLIJ ");
		
		ij.ui().show("Deconvolved Ops", deconvolvedOps);
		ij.ui().show("Deconvolved Cuda", deconvolvedCuda);
		ij.ui().show("Deconvolved MKL", deconvolvedMKL);
		//ij.ui().show("Deconvolved CLIJ", deconvolvedCLIJ);
		

	}
	
	private static RandomAccessibleInterval<FloatType> deconvolveOps(RandomAccessibleInterval<FloatType> imgF, RandomAccessibleInterval<FloatType> psfF, int iterations) {

		return (Img<FloatType>) ij.op().deconvolve()
			.richardsonLucy(imgF, psfF, null, null, null,
				null, null, iterations, false, false);

	}
	
	private static RandomAccessibleInterval<FloatType> deconvolveCuda(RandomAccessibleInterval<FloatType> imgF, RandomAccessibleInterval<FloatType> psfF, int iterations) {
		@SuppressWarnings("unchecked")
		final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver =
			(UnaryComputerOp) Computers.unary(ij.op(), UnaryComputerYacuDecu.class,
				RandomAccessibleInterval.class, imgF, psfF, iterations);


		RandomAccessibleInterval<FloatType> outputCuda = ij.op().create().img(
			imgF);

		deconvolver.compute(imgF, outputCuda);
		
		return outputCuda;

	}
	
	private static RandomAccessibleInterval<FloatType> deconvolveCLIJ(RandomAccessibleInterval<FloatType> imgF, RandomAccessibleInterval<FloatType> psfF, int iterations) {
    // init CLIJ
    stamp("");
    CLIJ clij = CLIJ.getInstance();
    stamp("init clij");

    ClearCLBuffer inputCL = clij.convert(imgF, ClearCLBuffer.class);
    ClearCLBuffer psfCL = clij.convert(psfF, ClearCLBuffer.class);
    ClearCLBuffer deconvolvedCL = clij.createCLBuffer(inputCL);
    stamp("allocate and convert with clij");

    Deconvolve.deconvolveWithCustomKernel(clij, inputCL, psfCL, deconvolvedCL, iterations);
    
    stamp("deconvolve with clij");

    return (RandomAccessibleInterval)clij.pullRAI(deconvolvedCL);

	
	}
	
	private static RandomAccessibleInterval<FloatType> deconvolveMKL(RandomAccessibleInterval<FloatType> imgF, RandomAccessibleInterval<FloatType> psfF, int iterations) {

		@SuppressWarnings("unchecked")
		final UnaryComputerOp<RandomAccessibleInterval<FloatType>, RandomAccessibleInterval<FloatType>> deconvolver =
			(UnaryComputerOp) Computers.unary(ij.op(), UnaryComputerMKLDecon.class,
				RandomAccessibleInterval.class, imgF, psfF, iterations);

		Img<FloatType> deconvolved = ij.op().create().img(imgF);

		deconvolver.compute(imgF, deconvolved);
		
		return deconvolved;

	}
	
	
	// stamp function originally from 
  private static long timeStamp;
  private static void stamp(String text) {
      if (text.length() > 0) {
          System.out.println(text + " took " + (System.currentTimeMillis() - timeStamp) + " msec");
      }

      timeStamp = System.currentTimeMillis();
  }

}
