package net.haesleinhuepf.clij.customconvolutionplugin;

import java.io.IOException;

import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imagej.ops.experiments.testImages.Bars;
import net.imagej.ops.experiments.testImages.DeconvolutionTestData;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

import org.junit.Test;
import org.scijava.Context;

import ij.ImagePlus;

public class DeconvolveBarsBenchmark {

		final static ImageJ ij = new ImageJ();
    private Context context;


  	public static <T extends RealType<T> & NativeType<T>> void main(
  		final String[] args) throws IOException
  	{
    	
  		DeconvolutionTestData testData = new Bars("../images/");
  		// DeconvolutionTestData testData = new CElegans("../images/");
  		// DeconvolutionTestData testData = new Bead("../images/");

  		testData.LoadImages(ij);
  		RandomAccessibleInterval<FloatType> imgF = testData.getImg();
  		RandomAccessibleInterval<FloatType> psfF = testData.getPSF();
  		
  		ij.ui().show("img ", imgF);
  		ij.ui().show("psf ", psfF);
    
  		stamp("PSF size is "+psfF.dimension(0)+" "+psfF.dimension(1)+" "+psfF.dimension(2));
  		
  		long startTime = System.currentTimeMillis();

      int numIterations = 10;
  		final int pad = 20;

  		Img<FloatType> deconvolved = (Img<FloatType>) ij.op().deconvolve()
  			.richardsonLucy(imgF, psfF, new long[] { pad, pad, pad }, null, null,
  				null, null, numIterations, false, false);

  		long endTime = System.currentTimeMillis();
  		
  		stamp("Time ops ");
  		
  		RandomAccessibleInterval deconvolvedCLIJ=runConvolveDeconvolveCLIJ(imgF, psfF, numIterations);
  		
  		stamp("CLIJ");
  		
  		ij.ui().show("Deconvolved Ops", deconvolved);
  		ij.ui().show("Deconvolved CLIJ", deconvolvedCLIJ);
   	  
    }
  	
    private static RandomAccessibleInterval runConvolveDeconvolveCLIJ(RandomAccessibleInterval input, RandomAccessibleInterval psf, int numIterations) {
        // init CLIJ
        stamp("");
        CLIJ clij = CLIJ.getInstance();
        stamp("init clij");

        ClearCLBuffer inputCL = clij.convert(input, ClearCLBuffer.class);
        ClearCLBuffer psfCL = clij.convert(psf, ClearCLBuffer.class);
        ClearCLBuffer deconvolvedCL = clij.createCLBuffer(inputCL);
        stamp("allocate and convert with clij");

        Deconvolve.deconvolveWithCustomKernel(clij, inputCL, psfCL, deconvolvedCL, numIterations);
        
        stamp("deconvolve with clij");
    
        return clij.pullRAI(deconvolvedCL);
    }

    private static long timeStamp;
    private static void stamp(String text) {
        if (text.length() > 0) {
            System.out.println(text + " took " + (System.currentTimeMillis() - timeStamp) + " msec");
        }

        timeStamp = System.currentTimeMillis();
    }
}