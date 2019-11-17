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

public class DeconvolveBenchmark {

    private Context context;

    @Test
    public void benchmarkCLIJOPS(){
        ImagePlus input = NewImage.createFloatImage("test", 100, 100, 100, NewImage.FILL_BLACK);
        input.setZ(50);
        ImageProcessor ip = input.getProcessor();
        ip.setf(10, 10, 100);
        ip.setf(40, 10, 100);
        ip.setf(10, 40, 100);

        ImagePlus psf = IJ.openImage("src/main/resources/PSF.tif");

        stamp("PSF size is "+psf.getWidth()+" "+psf.getHeight()+" "+psf.getNSlices());
        
        int numIterations = 16;

        context = new Context(OpService.class);

        runConvolveDeconvolveOps(input, psf, numIterations);
        runConvolveDeconvolveOps(input, psf, numIterations);
        runConvolveDeconvolveOps(input, psf, numIterations);

        runConvolveDeconvolveCLIJ(input, psf, numIterations);
        runConvolveDeconvolveCLIJ(input, psf, numIterations);
        runConvolveDeconvolveCLIJ(input, psf, numIterations);


    }

    private void runConvolveDeconvolveCLIJ(ImagePlus input, ImagePlus psf, int numIterations) {
        // init CLIJ
        stamp("");
        CLIJ clij = CLIJ.getInstance();
        stamp("init clij");

        ClearCLBuffer inputCL = clij.convert(input, ClearCLBuffer.class);
        ClearCLBuffer psfCL = clij.convert(psf, ClearCLBuffer.class);
        ClearCLBuffer convolvedCL = clij.createCLBuffer(inputCL);
        ClearCLBuffer deconvolvedCL = clij.createCLBuffer(inputCL);
        stamp("allocate and convert with clij");

        Convolve.convolveWithCustomKernel(clij, inputCL, psfCL, convolvedCL);
        stamp("convolve with clij");

        Deconvolve.deconvolveWithCustomKernel(clij, convolvedCL, psfCL, deconvolvedCL, numIterations);
        stamp("deconvolve with clij");
    }

    private void runConvolveDeconvolveOps(ImagePlus input, ImagePlus psf, int numIterations) {
        // init ops
        stamp("");
        OpService ops = context.getService(OpService.class);
        stamp("init ops");

        RandomAccessibleInterval<FloatType> inputRai = ImageJFunctions.convertFloat(input);
        RandomAccessibleInterval<FloatType> psfRai = ImageJFunctions.convertFloat(psf);
        stamp("allocate and convert with ops");

        RandomAccessibleInterval<FloatType> convolvedRai = ops.filter().convolve(inputRai, psfRai);
        stamp("convolve with ops");

        RandomAccessibleInterval<FloatType> deconvolvedRai = ops.deconvolve().richardsonLucy(convolvedRai, psfRai, numIterations);
        stamp("deconvolve with ops");
    }

    private long timeStamp;
    private void stamp(String text) {
        if (text.length() > 0) {
            System.out.println(text + " took " + (System.currentTimeMillis() - timeStamp) + " msec");
        }

        timeStamp = System.currentTimeMillis();
    }
}