package net.haesleinhuepf.clijx.plugins;


import ij.IJ;
import ij.ImagePlus;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import net.haesleinhuepf.clij2.AbstractCLIJ2Plugin;
import net.haesleinhuepf.clij2.CLIJ2;
import net.haesleinhuepf.clij2.utilities.HasAuthor;
import net.imagej.ops.experiments.filter.deconvolve.OpenCLFFTUtility;
import org.scijava.plugin.Plugin;

@Plugin(type = CLIJMacroPlugin.class, name = "CLIJx_deconvolveFFT")
public class DeconvolveFFT extends AbstractCLIJ2Plugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation, HasAuthor {

    @Override
    public boolean executeCL() {
        Object[] args = openCLBufferArgs();
        boolean result = deconvolveFFT(getCLIJ2(), (ClearCLBuffer)( args[0]), (ClearCLBuffer)(args[1]), (ClearCLBuffer)(args[2]));
        releaseBuffers(args);
        return result;
    }

    public static boolean deconvolveFFT(CLIJ2 clij2, ClearCLBuffer input, ClearCLBuffer convolution_kernel, ClearCLBuffer destination) {

        OpenCLFFTUtility.runDecon(clij2.getCLIJ(), input, convolution_kernel, destination);

        return true;
    }

    @Override
    public String getParameterHelpText() {
        return "Image input, Image convolution_kernel, Image destination";
    }

    @Override
    public String getDescription() {
        return "Applies Richardson-Lucy deconvolution using a Fast Fourier Transform using the clFFT library.";
    }

    @Override
    public String getAvailableForDimensions() {
        return "2D, 3D";
    }

    @Override
    public String getAuthorName() {
        return "Brian Northon, Robert Haase";
    }

    public static void main(String[] args) {

        ImagePlus input = IJ.openImage("C:/Users/rober/Downloads/images/Bars-G10-P15-stack-cropped.tif");
        ImagePlus psf = IJ.openImage("C:/Users/rober/Downloads/images/PSF-Bars-stack-cropped-64.tif");

        IJ.run(input, "32-bit", "");
        IJ.run(psf, "32-bit", "");

        CLIJ2 clij2 = CLIJ2.getInstance("RTX");

        ClearCLBuffer inputGPU = clij2.push(input);
        ClearCLBuffer psfGPU = clij2.push(psf);

        ClearCLBuffer output = clij2.create(inputGPU);

        DeconvolveFFT.deconvolveFFT(clij2, inputGPU, psfGPU, output);

        clij2.show(output, "output");

    }
}
