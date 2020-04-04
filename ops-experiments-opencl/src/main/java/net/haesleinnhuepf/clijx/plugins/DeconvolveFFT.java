package net.haesleinnhuepf.clijx.plugins;


import ij.IJ;
import ij.ImagePlus;
import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import net.haesleinhuepf.clij2.AbstractCLIJ2Plugin;
import net.haesleinhuepf.clij2.utilities.HasAuthor;
import net.imagej.ops.experiments.filter.deconvolve.OpenCLFFTUtility;
import org.scijava.plugin.Plugin;

@Plugin(type = CLIJMacroPlugin.class, name = "CLIJx_deconvolveFFT")
public class DeconvolveFFT extends AbstractCLIJ2Plugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation, HasAuthor {

    @Override
    public boolean executeCL() {
        Object[] args = openCLBufferArgs();
        boolean result = deconvolveFFT(clij, (ClearCLBuffer)( args[0]), (ClearCLBuffer)(args[1]), (ClearCLBuffer)(args[2]));
        releaseBuffers(args);
        return result;
    }

    public static boolean deconvolveFFT(CLIJ clij, ClearCLBuffer input, ClearCLBuffer convolution_kernel, ClearCLBuffer destination) {

        OpenCLFFTUtility.runDecon(clij, input, convolution_kernel, destination);

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

        CLIJ clij = CLIJ.getInstance("RTX");

        ClearCLBuffer inputGPU = clij.push(input);
        ClearCLBuffer psfGPU = clij.push(psf);

        ClearCLBuffer output = clij.create(inputGPU);

        DeconvolveFFT.deconvolveFFT(clij, inputGPU, psfGPU, output);

        clij.show(output, "output");

    }
}
