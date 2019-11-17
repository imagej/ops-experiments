package net.haesleinhuepf.clij.customconvolutionplugin;

import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.macro.AbstractCLIJPlugin;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import org.scijava.plugin.Plugin;

import java.util.HashMap;

/**
 *
 *
 * Author: @haesleinhuepf
 * 12 2018
 */
@Plugin(type = CLIJMacroPlugin.class, name = "CLIJ_convolve")
public class Convolve extends AbstractCLIJPlugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation {

    @Override
    public boolean executeCL() {
        Object[] args = openCLBufferArgs();
        boolean result = convolveWithCustomKernel(clij, (ClearCLBuffer)( args[0]), (ClearCLBuffer)(args[1]), (ClearCLBuffer)(args[2]));
        releaseBuffers(args);
        return result;
    }

    static boolean convolveWithCustomKernel(CLIJ clij, ClearCLBuffer src, ClearCLBuffer kernel, ClearCLBuffer dst) {
        HashMap<String, Object> parameters = new HashMap<>();
        parameters.put("src", src);
        parameters.put("kernelImage", kernel);
        parameters.put("dst", dst);

        return clij.execute(Convolve.class,
                "customConvolution.cl",
                "custom_convolution_" + src.getDimension() + "d",
                parameters);
    }

    @Override
    public String getParameterHelpText() {
        return "Image source, Image convolution_kernel, Image destination";
    }

    @Override
    public String getDescription() {
        return "Convolve the image with a given kernel image. Kernel image and source image should have the same\n" +
                "bit-type. Furthermore, it is recommended that the kernel image has an odd size in X, Y and Z.";
    }

    @Override
    public String getAvailableForDimensions() {
        return "2D, 3D";
    }

    @Override
    public ClearCLBuffer createOutputBufferFromSource(ClearCLBuffer input) {
        return super.createOutputBufferFromSource((ClearCLBuffer)args[0]);
    }

}