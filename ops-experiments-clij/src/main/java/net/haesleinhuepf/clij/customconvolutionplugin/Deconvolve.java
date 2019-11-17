package net.haesleinhuepf.clij.customconvolutionplugin;

import ij.IJ;
import ij.ImagePlus;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.kernels.Kernels;
import net.haesleinhuepf.clij.macro.AbstractCLIJPlugin;
import net.haesleinhuepf.clij.macro.CLIJMacroPlugin;
import net.haesleinhuepf.clij.macro.CLIJOpenCLProcessor;
import net.haesleinhuepf.clij.macro.documentation.OffersDocumentation;
import org.scijava.plugin.Plugin;

import static net.haesleinhuepf.clij.customconvolutionplugin.Convolve.convolveWithCustomKernel;

/**
 *
 *
 * Author: @haesleinhuepf
 * 12 2018
 */
@Plugin(type = CLIJMacroPlugin.class, name = "CLIJ_deconvolve")
public class Deconvolve extends AbstractCLIJPlugin implements CLIJMacroPlugin, CLIJOpenCLProcessor, OffersDocumentation {

    @Override
    public boolean executeCL() {
        Object[] args = openCLBufferArgs();
        boolean result = deconvolveWithCustomKernel(clij, (ClearCLBuffer)( args[0]), (ClearCLBuffer)(args[1]), (ClearCLBuffer)(args[2]), asInteger(args[3]));
        releaseBuffers(args);
        return result;
    }

    static public boolean deconvolveWithCustomKernel(CLIJ clij, ClearCLBuffer image, ClearCLBuffer psf, ClearCLBuffer dst, int iterations) {

        // the code here was inspired by
        // https://stackoverflow.com/questions/9854312/how-does-richardson-lucy-algorithm-work-code-example

        ClearCLBuffer est_conv = clij.createCLBuffer(image);
        ClearCLBuffer est_conv_min_1 = clij.createCLBuffer(image);
        ClearCLBuffer preliminary_dst = clij.createCLBuffer(image);
        ClearCLBuffer temp = clij.createCLBuffer(image);
        ClearCLBuffer error_est = clij.createCLBuffer(image);

        // initial guess
        Kernels.copy(clij, image, preliminary_dst);

        ClearCLBuffer psf_hat = clij.createCLBuffer(psf);

        if (psf.getDimension() == 2) {
            Kernels.flip(clij, psf, psf_hat, true, true);
        } else { // dimension = 3
            Kernels.flip(clij, psf, psf_hat, true, true, true);
        }

        for (int i = 0; i < iterations; i++) {
            /*
            ImagePlus intermediateResult = clij.pull(preliminary_dst);
            intermediateResult.show();
            if (i < 10) {
                IJ.saveAs("tiff", "C:/structure/data/decon/deconvolved_0" + i + ".tif");
            } else {
                IJ.saveAs("tiff", "C:/structure/data/decon/deconvolved_" + i + ".tif");
            }
            intermediateResult.close();
            */

            convolveWithCustomKernel(clij, preliminary_dst, psf, est_conv);

            // prevent division by zero by setting zero values to a small number
            if (est_conv.getNativeType() == NativeTypeEnum.Float) {
                Kernels.maximumImageAndScalar(clij, est_conv, est_conv_min_1, 0.0001f);
            } else {
                Kernels.maximumImageAndScalar(clij, est_conv, est_conv_min_1, 1f);
            }

            Kernels.divideImages(clij, image, est_conv_min_1, temp);

            convolveWithCustomKernel(clij, temp, psf_hat, error_est);

            Kernels.multiplyImages(clij, preliminary_dst, error_est, dst);

            if (i < iterations - 1) {
                Kernels.copy(clij, dst, preliminary_dst);
            }



        }

        psf_hat.close();
        est_conv.close();
        est_conv_min_1.close();
        preliminary_dst.close();
        temp.close();
        error_est.close();

        return true;
    }

    @Override
    public String getParameterHelpText() {
        return "Image source, Image convolution_kernel, Image destination, Number iterations";
    }

    @Override
    public String getDescription() {
        return "Richardson-Lucy implementation (experimental).";
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