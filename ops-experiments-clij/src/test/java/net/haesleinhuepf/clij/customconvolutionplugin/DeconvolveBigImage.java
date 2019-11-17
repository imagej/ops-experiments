package net.haesleinhuepf.clij.customconvolutionplugin;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.NewImage;
import net.haesleinhuepf.clij.CLIJ;
import net.haesleinhuepf.clij.clearcl.ClearCLBuffer;
import net.haesleinhuepf.clij.clearcl.ClearCLImage;
import net.haesleinhuepf.clij.clearcl.enums.ImageChannelDataType;
import net.haesleinhuepf.clij.coremem.enums.NativeTypeEnum;
import net.haesleinhuepf.clij.kernels.Kernels;
import org.junit.Ignore;
import org.junit.Test;

/**
 * DeconvolveBigImage
 * <p>
 * Author: @haesleinhuepf
 * January 2019
 */
public class DeconvolveBigImage {
    @Ignore // because the image data is not available on github
    @Test
    public void testDeconvolutionWithBigImage() {
        ImagePlus psfImp = IJ.openImage("C:/structure/data/PSF_through_agarose-052_052_200.tif");
        IJ.run(psfImp, "32-bit", "");
        ImagePlus sampleImp = IJ.openImage("C:/structure/data/decon2_calibZapWFixed000282.tif");
        IJ.run(sampleImp, "32-bit", "");

        CLIJ clij = CLIJ.getInstance("1070");
        ClearCLBuffer sample = //clij.createCLBuffer(new long[]{512, 1024, 128}, NativeTypeEnum.Float);
                clij.convert(sampleImp, ClearCLBuffer.class);
        ClearCLBuffer psf = //clij.createCLBuffer(new long[]{15, 15, 15}, NativeTypeEnum.Float);
                clij.convert(psfImp, ClearCLBuffer.class);

        ClearCLBuffer result = clij.create(sample);

        long time = System.currentTimeMillis();
        //Kernels.copy(clij, sample, result);

        ClearCLBuffer normPSF = clij.create(psf);
        double sum = Kernels.sumPixels(clij, psf);

        Kernels.multiplyImageAndScalar(clij, psf, normPSF, (float)(1.0 / sum));
        //Kernels.blur(clij, sample, result, 12, 12, 12 , 3f,3f,3f);
        //Convolve.convolveWithCustomKernel(clij, sample, normPSF, result);
        Deconvolve.deconvolveWithCustomKernel(clij, sample, normPSF, result, 101);

        System.out.println("duration " + (System.currentTimeMillis() - time) + " msec");

        clij.show(result, "Result");

        System.out.println(IJ.getImage().getProcessor().getf(0,0));

        sample.close();
        result.close();
        psf.close();
    }
}
