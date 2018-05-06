
package net.imagej.ops.experiments.testImages;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imglib2.Dimensions;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public abstract class AbstractDeconvolutionTestData<T extends RealType<T> & NativeType<T>>
	implements DeconvolutionTestData
{

	protected Img<FloatType> loadAndConvertToFloat(String name, ImageJ ij)
		throws IOException
	{

		@SuppressWarnings("unchecked")
		final Img<T> img = (Img<T>) ij.dataset().open(name).getImgPlus().getImg();
		return ij.op().convert().float32(img);
	}

	protected Img<FloatType> loadPSFAndNormalize(String name, ImageJ ij)
		throws IOException
	{
		Img<FloatType> psf = loadAndConvertToFloat(name, ij);

		// normalize PSF
		final FloatType sum = new FloatType(ij.op().stats().sum(psf)
			.getRealFloat());
		return (Img<FloatType>) ij.op().math().divide(psf, sum);
	}

	protected Img<FloatType> createTheoreticalPSF(Dimensions psfDimensions,
		double numericalAperture, double wavelength, double riSample,
		double riImmersion, double xySpacing, double zSpacing, double depth, ImageJ ij)
	{
		// create the diffraction based psf
		Img<FloatType> psfF = ij.op().create().kernelDiffraction(psfDimensions,
			numericalAperture, wavelength, riSample, riImmersion, xySpacing, zSpacing,
			depth, new FloatType());

		// normalize PSF energy to 1
		float sumPSF = ij.op().stats().sum(psfF).getRealFloat();
		FloatType val = new FloatType();
		val.set(sumPSF);
		return (Img<FloatType>) ij.op().math().divide(psfF, val);

	}

}
