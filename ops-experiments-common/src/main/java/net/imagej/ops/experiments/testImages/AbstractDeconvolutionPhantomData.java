
package net.imagej.ops.experiments.testImages;

import net.imagej.ImageJ;
import net.imglib2.Dimensions;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public abstract class AbstractDeconvolutionPhantomData<T extends RealType<T> & NativeType<T>>
	implements DeconvolutionTestData
{

	protected Img<FloatType> createTheoreticalPSF(Dimensions psfDimensions,
		double numericalAperture, double wavelength, double riSample,
		double riImmersion, double xySpacing, double zSpacing, double depth,
		ImageJ ij)
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
