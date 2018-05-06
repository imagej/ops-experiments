
package net.imagej.ops.experiments.testImages;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imglib2.Dimensions;
import net.imglib2.FinalDimensions;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class CElegans<T extends RealType<T> & NativeType<T>> extends
	AbstractDeconvolutionTestData<T>
{

	private Img<FloatType> imgF;
	private Img<FloatType> psfF;

	@Override
	public void LoadImages(ImageJ ij) throws IOException {
		final String inputName = "../images/CElegans-CY3-crop.tif";
		imgF = loadAndConvertToFloat(inputName, ij);

		Dimensions psfDimensions = new FinalDimensions(65, 65, 128);

		double numericalAperture = 1.4;
		double wavelength = 654E-09;
		double riImmersion = 1.5f;
		double riSample = 1.4f;
		double xySpacing = 64.5E-9;
		double zSpacing = 160E-9;
		double depth = 0;
	
		// create the diffraction based psf
		psfF = createTheoreticalPSF(psfDimensions, numericalAperture, wavelength,
			riSample, riImmersion, xySpacing, zSpacing, depth, ij);

	}

	@Override
	public RandomAccessibleInterval<FloatType> getImg() {
		return imgF;
	}

	@Override
	public RandomAccessibleInterval<FloatType> getPSF() {
		return psfF;
	}

}
