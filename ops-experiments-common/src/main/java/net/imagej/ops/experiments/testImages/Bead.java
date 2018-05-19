
package net.imagej.ops.experiments.testImages;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class Bead<T extends RealType<T> & NativeType<T>> extends
	AbstractDeconvolutionTestData<T>
{

	private Img<FloatType> imgF;
	private Img<FloatType> psfF;

	@Override
	public RandomAccessibleInterval<FloatType> getImg() {
		return imgF;
	}

	@Override
	public RandomAccessibleInterval<FloatType> getPSF() {
		return psfF;
	}

	@Override
	public void LoadImages(ImageJ ij) throws IOException {
		final String inputName = "../images/BeadStack-crop.tif";
		final String psfName = "../images/PSF-BeadStack-crop.tif";

		imgF = loadAndConvertToFloat(inputName, ij);
		psfF = loadPSFAndNormalize(psfName, ij);

	}

}
