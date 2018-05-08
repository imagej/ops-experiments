
package net.imagej.ops.experiments.testImages;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.Ops;
import net.imglib2.FinalInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

public class HalfBead<T extends RealType<T> & NativeType<T>> extends
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
		final String psfName = "../images/PSF-BeadStack-crop-64.tif";

		imgF = loadAndConvertToFloat(inputName, ij);
		psfF = loadPSFAndNormalize(psfName, ij);

		RandomAccessibleInterval<FloatType> rai =  ij.op().transform().crop(imgF, new FinalInterval(
			new long[] { 0, 0, 0 }, new long[] { imgF.dimension(0) - 1, imgF
				.dimension(1) - 1, 40 }));
		
		imgF=ij.op().create().img(rai);
		
		ij.op().copy().rai(imgF,rai);
	}

}
