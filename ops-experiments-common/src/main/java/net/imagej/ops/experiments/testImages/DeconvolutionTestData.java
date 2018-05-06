
package net.imagej.ops.experiments.testImages;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.real.FloatType;

public interface DeconvolutionTestData {

	void LoadImages(ImageJ ij) throws IOException;

	RandomAccessibleInterval<FloatType> getImg();

	RandomAccessibleInterval<FloatType> getPSF();

}
