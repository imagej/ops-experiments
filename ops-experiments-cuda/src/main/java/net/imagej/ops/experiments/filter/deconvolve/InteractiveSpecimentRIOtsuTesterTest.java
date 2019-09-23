
package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ops.Ops;
import net.imagej.ops.experiments.testImages.Bead;
import net.imagej.ops.experiments.testImages.BeedPhantom;
import net.imagej.ops.experiments.testImages.DeconvolutionTestData;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;

public class InteractiveSpecimentRIOtsuTesterTest {

	public static void main(final String... args) throws Exception {
		// create the ImageJ application context with all available services
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		ij.log().error("show me");

		DeconvolutionTestData testData = new Bead("../images/");

//		testData.LoadImages(ij);
//		RandomAccessibleInterval<FloatType> imgF = testData.getImg();
		
		DeconvolutionTestData beadData = new BeedPhantom();

		beadData.LoadImages(ij);

		ij.ui().show(beadData.getImg());

		ij.command().run(YacuDecuSpecimenRIOtsuTester.class, true, "img",
			beadData.getImg());

	}
}
