package net.imagej.ops.experiments.filter.deconvolve;

import net.imagej.Dataset;
import net.imagej.ImageJ;

public class InteractiveCudaTheoreticalTest {
	public static void main(final String... args) throws Exception {
		// create the ImageJ application context with all available services
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();

		ij.log().error("show me");
		
		// load the dataset
		Dataset dataset = (Dataset) ij.io().open("../../images/Slide_17015-02-1.tif");
		Dataset psf= (Dataset) ij.io().open("../images/PSF-Bars-stack-cropped.tif");
		
		ij.ui().show(dataset);
		
		ij.command().run(YacuDecuRichardsonLucyTheoreticalPSFCommand.class, true, "img", dataset);
	
	
	}
}
