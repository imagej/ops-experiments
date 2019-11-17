// convolveDecovolve3D.ijm
//
// Convolution and deconvolution using CLIJ
//
// Author: haesleinhuepf
// Jan 2019

run("CLIJ Macro Extensions", "cl_device=1070");

for (i = 1; i < 50; i++) {
	run("Close All");
	open("C:/structure/data/benchm_20190104/PSF3.tif");
	run("32-bit");
	rename("psf");
	open("C:/structure/data/benchm_20190104/decon3_hisYFP_SPIM.tif");
	//makeRectangle(0, 0, 512, 256);
	//run("Crop");
	run("32-bit");
	rename("sample");
	
	// convolve in GPU
	Ext.CLIJ_clear();
	
	Ext.CLIJ_push("sample");
	Ext.CLIJ_push("psf");
	
	// normalize kernel
	//Ext.CLIJ_sumOfAllPixels("psf");
	//sumPixels = getResult("Sum", nResults() - 1);
	//Ext.CLIJ_multiplyImageAndScalar("psf", "normalizedPSF", 1.0 / sumPixels);
	
	Ext.CLIJ_copy("psf", "normalizedPSF");
	Ext.CLIJ_copy("sample", "convolved");
	
	Ext.CLIJ_reportMemory();
	
	// deconvolve
	time = getTime();
	Ext.CLIJ_deconvolve("convolved", "normalizedPSF", "deconvolved", i);
	deltaTime = getTime() - time;
	IJ.log("duration: " + deltaTime);
	
	// show results
	Ext.CLIJ_pull("deconvolved");
	Stack.setSlice(50);
	if (i < 10) {
		saveAs("Tiff", "C:/structure/data/benchm_20190104/decon3/deconvolved_0" + i + ".tif");
	}
	else {
		saveAs("Tiff", "C:/structure/data/benchm_20190104/decon3/deconvolved_" + i + ".tif");	
	}
}



