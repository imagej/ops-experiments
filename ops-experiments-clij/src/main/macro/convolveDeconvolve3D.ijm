// convolveDecovolve3D.ijm
//
// Convolution and deconvolution using CLIJ
//
// Author: haesleinhuepf
// Jan 2019

psf_folder = "C:/structure/code/clij-custom-convolution-plugin/src/main/resources/"
run("Close All");

// get test data
newImage("spots", "32-bit black", 100, 100, 100);
Stack.setSlice(50);
makeRectangle(15, 35, 1, 1);
run("Add...", "value=255 slice");
makeRectangle(45, 35, 1, 1);
run("Add...", "value=255 slice");
makeRectangle(50, 35, 1, 1);
run("Add...", "value=255 slice");
makeRectangle(50, 70, 1, 1);
run("Add...", "value=255 slice");

// get custom convolution kernel
open(psf_folder + "PSF.TIF");
rename("kernelImage");
run("32-bit");

// init GPU
run("CLIJ Macro Extensions", "cl_device=");
Ext.CLIJ_clear();
Ext.CLIJ_push("spots");
Ext.CLIJ_push("kernelImage");

// normalize kernel
Ext.CLIJ_sumOfAllPixels("kernelImage");
sumPixels = getResult("Sum", nResults() - 1);
Ext.CLIJ_multiplyImageAndScalar("kernelImage", "normalizedKernel", 1.0 / sumPixels);

// convolve with normalized kernel
Ext.CLIJ_convolve("spots", "normalizedKernel", "convolved");

// show results
Ext.CLIJ_pull("normalizedKernel");
Ext.CLIJ_pull("convolved");
Stack.setSlice(50);

// deconvolve using Richardson-Lucy algorithm running for 8 iterations
Ext.CLIJ_deconvolve("convolved", "normalizedKernel", "deconvolved", 8);

// show results
Ext.CLIJ_pull("deconvolved");
Stack.setSlice(50);




