// convolveDecovolve.ijm
//
// Convolution and deconvolution using CLIJ
//
// Author: haesleinhuepf
// Jan 2019
run("Close All");

// get test data
open("C:/structure/data/blobs.gif");
// run("Blobs (25K)");
run("32-bit");

// get custom convolution kernel
newImage("kernelImage", "32-bit black", 25, 7, 1);
makeRectangle(0, 3, 25, 1);
run("Add...", "value=1.0");

// convolve in GPU
run("CLIJ Macro Extensions", "cl_device=[Intel(R) UHD Graphics 620]");
Ext.CLIJ_clear();
Ext.CLIJ_push("blobs.gif");
Ext.CLIJ_push("kernelImage");

// normalize kernel
Ext.CLIJ_sumOfAllPixels("kernelImage");
sumPixels = getResult("Sum", nResults() - 1);
Ext.CLIJ_multiplyImageAndScalar("kernelImage", "normalizedKernel", 1.0 / sumPixels);

// convolve
Ext.CLIJ_convolve("blobs.gif", "normalizedKernel", "convolved");
// show result
Ext.CLIJ_pull("convolved");
run("Invert LUT");

// deconvolve
Ext.CLIJ_deconvolve("convolved", "normalizedKernel", "deconvolved", 16);
// show result
Ext.CLIJ_pull("deconvolved");
run("Invert LUT");


