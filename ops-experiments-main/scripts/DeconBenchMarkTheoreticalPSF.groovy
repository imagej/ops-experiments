// @OpService ops
// @UIService ui
// @ImgPlus img
// @Integer numIterations(value=100)
// @Float numericalAperture(value=1.4)
// @Float wavelength(value=700)
// @Float riImmersion(value=1.5)
// @Float riSample(value=1.4)
// @Float xySpacing(value=62.9)
// @Float zSpacing(value=160)
// @OUTPUT ImgPlus psf
// @OUTPUT ImgPlus deconvolved

import net.imglib2.FinalDimensions
import net.imglib2.type.numeric.real.FloatType;

import net.imagej.ops.experiments.filter.deconvolve.YacuDecuRichardsonLucyOp;
import net.imagej.ops.experiments.filter.deconvolve.MKLRichardsonLucyOp;

// convert to float (TODO: make sure deconvolution op works on other types)
imgF=ops.convert().float32(img)

// make psf same size as image
psfSize=new FinalDimensions(img.dimension(0), img.dimension(1), img.dimension(2));

// add border in z direction
borderSize=[0,0,0] as long[];

wavelength=wavelength*1E-9
xySpacing=xySpacing*1E-9
zSpacing=zSpacing*1E-9

riImmersion = 1.5;
riSample = 1.4;
xySpacing = 62.9E-9;
zSpacing = 160E-9;
depth = 0;

psf = ops.create().kernelDiffraction(psfSize, numericalAperture, wavelength,
				riSample, riImmersion, xySpacing, zSpacing, depth, new FloatType());

startTime = System.currentTimeMillis();

deconvolved=ops.run(YacuDecuRichardsonLucyOp.class, imgF, psf, borderSize, numIterations);
//deconvolved=ops.run(MKLRichardsonLucyOp.class, imgF, psf, borderSize, numIterations);

//deconvolved =  ops.deconvolve().richardsonLucy(imgF, psf, borderSize, null,
	//				null, null, null, numIterations, false, false);

endTime = System.currentTimeMillis();

print "Total execution time  is: " + (endTime - startTime);

//deconvolved = ops.deconvolve().richardsonLucy(imgF, psf, borderSize, None,
//					None, None, None, numIterations, True, True);
