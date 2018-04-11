package net.imagej.ops.experiments.kernel;

import java.io.IOException;

import net.imagej.ImageJ;
import net.imagej.ops.experiments.kernel.WidefieldKernel;
import net.imglib2.FinalDimensions;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;
import net.imagej.ops.experiments.kernel.GibsonLanni;

import ij.ImageStack;

public class InteractiveKernelTest {

	final static ImageJ ij = new ImageJ();

	public static <T extends RealType<T> & NativeType<T>> void main(final String[] args) throws IOException {

		ij.launch(args);

		WidefieldKernel wfk = new WidefieldKernel();
/*
		wfk.setDEPTH(12000);
		wfk.setIndexImmersion(1.514);
		wfk.setIndexSp(1.37);

		Img widefieldKernel = wfk.compute(ij.op());

		GibsonLanni gl = new GibsonLanni();

		gl.setpZ(-10000e-9);
		gl.setNz(128);

		long startTime = System.currentTimeMillis();

		gl.setNumBasis(100);
		gl.setNumSamp(1000);
		Img psf3d = gl.compute(ij.op());
		
		gl.setpZ(10000e-9);
		Img psf3d2 = gl.compute(ij.op());
		*/
		
		Img widefieldKernel=(Img)ij.op().create().kernelDiffraction(new FinalDimensions(64,64,100), 1.3, 550E-09, 1.3, 1.5, 100E-9, 250E-9, 0, new FloatType());
		//Img widefieldKernel=(Img)ij.op().create().kernelDiffraction(new FinalDimensions(256,256,100), 1.4, 610E-09, 1.3, 1.5, 100E-9, 250E-9, -10000e-9, new FloatType());
		
		//Img widefieldKernel2=(Img)ij.op().create().kernelDiffraction(new FinalDimensions(256,256,128), 1.4, 610E-09, 1.3, 1.5, 100E-9, 250E-9, 10000e-9, new FloatType());
		
		ij.ui().show(widefieldKernel);
		//ij.ui().show(widefieldKernel2);
		
		//ij.ui().show(psf3d);
		//ij.ui().show(psf3d2);
	}

}
